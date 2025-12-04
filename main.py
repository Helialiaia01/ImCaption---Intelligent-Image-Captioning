"""
Inference script for Image Captioning.
Generate captions for images using beam search.
"""

import os
import cv2
import torch
import json
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Generate caption for an image using beam search.
    
    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model  
        image_path: Path to the image file
        word_map: Word to index mapping
        beam_size: Number of beams for beam search
    
    Returns:
        seq: Generated caption sequence (word indices)
        alphas: Attention weights for visualization
    """
    k = beam_size
    vocab_size = len(word_map)

    # Read and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    img = cv2.resize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([normalize])
    image = transform(img)

    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
        
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar
        awe = gate * awe
        
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size

        # Add new words to sequences
        prev_word_inds = prev_word_inds.long()
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                          next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        
        k -= len(complete_inds)

        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
            break
        step += 1

    if len(complete_seqs_scores) == 0:
        return seqs[0].tolist(), seqs_alpha[0].tolist()
    
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_attention(image_path, seq, alphas, rev_word_map, word_map, smooth=True):
    """
    Visualize attention weights on the image.
    
    Args:
        image_path: Path to the image
        seq: Generated caption sequence
        alphas: Attention weights
        rev_word_map: Index to word mapping
        word_map: Word to index mapping
        smooth: Whether to apply gaussian smoothing
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))

    words = [rev_word_map[ind] for ind in seq if ind not in 
             {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
    
    # Filter alphas to match words
    filtered_alphas = []
    for idx, ind in enumerate(seq):
        if ind not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}:
            filtered_alphas.append(alphas[idx])

    plt.figure(figsize=(15, 15))
    
    for t, (word, alpha) in enumerate(zip(words, filtered_alphas)):
        if t >= 25:  # Limit display
            break
        
        plt.subplot(5, 5, t + 1)
        plt.text(0, 1, f'{word}', color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        
        alpha = np.array(alpha)
        if smooth:
            alpha = cv2.resize(alpha, (256, 256))
            alpha = cv2.GaussianBlur(alpha, (13, 13), 0)
        else:
            alpha = cv2.resize(alpha, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        plt.imshow(alpha, alpha=0.7, cmap='jet')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate caption for an image')
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to the image')
    parser.add_argument('--model', '-m', type=str, default='Models/Version01_BEST_Captioning_model.pth.tar',
                        help='Path to the trained model')
    parser.add_argument('--word_map', '-w', type=str, default='json_data/word_map.json',
                        help='Path to the word map JSON file')
    parser.add_argument('--beam_size', '-b', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize attention weights')
    
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = torch.load(args.model, map_location=str(device))
    
    decoder = model['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    
    encoder = model['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    # Generate caption
    print(f"Generating caption for {args.image}...")
    seq, alphas = caption_image(encoder, decoder, args.image, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Print caption
    caption = ' '.join([rev_word_map[ind] for ind in seq if ind not in 
                        {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
    print(f"\nPredicted caption: {caption}")

    # Visualize attention if requested
    if args.visualize:
        visualize_attention(args.image, seq, alphas.numpy(), rev_word_map, word_map)


if __name__ == '__main__':
    main()
