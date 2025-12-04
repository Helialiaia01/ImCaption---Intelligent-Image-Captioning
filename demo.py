"""
Demo script for Image Captioning.
This script provides an easy way to test the model with sample images.
"""

import torch
import json
import h5py
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path='Models/Version01_BEST_Captioning_model.pth.tar'):
    """Load the trained encoder and decoder models."""
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    
    encoder = model['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    
    decoder = model['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    
    return encoder, decoder


def load_word_map(word_map_path='json_data/word_map.json'):
    """Load the word map."""
    with open(word_map_path, 'r') as f:
        word_map = json.load(f)
    rev_word_map = {v: k for k, v in word_map.items()}
    return word_map, rev_word_map


def generate_caption_greedy(encoder, decoder, image_tensor, word_map, rev_word_map, max_len=50):
    """Generate a caption using greedy decoding."""
    with torch.no_grad():
        encoder_out = encoder(image_tensor)
        
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        
        h_state, c_state = decoder.init_hidden_state(encoder_out)
        
        prev_word = torch.LongTensor([[word_map['<start>']]]).to(device)
        
        caption = []
        for step in range(max_len):
            embeddings = decoder.embedding(prev_word).squeeze(1)
            awe, alpha = decoder.attention(encoder_out, h_state)
            gate = decoder.sigmoid(decoder.f_beta(h_state))
            awe = gate * awe
            h_state, c_state = decoder.decode_step(
                torch.cat([embeddings, awe], dim=1), 
                (h_state, c_state)
            )
            scores = decoder.fc(h_state)
            
            _, predicted = scores.max(1)
            word_idx = predicted.item()
            
            if word_idx == word_map['<end>']:
                break
            
            if word_idx not in [word_map['<start>'], word_map['<pad>'], word_map['<end>']]:
                caption.append(rev_word_map[word_idx])
            
            prev_word = predicted.unsqueeze(0)
        
        return ' '.join(caption)


def caption_from_validation_set(encoder, decoder, word_map, rev_word_map, 
                                 image_index=0, hdf5_path='json_data/IMAGES_VAL.hdf5'):
    """Generate caption for an image from the validation set."""
    with h5py.File(hdf5_path, 'r') as h:
        img = h['images'][image_index]  # (3, 256, 256)
        img = img / 255.0
        img = torch.FloatTensor(img).to(device)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        img = (img - mean) / std
        img = img.unsqueeze(0)
        
        return generate_caption_greedy(encoder, decoder, img, word_map, rev_word_map)


def caption_from_file(encoder, decoder, word_map, rev_word_map, image_path):
    """Generate caption for an image file."""
    import cv2
    import torchvision.transforms as transforms
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    img = cv2.resize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.0
    img = torch.FloatTensor(img).to(device)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img = normalize(img)
    img = img.unsqueeze(0)
    
    return generate_caption_greedy(encoder, decoder, img, word_map, rev_word_map)


def demo_validation_images(num_images=5):
    """Demo: Generate captions for sample images from validation set."""
    print("=" * 60)
    print("IMAGE CAPTIONING DEMO")
    print("=" * 60)
    print(f"\nDevice: {device}")
    
    # Load model and word map
    encoder, decoder = load_model()
    word_map, rev_word_map = load_word_map()
    print(f"Vocabulary size: {len(word_map)} words")
    
    print(f"\nGenerating captions for {num_images} validation images:\n")
    print("-" * 60)
    
    for i in range(num_images):
        caption = caption_from_validation_set(
            encoder, decoder, word_map, rev_word_map, image_index=i
        )
        print(f"Image {i+1}: {caption}")
    
    print("-" * 60)
    print("\nDemo complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Captioning Demo')
    parser.add_argument('--image', '-i', type=str, help='Path to an image file')
    parser.add_argument('--num_samples', '-n', type=int, default=5, 
                        help='Number of validation images to caption')
    parser.add_argument('--model', '-m', type=str, 
                        default='Models/Version01_BEST_Captioning_model.pth.tar',
                        help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    if args.image:
        # Caption a specific image
        encoder, decoder = load_model(args.model)
        word_map, rev_word_map = load_word_map()
        
        print(f"\nGenerating caption for: {args.image}")
        caption = caption_from_file(encoder, decoder, word_map, rev_word_map, args.image)
        print(f"Caption: {caption}")
    else:
        # Run demo with validation images
        demo_validation_images(args.num_samples)
