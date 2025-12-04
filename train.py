"""
Training script for the Image Captioning model.
Trains an encoder-decoder model with attention mechanism.
"""

import time
import os
import json
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from dataset import CaptionDataset
from utils import AverageMeter, accuracy, adjust_learning_rate, clip_gradient, save_model
from nltk.translate.bleu_score import corpus_bleu

# Configuration
data_folder = 'json_data'  # Folder containing the JSON data files
model_folder = 'Models'    # Folder to save trained models

# Model parameters
emb_dim = 512           # Embedding dimension
attention_dim = 512     # Attention dimension
decoder_dim = 512       # Decoder dimension
dropout = 0.5           # Dropout rate

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

start_epoch = 0
epochs = 10
epochs_since_improvement = 0
batch_size = 32
workers = 0  # Number of data loading workers
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5.0
alpha_c = 1.0
best_bleu4 = 0.0
print_freq = 100
checkpoint = None  # Path to checkpoint, None if starting from scratch


def train(train_loader, encoder, decoder, criterion, decoder_optimizer, epoch, word_map):
    """
    Performs one epoch of training.
    """
    track = {}
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward pass
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        # Pack sequences
        scores, _, *extra = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _, *extra = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Backward pass
        decoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)

        decoder_optimizer.step()

        # Track metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            track[i] = (losses.avg, top5accs.avg)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top5=top5accs))
    return track


def validate(val_loader, encoder, decoder, criterion, word_map):
    """
    Performs one epoch of validation.
    """
    track = {}
    decoder.eval()

    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # True captions
    hypotheses = list()  # Predicted captions

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores, _, *extra = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _, *extra = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            track[i] = (losses.avg, top5accs.avg)
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      i, len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top5=top5accs))

            # Store references
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)

            # Store hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)

        print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4, track


def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch

    # Load word map
    word_map_file = os.path.join(data_folder, 'word_map.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize or load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(word_map),
            dropout=dropout
        )
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr
        )
        encoder = Encoder()
    else:
        checkpoint_data = torch.load(checkpoint)
        start_epoch = checkpoint_data['epoch'] + 1
        epochs_since_improvement = checkpoint_data['epochs_since_improvement']
        best_bleu4 = checkpoint_data['bleu-4']
        decoder = checkpoint_data['decoder']
        decoder_optimizer = checkpoint_data['decoder_optimizer']
        encoder = checkpoint_data['encoder']

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # Data transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    # Training loop
    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            break

        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # Train for one epoch
        track_train = train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
            word_map=word_map
        )

        # Validate
        recent_bleu4, track_val = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            word_map=word_map
        )

        # Check if this is the best model
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_model(
            epoch, epochs_since_improvement, encoder, decoder,
            decoder_optimizer, recent_bleu4, is_best, model_folder
        )
        print("Model saved")
        print("Loss and accuracy for train:", track_train)
        print("Loss and accuracy for val:", track_val)


if __name__ == '__main__':
    main()
