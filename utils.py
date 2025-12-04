import numpy as np
import torch
import json
import os


def save_json(json_data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        f.write(json.dumps(json_data))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients to prevent gradient explosion.
    Keeps gradients in range (-grad_clip, grad_clip).
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_model(epoch, epochs_since_improvement, encoder, decoder, decoder_optimizer, bleu4, is_best, model_dir='Models'):
    """
    Save model checkpoint.
    
    Args:
        epoch: Current epoch number
        epochs_since_improvement: Number of epochs since last improvement
        encoder: Encoder model
        decoder: Decoder model
        decoder_optimizer: Optimizer for decoder
        bleu4: BLEU-4 score
        is_best: Boolean indicating if this is the best model so far
        model_dir: Directory to save models
    """
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'decoder_optimizer': decoder_optimizer
    }
    
    os.makedirs(model_dir, exist_ok=True)
    filename = os.path.join(model_dir, 'Captioning_model.pth.tar')
    torch.save(state, filename)
    
    # If this is the best checkpoint, save a copy
    if is_best:
        torch.save(state, os.path.join(model_dir, 'BEST_Captioning_model.pth.tar'))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    Used for tracking statistics during training.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    Helps model converge and improve accuracy.
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy of the model's predictions.
    
    Args:
        scores: Model predictions
        targets: Ground truth labels
        k: Top-k value
    
    Returns:
        Top-k accuracy as a percentage
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def init_embedding(embeddings):
    """
    Initialize embedding weights with uniform distribution.
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map.
    
    Args:
        emb_file: File containing embeddings (GloVe format)
        word_map: Word to index mapping
    
    Returns:
        embeddings: Tensor of embeddings
        emb_dim: Embedding dimension
    """
    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')
        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in vocabulary
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim
