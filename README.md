# ImCaption - Intelligent Image Captioning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.9+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Year-2024-yellow.svg" alt="Year">
</p>

<p align="center">
  <b>Generating natural language descriptions from images using deep learning</b><br>
  <i>AI Lab Project • ACSAI • La Sapienza University of Rome • 2024</i>
</p>

---

## Overview

This project implements an encoder-decoder architecture with attention for automatic image captioning. Given an input image, the model generates a natural language description of its content.

The system combines:
- **ResNet-101** for visual feature extraction
- **Attention mechanism** for focusing on relevant image regions
- **LSTM** for sequential caption generation
- **Beam search** for improved inference

We also provide an integration with BLIP for comparison with modern vision-language models.

## Project Structure

```
Image-Captioning/
├── models.py           # Encoder, Attention, Decoder architectures
├── dataset.py          # PyTorch Dataset for COCO
├── train.py            # Training and validation script
├── main.py             # Inference with beam search
├── demo.py             # Easy-to-use demo script
├── blip_caption.py     # Modern BLIP model integration
├── utils.py            # Helper functions
├── json_generator.py   # COCO dataset preprocessing
├── requirements.txt    # Python dependencies
├── config.ini.example  # Configuration template
├── json_data/          # Preprocessed data (download required)
│   ├── word_map.json
│   ├── IMAGES_VAL.hdf5
│   ├── Encoded_Captions_*.json
│   └── Captions_len_*.json
└── Models/             # Trained models (download required)
    └── *.pth.tar
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Image-Captioning.git
cd Image-Captioning
```

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Pre-trained Models & Data

Download from Google Drive and place in the respective folders:

| File | Link | Location |
|------|------|----------|
| Trained Models | [Download](https://drive.google.com/drive/folders/1-aMdOn1tAiPzCP6YAn-4befzBLsgV-d2) | `Models/` |
| Dataset Files | [Download](https://drive.google.com/drive/folders/1-3zBO-JLRrRJNFLyYbbLITnkA0h9UmyC) | `json_data/` |

### 4. Run Demo

```bash
# Caption validation images
python demo.py --num_samples 5

# Caption your own image
python demo.py --image path/to/your/image.jpg
```

## Usage Examples

### Basic Captioning (COCO Model)

```python
from demo import load_model, load_word_map, caption_from_file

encoder, decoder = load_model()
word_map, rev_word_map = load_word_map()

caption = caption_from_file(encoder, decoder, word_map, rev_word_map, "image.jpg")
print(f"Caption: {caption}")
```

### Advanced Captioning with Beam Search

```bash
python main.py --image image.jpg --beam_size 5 --visualize
```

### Modern BLIP Model (Recommended for diverse images)

```bash
# Simple caption
python blip_caption.py --image image.jpg

# Detailed analysis
python blip_caption.py --image image.jpg --detailed

# Compare both models
python blip_caption.py --image image.jpg --compare
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      IMAGE CAPTIONING                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [Image] ──→ [ResNet-101 Encoder] ──→ [14×14×2048 Features]│
│                                              ↓              │
│                                    [Attention Mechanism]    │
│                                              ↓              │
│   [<start>] ──→ [LSTM Decoder] ──→ [word₁] ──→ ... ──→ [<end>]│
│                                                             │
│   Output: "a dog playing with a frisbee in the park"       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **Encoder** | Pre-trained ResNet-101, extracts visual features | `models.py:Encoder` |
| **Attention** | Soft attention over image regions | `models.py:Attention` |
| **Decoder** | LSTM with attention for text generation | `models.py:DecoderWithAttention` |

## Training

### Prerequisites
- COCO 2017 dataset (~20GB)
- GPU recommended (training takes several hours)

### Steps

1. Download COCO 2017 dataset from [cocodataset.org](https://cocodataset.org/#download)

2. Create configuration file:
```bash
cp config.ini.example config.ini
# Edit config.ini with your paths
```

3. Generate preprocessed data:
```bash
python json_generator.py
```

4. Train the model:
```bash
python train.py
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emb_dim` | 512 | Word embedding dimension |
| `attention_dim` | 512 | Attention layer dimension |
| `decoder_dim` | 512 | LSTM hidden state dimension |
| `dropout` | 0.5 | Dropout rate |
| `batch_size` | 32 | Training batch size |
| `epochs` | 10 | Number of training epochs |
| `encoder_lr` | 1e-4 | Encoder learning rate |
| `decoder_lr` | 4e-4 | Decoder learning rate |

## Results

We evaluated our model on the COCO validation set:

| Metric | Value |
|--------|-------|
| BLEU-4 | ~0.25 |
| Top-5 Accuracy | ~70% |

## BLIP Integration

In addition to our custom-trained model, we integrated BLIP (Bootstrapped Language-Image Pre-training) to compare traditional encoder-decoder approaches with modern vision-language models. While our CNN+LSTM model performs well on real-world photographs similar to the COCO dataset, BLIP generalizes better to diverse image types including illustrations and artwork.

## Technical Details

### Encoder (ResNet-101)
The encoder uses a pre-trained ResNet-101 to extract visual features from input images. The network outputs a 14×14 grid of 2048-dimensional feature vectors, providing a spatial representation of the image content.

### Attention Mechanism
The attention module allows the decoder to focus on different image regions when generating each word. This improves caption quality and provides interpretability—we can visualize which parts of the image the model attended to for each word.

### Decoder (LSTM)
The LSTM decoder generates captions word by word, using the attention-weighted image features and its internal memory state. The gates (forget, input, output) help maintain relevant information across the sequence.

### Beam Search
At inference time, beam search explores multiple candidate sequences simultaneously, keeping the top-k hypotheses at each step. This typically produces more coherent captions than greedy decoding.

## Authors

This project was developed as part of the **AI Lab** course in the ACSAI program (Applied Computer Science and Artificial Intelligence) at La Sapienza University of Rome, 2024.

## License

This project is open source under the MIT License — feel free to use, modify, and learn from it!

## Acknowledgments

We built upon the work of many researchers and open-source contributors:

- **[COCO Dataset](https://cocodataset.org/)** — The foundation for training and evaluation
- **[Show, Attend and Tell](https://arxiv.org/abs/1502.03044)** — The paper that introduced attention for image captioning
- **[Salesforce BLIP](https://github.com/salesforce/BLIP)** — State-of-the-art vision-language model
- **[PyTorch](https://pytorch.org/)** & **[Hugging Face](https://huggingface.co/)** — Amazing deep learning tools

## References

1. Lin, T.Y., et al. "Microsoft COCO: Common Objects in Context." *ECCV 2014*
2. He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR 2016*
3. Xu, K., et al. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." *ICML 2015*
4. Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training." *ICML 2022*

---

<p align="center">
  <b>La Sapienza University of Rome • ACSAI • 2024</b><br>
  Made with ❤️ for learning AI
</p>
