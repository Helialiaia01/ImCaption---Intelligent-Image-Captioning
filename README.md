# ImCaption - Intelligent Image Captioning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.9+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/University-La%20Sapienza-orange.svg" alt="University">
</p>

<p align="center">
  <b>A deep learning model for generating descriptive captions from images</b><br>
  <i>AI Lab Project - ACSAI, La Sapienza University of Rome</i>
</p>

---

## 🎯 Overview

This project implements an **encoder-decoder architecture with attention mechanism** for automatic image captioning. The model uses:

- **ResNet-101** (CNN) for visual feature extraction
- **Attention Mechanism** for focusing on relevant image regions
- **LSTM** for sequential caption generation
- **Beam Search** for improved inference

Additionally, we provide a **BLIP integration** for state-of-the-art captioning on any image type.

## 📁 Project Structure

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

## 🚀 Quick Start

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

## 💡 Usage Examples

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

## 🏗️ Architecture

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

## 📊 Training

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

## 📈 Results

| Metric | Value |
|--------|-------|
| BLEU-4 | ~0.25 |
| Top-5 Accuracy | ~70% |

## 🔄 Model Comparison

| Model | Training Data | Strengths | Weaknesses |
|-------|---------------|-----------|------------|
| **COCO Model** (ours) | 118K images | Educational, interpretable | Limited to real photos |
| **BLIP** (integrated) | 129M images | Works on anything | Black box |

## 📚 Technical Details

### Why ResNet-101?
- **Residual connections** prevent vanishing gradients
- **Pre-trained on ImageNet** (transfer learning)
- Outputs rich 2048-dimensional feature vectors

### Why Attention?
- Allows decoder to **focus on relevant image regions**
- Each word attends to different parts of the image
- Improves caption quality and interpretability

### Why LSTM?
- **Long Short-Term Memory** handles sequential data
- Gates (forget, input, output) manage information flow
- Generates coherent multi-word captions

### Beam Search
- Explores multiple candidate sequences
- Keeps top-k hypotheses at each step
- Produces better captions than greedy decoding

## 👥 Authors

- **ACSAI Students** - La Sapienza University of Rome

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [COCO Dataset](https://cocodataset.org/)
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) - Original attention paper
- [Salesforce BLIP](https://github.com/salesforce/BLIP) - State-of-the-art model
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 📖 References

1. Lin, T.Y., et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.
2. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
3. Xu, K., et al. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." ICML 2015.
4. Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training." ICML 2022.

---

<p align="center">
  Made with ❤️ at La Sapienza University of Rome
</p>
