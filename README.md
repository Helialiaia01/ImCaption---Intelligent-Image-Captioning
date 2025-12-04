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

## 🎯 What is this project?

Have you ever wondered how AI can "see" an image and describe it in words? This project explores exactly that!

We built an **image captioning system** that takes any photograph and generates a descriptive sentence about its content. For example, given a photo of a park, it might output: *"a dog playing with a frisbee in the grass"*.

### How it works

Our model combines two powerful neural network architectures:

1. **CNN (ResNet-101)** — Acts as the "eyes", extracting visual features from images
2. **LSTM with Attention** — Acts as the "brain", generating words one by one while focusing on relevant image regions

We also integrated **BLIP**, a modern vision-language model, for comparison and enhanced capabilities.

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

We evaluated our model using standard image captioning metrics:

| Metric | Value | What it measures |
|--------|-------|------------------|
| BLEU-4 | ~0.25 | N-gram overlap with reference captions |
| Top-5 Accuracy | ~70% | Correct word in top 5 predictions |

## 🔄 Our Model vs BLIP

We also integrated BLIP (a state-of-the-art model) for comparison:

| Aspect | Our COCO Model | BLIP |
|--------|----------------|------|
| **Training Data** | 118K COCO images | 129M image-text pairs |
| **Best For** | Real-world photos | Any image (cartoons, art, memes) |
| **Educational Value** | ✅ You can understand every component | Black box |
| **Speed** | Faster | Slower (larger model) |

## 📚 Under the Hood

### The Encoder: ResNet-101
We use a pre-trained ResNet-101 as our "visual feature extractor". Think of it as the model's eyes — it looks at the image and creates a rich numerical representation (14×14 grid of 2048-dimensional vectors). The **residual connections** in ResNet allow us to use a very deep network without the gradients vanishing during training.

### The Attention Mechanism  
This is the key innovation! Instead of looking at the entire image equally, the model learns to **focus on different regions** when generating each word. When saying "dog", it looks at the dog. When saying "frisbee", it shifts attention to the frisbee. This makes the captions more accurate and the model more interpretable.

### The Decoder: LSTM
The LSTM (Long Short-Term Memory) network generates the caption word by word. It maintains a "memory" of what it has said so far, and uses the attention-weighted image features to decide the next word. The **gates** (forget, input, output) help it remember important information over long sequences.

### Beam Search
During inference, instead of always picking the most likely next word (greedy), we keep track of the **top-k best sequences** at each step. This often produces more fluent and accurate captions.

## 👥 Authors

This project was developed as part of the **AI Lab** course in the ACSAI program (Applied Computer Science and Artificial Intelligence) at La Sapienza University of Rome, 2024.

## 📄 License

This project is open source under the MIT License — feel free to use, modify, and learn from it!

## 🙏 Acknowledgments

We built upon the work of many researchers and open-source contributors:

- **[COCO Dataset](https://cocodataset.org/)** — The foundation for training and evaluation
- **[Show, Attend and Tell](https://arxiv.org/abs/1502.03044)** — The paper that introduced attention for image captioning
- **[Salesforce BLIP](https://github.com/salesforce/BLIP)** — State-of-the-art vision-language model
- **[PyTorch](https://pytorch.org/)** & **[Hugging Face](https://huggingface.co/)** — Amazing deep learning tools

## 📖 References

1. Lin, T.Y., et al. "Microsoft COCO: Common Objects in Context." *ECCV 2014*
2. He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR 2016*
3. Xu, K., et al. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." *ICML 2015*
4. Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training." *ICML 2022*

---

<p align="center">
  <b>La Sapienza University of Rome • ACSAI • 2024</b><br>
  Made with ❤️ for learning AI
</p>
