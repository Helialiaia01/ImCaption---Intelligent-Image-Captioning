"""
Modern Image Captioning using BLIP (Bootstrapped Language-Image Pre-training)

BLIP is a state-of-the-art vision-language model that can:
- Caption any type of image (photos, cartoons, anime, memes, art)
- Understand colors, emotions, and complex scenes
- Answer questions about images (Visual Question Answering)

Model: Salesforce/blip-image-captioning-large
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import argparse
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model cache
_processor = None
_model = None


def load_blip_model():
    """Load the BLIP model (downloads ~1.5GB on first run)."""
    global _processor, _model
    
    if _processor is None or _model is None:
        print("Loading BLIP model (this may take a minute on first run)...")
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device)
        _model.eval()
        print(f"✓ Model loaded on {device}")
    
    return _processor, _model


def caption_image(image_path, conditional_text=None):
    """
    Generate a caption for an image.
    
    Args:
        image_path: Path to the image file
        conditional_text: Optional text prompt to guide the caption
                         e.g., "a photograph of" or "this is"
    
    Returns:
        Generated caption string
    """
    processor, model = load_blip_model()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Process image
    if conditional_text:
        # Conditional captioning - guide the model with a prompt
        inputs = processor(image, conditional_text, return_tensors="pt").to(device)
    else:
        # Unconditional captioning
        inputs = processor(image, return_tensors="pt").to(device)
    
    # Generate caption
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def caption_with_questions(image_path):
    """
    Generate multiple captions with different prompts to get richer descriptions.
    """
    prompts = [
        None,                           # Default caption
        "a photograph of",              # Photo-style
        "this image shows",             # Descriptive
        "the main subject is",          # Focus on subject
        "the colors in this image are", # Color focus
    ]
    
    print(f"\nAnalyzing: {image_path}\n")
    print("-" * 60)
    
    for prompt in prompts:
        caption = caption_image(image_path, prompt)
        if prompt:
            print(f"[{prompt}...] {caption}")
        else:
            print(f"[Default] {caption}")
    
    print("-" * 60)


def compare_models(image_path):
    """
    Compare the old COCO model with the new BLIP model.
    """
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\nImage: {image_path}\n")
    
    # BLIP caption
    print("🆕 BLIP Model (state-of-the-art):")
    blip_caption = caption_image(image_path)
    print(f"   {blip_caption}\n")
    
    # Old model caption (if available)
    try:
        from demo import load_model, load_word_map, caption_from_file
        print("📦 Original COCO Model:")
        encoder, decoder = load_model()
        word_map, rev_word_map = load_word_map()
        old_caption = caption_from_file(encoder, decoder, word_map, rev_word_map, image_path)
        print(f"   {old_caption}\n")
    except Exception as e:
        print(f"   (Could not load old model: {e})\n")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='BLIP Image Captioning')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to the image file')
    parser.add_argument('--prompt', '-p', type=str, default=None,
                        help='Optional prompt to guide caption (e.g., "a photo of")')
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Generate multiple captions with different prompts')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare BLIP with the original COCO model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    if args.compare:
        compare_models(args.image)
    elif args.detailed:
        caption_with_questions(args.image)
    else:
        caption = caption_image(args.image, args.prompt)
        print(f"\nCaption: {caption}")


if __name__ == '__main__':
    main()
