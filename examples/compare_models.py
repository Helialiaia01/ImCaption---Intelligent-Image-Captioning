"""
Example: Compare COCO Model vs BLIP

This script shows the difference between the custom-trained COCO model
and the state-of-the-art BLIP model.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compare_models(image_path):
    """Compare both models on the same image."""
    
    print("=" * 60)
    print("Model Comparison: COCO vs BLIP")
    print("=" * 60)
    print(f"\nImage: {image_path}\n")
    
    # COCO Model (custom trained)
    print("📦 COCO Model (ResNet + LSTM + Attention):")
    print("   Training: 118K COCO images, custom training")
    try:
        from demo import load_model, load_word_map, caption_from_file
        encoder, decoder = load_model()
        word_map, rev_word_map = load_word_map()
        coco_caption = caption_from_file(encoder, decoder, word_map, rev_word_map, image_path)
        print(f"   Caption: {coco_caption}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # BLIP Model (pre-trained)
    print("🆕 BLIP Model (Vision Transformer + Language Model):")
    print("   Training: 129M image-text pairs by Salesforce")
    try:
        from blip_caption import caption_image
        blip_caption = caption_image(image_path)
        print(f"   Caption: {blip_caption}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("Key Differences:")
    print("  • COCO: Best for everyday real-world photos")
    print("  • BLIP: Works on cartoons, art, memes, anything")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True, help="Path to image")
    args = parser.parse_args()
    
    compare_models(args.image)
