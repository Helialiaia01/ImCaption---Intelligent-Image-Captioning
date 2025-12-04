"""
Example: Basic Image Captioning Usage

This script demonstrates how to use the trained model for image captioning.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import load_model, load_word_map, caption_from_validation_set

def main():
    print("=" * 60)
    print("Image Captioning - Basic Example")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    encoder, decoder = load_model()
    word_map, rev_word_map = load_word_map()
    print(f"   ✓ Model loaded (vocabulary: {len(word_map)} words)")
    
    # Generate captions for validation images
    print("\n2. Generating captions for sample images:\n")
    
    for i in range(5):
        caption = caption_from_validation_set(
            encoder, decoder, word_map, rev_word_map, 
            image_index=i
        )
        print(f"   Image {i+1}: {caption}")
    
    print("\n" + "=" * 60)
    print("Done! Try with your own images using:")
    print("  python demo.py --image your_image.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()
