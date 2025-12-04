"""
JSON Generator for COCO Dataset Preprocessing.
This script processes the COCO dataset to create the required JSON files for training.
"""

import os
import json
import cv2
import h5py
import numpy as np
import configparser
import pandas as pd
from tqdm import tqdm
from collections import Counter
from random import choice, sample
from nltk.tokenize import word_tokenize


def read_file(file_path):
    """Read a JSON file and return its contents."""
    with open(file_path, 'r') as f:
        COCO = json.load(f)
    return COCO


def save_json(json_data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        f.write(json.dumps(json_data))


def tokenize_captions(caption):
    """Tokenize a caption into words."""
    tokens = word_tokenize(caption.lower())
    tokens = [word for word in tokens if word.isalpha()]
    return tokens


def make_json_structure(info, dataset_path):
    """
    Create the main dataset structure from COCO annotations.
    
    Args:
        info: Dictionary with 'train' and 'val' paths to COCO annotation files
        dataset_path: Path to save the output dataset JSON
    """
    rows = []
    for key, value in info.items():
        COCO = read_file(value)
        
        COCO_df = pd.DataFrame(COCO["annotations"])
        images = pd.DataFrame(COCO["images"])
        image_id_to_filename = dict(zip(images['id'], images['file_name']))
        
        for image_id, group in COCO_df.groupby("image_id"):
            print(image_id, key)
            row = {}
            row["filename"] = image_id_to_filename[image_id]
            row["sentences_ids"] = [id for id in group["id"]]
            row["split"] = key
            row["image_id"] = image_id
            row["sentences"] = []
            
            for _, annotation in group.iterrows():
                sentence = {
                    "tokens": tokenize_captions(annotation["caption"]),
                    "raw": annotation["caption"],
                    "sentence_id": annotation["id"],
                    "img_id": image_id,
                }
                row["sentences"].append(sentence)
            
            rows.append(row)
    
    print("Number of pictures inside the dataset:", len(rows))
    print("Number of captions inside the dataset:", sum([len(row["sentences"]) for row in rows]))
    
    res = {"images": rows}
    save_json(res, dataset_path)


def create_files(word_map_path, json_dataset_path, image_folder, base_path,
                 captions_for_image=5, min_word_freq=3, max_len=100):
    """
    Create training files: HDF5 images, encoded captions, and word map.
    
    Args:
        word_map_path: Path to save the word map
        json_dataset_path: Path to the dataset JSON file
        image_folder: Folder containing COCO images
        base_path: Base path for output files
        captions_for_image: Number of captions per image
        min_word_freq: Minimum word frequency for vocabulary
        max_len: Maximum caption length
    """
    with open(json_dataset_path, 'r') as j:
        data = json.load(j)

    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        if img['split'] in {'train'}:
            train_image_paths.append(os.path.join(image_folder, "train2017", img['filename']))
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(os.path.join(image_folder, "val2017", img['filename']))
            val_image_captions.append(captions)

    # Verify list lengths
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    print("Vocabulary size:", len(words))
    save_json(word_map, word_map_path)
    print("Word map saved to:", word_map_path)

    # Create HDF5 files and encoded captions
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                    (val_image_paths, val_image_captions, 'VAL')]:

        with h5py.File(os.path.join(base_path, "IMAGES_" + split + '.hdf5'), 'a') as h:
            h.attrs['captions_for_image'] = captions_for_image
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("Reading %s images and captions, storing to file" % split)

            encoded_captions = []
            caption_lens = []

            for i, path in enumerate(tqdm(impaths)):
                # Sample captions
                if len(imcaps[i]) < captions_for_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_for_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_for_image)

                assert len(captions) == captions_for_image

                # Read and preprocess image
                img = cv2.imread(impaths[i])
                if img is None:
                    print(f"Warning: Could not read image {impaths[i]}")
                    continue
                    
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                
                img = cv2.resize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                images[i] = img

                # Encode captions
                for j, c in enumerate(captions):
                    encoded_caption = [word_map['<start>']] + \
                                     [word_map.get(word, word_map['<unk>']) for word in c] + \
                                     [word_map['<end>']] + \
                                     [word_map['<pad>']] * (max_len - len(c))

                    caption_len = len(c) + 2

                    encoded_captions.append(encoded_caption)
                    caption_lens.append(caption_len)

            assert images.shape[0] * captions_for_image == len(encoded_captions) == len(caption_lens)

            # Save encoded captions and lengths
            save_json(encoded_captions, os.path.join(base_path, "Encoded_Captions_" + split + '.json'))
            save_json(caption_lens, os.path.join(base_path, "Captions_len_" + split + '.json'))


if __name__ == '__main__':
    # Configuration - Update these paths according to your setup
    config = configparser.ConfigParser()
    
    # Check if config file exists
    if os.path.exists('config.ini'):
        config.read('config.ini')
        coco_caption_val_path = config['coco_path']['val_captions']
        coco_caption_train_path = config['coco_path']['train_captions']
        dataset_path = config['json_path']['json_dataset']
        word_map_path = config['json_path']['word_map_path']
        image_folder = config['coco_path']['image_folder']
        base_path = config['json_path']['base_path_json']
    else:
        # Default paths - update these for your system
        print("No config.ini found. Using default paths.")
        print("Please create a config.ini file or update the paths below.")
        
        # Example paths - UPDATE THESE
        coco_caption_val_path = 'path/to/annotations/captions_val2017.json'
        coco_caption_train_path = 'path/to/annotations/captions_train2017.json'
        dataset_path = 'json_data/dataset.json'
        word_map_path = 'json_data/word_map.json'
        image_folder = 'path/to/coco/images'
        base_path = 'json_data'

    info = {
        "val": coco_caption_val_path,
        "train": coco_caption_train_path,
    }

    make_json_structure(info, dataset_path)
    create_files(word_map_path, dataset_path, image_folder, base_path, 5, 3)
