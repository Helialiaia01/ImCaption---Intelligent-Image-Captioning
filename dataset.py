import os
import h5py
import json
import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """
    Dataset class for loading and preprocessing image captioning data.
    Loads images from HDF5 files and captions from JSON files.
    """

    def __init__(self, data_folder, split, transform=None):
        self.split = split
        self.h = h5py.File(os.path.join(data_folder, 'IMAGES_' + self.split + '.hdf5'), 'r')
        self.imgs = self.h['images']
        self.cpi = self.h.attrs['captions_for_image']

        with open(os.path.join(data_folder, 'Encoded_Captions_' + self.split + '.json'), 'r') as j:
            self.captions = json.load(j)

        with open(os.path.join(data_folder, 'Captions_len_' + self.split + '.json'), 'r') as j:
            self.caplens = json.load(j)

        self.transform = transform
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # The Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation/testing, also return all captions to calculate BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
