""" Dataset and collate function """

import os
import random
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import BertTokenizer


class Dataset(CocoCaptions):
    """ Extends the CocoCaptions class with a custom transform """

    def __init__(self, root, annFile, tokenizer):
        self.tokenizer = tokenizer
        transforms = Compose([Resize((400, 400)), ToTensor()])
        super().__init__(root, annFile, transform=transforms)

    def __getitem__(self, idx):
        image, captions = super().__getitem__(idx)
        caption = random.choice(captions)
        caption = self.tokenizer.encode(caption, add_special_tokens=True)
        caption = torch.tensor(caption)
        return image, caption


class TestDataset:
    """ Dataset to return the preprocessed test image and the start tokens """

    def __init__(self, root, tokenizer, start_token_id):
        self.root = root
        self.images = os.listdir(root)
        self.tokenizer = tokenizer
        self.start_token_id = start_token_id

    def __getitem__(self, idx):
        # Image
        image_path = os.path.join(self.root, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        transforms = Compose([Resize((400, 400)), ToTensor()])
        image = transforms(image)

        # Caption
        caption = torch.zeros(1, dtype=torch.long) + self.start_token_id
        return image, caption

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    """ Collate Function: Helps batch data of irregular sizes """
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions
