""" Dataset and collate function """

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import BertTokenizer


class Dataset(CocoCaptions):
    """ Extends the CocoCaptions class with a custom transform """

    def __init__(self, root, annFile):
        transforms = Compose([Resize((400, 400)), ToTensor()])
        super().__init__(root, annFile, transform=transforms)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):
        image, captions = super().__getitem__(idx)
        caption = random.choice(captions)
        caption = self.tokenizer.encode(caption, add_special_tokens=True)
        caption = torch.tensor(caption)
        return image, caption


def collate_fn(batch):
    """ Collate Function: Helps batch data of irregular sizes """
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions
