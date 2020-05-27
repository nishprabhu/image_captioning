""" Contains the models and dataloaders for image captioning  """


import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.models import resnet18
from transformers import BertTokenizer
from seq2seq import Decoder


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


class CaptioningModel(nn.Module):
    """ Image captioning model """

    def __init__(self, vocab_size, embedding_dim, encoder_output_dim):
        super().__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.intermediate = nn.Linear(512, 120)
        self.rnn = RNN(vocab_size, embedding_dim, encoder_output_dim)

    def forward(self, images, captions, predict=False):
        """ Forward pass of the image captioning model """
        features = self.cnn(images)
        features = self.intermediate(features.squeeze())
        captions = self.rnn(features, captions, predict)
        return captions


class RNN(Decoder):
    """ RNN Model """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        encoder_output_dim,
        num_layers=2,
        max_target_length=30,
    ):
        super().__init__()

        # Hyperparamters
        self.max_target_length = max_target_length
        self.total_size = embedding_dim + encoder_output_dim
        self.num_layers = num_layers

        # Paramters
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            self.total_size, self.total_size, num_layers, batch_first=True
        )
        self.output_layer = nn.Linear(self.total_size, vocab_size)

        # Hidden State. Variable to store the
        self.hidden_state = None

    def forward(
        self, encoder_output, decoder_input, predict=False, max_target_length=30
    ):
        """ Forward pass """
        batch_size = decoder_input.shape[0]
        shape = (self.num_layers, batch_size, self.total_size)
        self.reset_state(shape)
        output = super(RNN, self).forward(
            encoder_output, decoder_input, predict, max_target_length
        )
        return output

    def forward_step(self, features, text):
        """ A single time-step """

        # Embedding
        output = self.embedding_layer(text)

        # RNN
        seq_length = text.shape[1]
        features = features.unsqueeze(1)
        output = torch.cat([output, features.repeat(1, seq_length, 1)], dim=-1)
        output, self.hidden_state = self.lstm(output, self.hidden_state)

        # Linear Layer
        output = self.output_layer(output)

        return output

    def reset_state(self, shape):
        """ Reset the LSTM state """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        self.hidden_state = (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )


def main():
    """ Main function """
    root = "coco_dataset/train2017"
    ann_file = "coco_dataset/annotations/captions_train2017.json"
    dataset = Dataset(root, ann_file)
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)

    vocab_size = dataset.tokenizer.vocab_size
    embedding_dim = 300
    encoder_output_dim = 120

    model = CaptioningModel(vocab_size, embedding_dim, encoder_output_dim)

    for batch in dataloader:
        images, captions = batch
        print(images.shape)
        print(captions.shape)

        output = model(images, captions, predict=True)
        print(output.shape)
        break


if __name__ == "__main__":
    main()
