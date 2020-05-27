""" Contains the Encoder-Decoder model for image captioning  """

import torch.nn as nn
from cnn import CNN
from rnn import RNN
from transformer import Transformer


class CaptioningModel(nn.Module):
    """ Image captioning model """

    def __init__(
        self, encoder_output_dim, decoder_type, vocab_size, embedding_dim, num_layers,
    ):
        super().__init__()

        # CNN Model
        self.cnn = CNN(encoder_output_dim)

        # Decoder (RNN or Transformer)
        d_model = embedding_dim + encoder_output_dim
        if decoder_type == "rnn":
            self.decoder = RNN(vocab_size, embedding_dim, d_model, num_layers)
        else:
            self.decoder = Transformer(vocab_size, embedding_dim, d_model, num_layers)

    def forward(self, images, captions, predict=False):
        """ Forward pass of the image captioning model """
        features = self.cnn(images)
        captions = self.decoder(features, captions, predict)
        return captions
