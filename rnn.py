""" The RNN Decoder Model """

import torch
import torch.nn as nn
from seq2seq import Decoder


class RNN(Decoder):
    """ RNN Model """

    def __init__(self, vocab_size, embedding_dim, d_model, num_layers=2):
        super().__init__()

        # Hyperparamters
        self.num_layers = num_layers
        self.d_model = d_model

        # Paramters
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.output_layer = nn.Linear(d_model, vocab_size)

        # Hidden State. Variable to store the
        self.hidden_state = None

    def forward(
        self, encoder_output, decoder_input, predict=False, max_target_length=30
    ):
        """ Forward pass """
        batch_size = decoder_input.shape[0]
        shape = (self.num_layers, batch_size, self.d_model)
        self.reset_state(shape, decoder_input.device)
        output = super().forward(
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

    def reset_state(self, shape, device):
        """ Reset the LSTM state """
        self.hidden_state = (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )
