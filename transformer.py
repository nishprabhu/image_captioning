""" The Transfomer Decoder model """


import torch
import torch.nn as nn
from seq2seq import Decoder


class Transformer(Decoder):
    """ The transformer decoder """

    def __init__(
        self, vocab_size, embedding_dim=300, d_model=420, num_layers=6, nhead=6
    ):
        super().__init__()
        # Embedding Layer
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Using nn.TransformerEncoderLayer because we do not want encoder-decoder attention
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward_step(self, features, text, features_mask=None):
        """ A single forward step """
        # Embedding
        output = self.embedding_layer(text)

        # Transformer
        seq_length = text.shape[1]
        features = features.unsqueeze(1)
        output = torch.cat([output, features.repeat(1, seq_length, 1)], dim=-1)
        output = output.permute(1, 0, 2)
        mask = self.create_autoregressive_mask(seq_length, output.device)
        output = self.transformer(output, mask=mask, src_key_padding_mask=features_mask)
        output = output.permute(1, 0, 2)

        # Linear Layer
        output = self.output_layer(output)

        return output

    def create_autoregressive_mask(self, size, device):
        """ Returns an autoregressive mask of shape (size, size) on the specified device. """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask.to(device=device)
