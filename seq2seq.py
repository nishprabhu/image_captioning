""" Encoder and Decoder wrappers written in PyTorch 1.5.0 """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """ Encoder base class """

    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, *kwargs):
        """ Forward pass """
        raise NotImplementedError


class Decoder(nn.Module):
    """ Decoder base class """

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(
        self,
        encoder_output,
        decoder_input,
        predict=False,
        max_target_length=30,
        **kwargs
    ):
        """
        Forward pass

        Arguments:
        ---------
        encoder_output: Tensor of shape (batch_size, e, embedding_size)
        decoder_input: Tensor of shape (batch_size, d)
                       Provide a vector of shape (batch_size, 1) consisting of <START> tokens for prediction.
        predict: Boolean tensor. Use teacher forcing if False.
        max_target_length: Integer indicating the number of time-steps to be run during inference.
        """
        if not predict:
            outputs = self.forward_step(encoder_output, decoder_input, **kwargs)
        else:
            previous_outputs = [decoder_input[:, 0]]
            current_input = torch.stack(previous_outputs, dim=-1)
            for _ in range(max_target_length):
                current_output = self.forward_step(
                    encoder_output, current_input, **kwargs
                )
                indices = torch.argmax(F.softmax(current_output, dim=-1), dim=-1)[
                    :, -1:
                ]
                current_input = torch.cat([current_input, indices], dim=-1)
            outputs = current_output

        return outputs

    def forward_step(self, encoder_output, decoder_input, **kwargs):
        """
        A single forward step

        Arguments:
        ---------
        encoder_output: Tensor of shape (batch_size, e, embedding_size)
        decoder_input: Tensor of shape (batch_size, d)
        [e and d are encoder and decoder sequence lengths respectively.]

        Output:
        ------
        outputs: Tensor of shape (batch_size, d, vocabulary_size)
        """
        raise NotImplementedError
