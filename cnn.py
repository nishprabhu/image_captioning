""" The CNN Encoder Model """

import torch.nn as nn
from torchvision.models import resnet18


class CNN(nn.Module):
    """ The CNN Model """

    def __init__(self, encoder_output_dim):
        super().__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.intermediate = nn.Linear(512, encoder_output_dim)

    def forward(self, image):
        """ Forward function  """
        output = self.cnn(image)
        output = self.intermediate(output.squeeze())
        return output
