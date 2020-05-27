""" Lightining Module for Image Captioning. """

import os
from collections import OrderedDict
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule

from captioning_model import CaptioningModel
from dataset import Dataset, collate_fn
from utils import get_text


class ImageCaptioning(LightningModule):
    """ The lightning class for covid projection. """

    def __init__(self, hparams):
        # init superclass
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = tokenizer.vocab_size
        self.model = CaptioningModel(
            hparams.encoder_output_dim,
            hparams.decoder_type,
            vocab_size,
            hparams.embedding_dim,
            hparams.num_decoder_layers,
        )

    def forward(self):
        pass

    def loss(self, outputs, captions):
        """ Method to compute cross entropy loss  """
        vocab_size = outputs.shape[-1]
        outputs = outputs.view(-1, vocab_size)
        captions = captions.view(-1)
        ce_loss = F.cross_entropy(outputs, captions, ignore_index=0)
        return ce_loss

    def training_step(self, batch, batch_idx):
        """ Training Step """
        # forward pass
        images, captions = batch
        outputs = self.model(images, captions)

        # calculate loss
        loss_val = self.loss(outputs, captions)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """ Validation Step """
        images, captions = batch
        outputs = self.model(images, captions)

        # calculate loss
        loss_val = self.loss(outputs, captions)

        output = OrderedDict({"val_loss": loss_val})
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

        val_loss_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def test_step(self, batch, batch_idx):
        """ Testing Step """
        images, captions = batch
        outputs = self.model(images, captions)

        # Get token indices
        captions = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
        output = {"captions": captions}
        return output

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """

        captions = []
        for output in outputs:
            captions.append(output["captions"])
        captions = torch.cat(captions, dim=0)

        captions = get_text(captions.cpu().numpy())

        # Write outputs to disk
        with open("outputs.txt", "w") as file:
            for caption in captions:
                file.write(caption)
                file.write("\n")

        # create output dict
        tqdm_dict = {}
        results = {}
        results["progress_bar"] = tqdm_dict
        results["log"] = tqdm_dict

        return results

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def __dataloader(self, split):
        root = os.path.join(self.hparams.data_root, split)
        annotations_filename = "captions_" + split + ".json"
        annotations_path = os.path.join(
            self.hparams.data_root, "annotations", annotations_filename
        )
        dataset = Dataset(root=root, annFile=annotations_path)
        # when using multi-node (ddp) we need to add the  datasampler
        batch_size = self.hparams.batch_size
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_fn
        )
        return loader

    def train_dataloader(self):
        """ Training DataLoader """
        log.info("Training data loader called.")
        return self.__dataloader("train2017")

    def val_dataloader(self):
        """ Validation DataLoader """
        log.info("Validation data loader called.")
        return self.__dataloader("val2017")

    def test_dataloader(self):
        """ Testing DataLoader """
        log.info("Test data loader called.")
        return self.__dataloader("test2017")

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through `self.hparams`.
        """
        parser = ArgumentParser(parents=[parent_parser])

        # CNN Hyperparams
        parser.add_argument("--encoder_output_dim", default=120, type=int)

        # Decoder Hyperparams
        parser.add_argument("--decoder_type", default="transformer", type=str)
        parser.add_argument("--embedding_dim", default=300, type=int)
        parser.add_argument("--num_decoder_layers", default=6, type=int)

        # Training Hyperparams
        parser.add_argument("--drop_prob", default=0.2, type=float)
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--epochs", default=30, type=int)
        parser.add_argument("--batch_size", default=64, type=int)

        # Data Path
        parser.add_argument(
            "--data_root",
            default=os.path.join(root_dir, "../image_captioning/coco_dataset"),
            type=str,
        )

        return parser
