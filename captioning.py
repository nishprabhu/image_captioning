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

from modeling import Dataset, CaptioningModel, collate_fn


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
            vocab_size, hparams.embedding_dim, hparams.encoder_output_dim
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

        ## Write code to get captions from model outputs
        return outputs

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """

        predictions = []
        cases = []
        geoids = []
        mean = []
        std = []
        for output in outputs:
            predictions.append(output["predictions"])
            cases.append(output["cases"])
            geoids.append(output["geoids"])
            mean.append(output["mean"])
            std.append(output["std"])
        predictions = torch.cat(predictions, dim=0)
        cases = torch.cat(cases, dim=0)
        geoids = torch.cat(geoids, dim=0).squeeze()
        mean = torch.cat(mean, dim=0).squeeze()
        std = torch.cat(std, dim=0).squeeze()

        # Un-normalize the data
        predictions = (predictions * std.unsqueeze(1)) + mean.unsqueeze(1)
        cases = (cases * std.unsqueeze(1)) + mean.unsqueeze(1)

        predictions[predictions < 0] = 0

        # compute mean absolute error
        mae = F.l1_loss(predictions, cases)

        # Write outputs to disk
        columns = list(range(1, 8))
        predictions = pd.DataFrame(data=predictions.numpy(), columns=columns)
        cases = pd.DataFrame(data=cases.numpy(), columns=columns)
        geoids = geoids.numpy().squeeze().tolist()
        predictions.insert(0, "geoid", geoids)
        cases.insert(0, "geoid", geoids)
        predictions.to_csv("cases_predictions.csv", header=True, index=False)
        cases.to_csv("cases.csv", header=True, index=False)

        # create output dict
        tqdm_dict = {"mae": mae}
        results = {}
        results["progress_bar"] = tqdm_dict
        results["log"] = tqdm_dict
        results["mae"] = mae

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

        # network params
        parser.add_argument("--embedding_dim", default=300, type=int)
        parser.add_argument("--encoder_output_dim", default=120, type=int)
        parser.add_argument("--num_rnn_layers", default=2, type=int)
        parser.add_argument("--drop_prob", default=0.2, type=float)
        parser.add_argument("--learning_rate", default=0.001, type=float)

        # data
        parser.add_argument(
            "--data_root", default=os.path.join(root_dir, "coco_dataset"), type=str
        )

        # training params (opt)
        parser.add_argument("--epochs", default=30, type=int)
        parser.add_argument("--batch_size", default=64, type=int)
        return parser
