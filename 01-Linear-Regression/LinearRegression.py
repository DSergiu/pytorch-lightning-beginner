from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.core import LightningModule
from torch import optim
from torch.utils.data import DataLoader


class LinearRegression(LightningModule):
    def __init__(self,
                 learning_rate: float = 0.05,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.lin = nn.Linear(1, 1)
        self.example_input_array = torch.rand(1, 1)

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(loss, checkpoint_on=loss)
        result.log('train_loss', loss, logger=True, on_epoch=True)
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

    def setup(self, stage):
        x_train = torch.tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                            [9.779], [6.182], [7.59], [2.167], [7.042],
                            [10.791], [5.313], [7.997], [3.1]], dtype=torch.float32)
        y_train = torch.tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                            [3.366], [2.596], [2.53], [1.221], [2.827],
                            [3.465], [1.65], [2.904], [1.3]], dtype=torch.float32)
        self.data_train = [[x_train, y_train]]

    def train_dataloader(self):
        return DataLoader(self.data_train, num_workers=4)


def main(args):
    model = LinearRegression(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
