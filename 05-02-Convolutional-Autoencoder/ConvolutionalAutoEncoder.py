import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.BaseLightningModule import BaseLightningModule
from pytorch_lightning import Trainer
from torch import optim


class ConvolutionalAutoEncoder(BaseLightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.example_input_array = torch.rand(self.hparams.batch_size, 1, 28, 28)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _share_step(self, batch, prefix, log_acc=True):
        x, _ = batch
        # Pass image through model and get a similar image back
        x_hat = self(x)
        # Compare initial and generated images
        loss = F.binary_cross_entropy(x_hat, x)
        self.log(f'{prefix}_loss', loss, logger=True, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def main(args):
    model = ConvolutionalAutoEncoder(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    parser = ConvolutionalAutoEncoder.add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(accelerator='gpu', devices=1, check_val_every_n_epoch=5, learning_rate=0.001)
    args = parser.parse_args()
    main(args)
