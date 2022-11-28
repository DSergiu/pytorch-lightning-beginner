import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.BaseLightningModule import BaseLightningModule
from pytorch_lightning import Trainer
from torch import optim


class SimpleAutoEncoder(BaseLightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )
        self.example_input_array = torch.rand(self.hparams.batch_size, 1, 28, 28)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)
        return x

    def _share_step(self, batch, prefix, log_acc=True):
        x, _ = batch
        # Pass image through model and get a similar image back
        x_hat = self(x)
        # Compare initial and generated images
        loss = F.mse_loss(x_hat, x)
        self.log(f'{prefix}_loss', loss, logger=True, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


def main(args):
    model = SimpleAutoEncoder(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    parser = SimpleAutoEncoder.add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(accelerator='gpu', devices=1, check_val_every_n_epoch=5, learning_rate=0.001)
    args = parser.parse_args()
    main(args)
