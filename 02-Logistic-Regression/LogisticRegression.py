import sys

sys.path.append('../src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseLightningModule import BaseLightningModule
from pytorch_lightning import Trainer
from torch import optim


class LogisticRegression(BaseLightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10)
        )
        self.example_input_array = torch.rand(1, 1, 28 * 28)

    def forward(self, x):
        return self.seq(x)

    def _share_step(self, batch, prefix, log_acc=True):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(f'{prefix}_loss', loss, logger=True, prog_bar=True, on_epoch=True)
        if log_acc:
            # Get prediction for each label
            labels_hat = torch.argmax(y_hat, dim=1)
            # Calculate accuracy: how many are correct divided by the number of tests
            test_acc = torch.true_divide(torch.sum(y == labels_hat), len(x))
            self.log(f'{prefix}_acc', test_acc, logger=True, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def main(args):
    model = LogisticRegression(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    parser = LogisticRegression.add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(accelerator='gpu', devices=1, check_val_every_n_epoch=5)
    args = parser.parse_args()
    main(args)
