from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.core import LightningModule
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(LightningModule):
    def __init__(self,
                 data_root: str = '../datasets',
                 batch_size: int = 64,
                 learning_rate: float = 0.01,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(True),
            nn.Linear(300, 100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.ReLU(True)
        )
        self.example_input_array = torch.rand(1, 1, 28 * 28)

    def forward(self, x):
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.TrainResult(loss, checkpoint_on=loss)
        result.log('train_loss', loss, logger=True, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        return self._share_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._share_step(batch, 'test')

    def _share_step(self, batch, prefix):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Get prediction for each label
        labels_hat = torch.argmax(y_hat, dim=1)
        # Calculate accuracy: how many are correct divided by the number of tests
        test_acc = torch.true_divide(torch.sum(y == labels_hat), len(x))

        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f'{prefix}_loss', loss, logger=True, prog_bar=True, on_epoch=True)
        result.log(f'{prefix}_acc', test_acc, logger=True, prog_bar=True, on_epoch=True)
        return result

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

    def prepare_data(self):
        datasets.MNIST(self.hparams.data_root, train=True, download=True)
        datasets.MNIST(self.hparams.data_root, train=False, download=True)

    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(self.hparams.data_root, train=True, download=False, transform=transform)
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(train, [50000, 10000])
        self.mnist_test = datasets.MNIST(self.hparams.data_root, train=False, download=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, num_workers=4)


def main(args):
    model = NeuralNetwork(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

    # Manually call test which runs the test loop and logs accuracy and loss
    trainer.test()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
