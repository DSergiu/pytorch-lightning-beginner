import torch
from argparse import ArgumentParser
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BaseLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        return self._share_step(batch, 'train', log_acc=False)

    def validation_step(self, batch, batch_idx):
        return self._share_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._share_step(batch, 'test')

    def _share_step(self, batch, prefix, log_acc=True):
        raise Exception("Not implemented!")

    def prepare_data(self):
        datasets.MNIST(self.hparams.data_root, train=True, download=True)
        datasets.MNIST(self.hparams.data_root, train=False, download=True)

    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor()])
        if stage == 'fit':
            train = datasets.MNIST(self.hparams.data_root, train=True, download=False, transform=transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(train, [50000, 10000])
        elif stage == 'test':
            self.mnist_test = datasets.MNIST(self.hparams.data_root, train=False, download=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train,
                          persistent_workers=True,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val,
                          persistent_workers=True,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test,
                          persistent_workers=True,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    @staticmethod
    def add_model_specific_args():
        parser = ArgumentParser()
        parser.add_argument('--data_root', default='../datasets', type=str, help='path to dataset')
        parser.add_argument('--learning_rate', default=0.01, type=float, help='the learning rate')
        parser.add_argument('--batch_size', default=64, type=int, help='the batch size')
        parser.add_argument('--num_workers', default=4, type=int, help='the number of processes used for data loading')
        return parser
