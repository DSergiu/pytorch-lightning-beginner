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


class VariationalAutoEncoder(LightningModule):
    def __init__(self,
                 data_root: str = '../datasets',
                 batch_size: int = 64,
                 latent_variables: int = 20,
                 learning_rate: float = 0.01,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc_mu = nn.Linear(196, latent_variables)
        self.fc_logvar = nn.Linear(196, latent_variables)
        self.dec_fc = nn.Linear(latent_variables, 196)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.example_input_array = torch.rand(batch_size, 1, 28, 28)

    def encode(self, x):
        nr_batches = x.shape[0]
        x = self.encoder(x)
        x = x.view(nr_batches, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma) # Use a Normal Gaussian to distribute the latent variables evenly
        return eps.mul(sigma).add_(mu)

    def decode(self, z):
        z = self.dec_fc(z)
        z = z.view(-1, 4, 7, 7)
        z = self.decoder(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.hparams.latent_variables)
        samples = self.decode(z)
        return samples

    def loss_function(self, x, x_hat, mu, logvar):
        loss = F.binary_cross_entropy(x_hat.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss + kl_divergence

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, sigma = self(x)
        loss = self.loss_function(x, x_hat, mu, sigma)
        result = pl.TrainResult(loss, checkpoint_on=loss)
        result.log('train_loss', loss, logger=True, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        return self._share_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._share_step(batch, 'test')

    def _share_step(self, batch, prefix):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss = self.loss_function(x, x_hat, mu, logvar)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f'{prefix}_loss', loss, logger=True, prog_bar=True, on_epoch=True)
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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

    @staticmethod
    def add_model_specific_args():
        parser = ArgumentParser()
        parser.add_argument('--latent_variables', default=5, type=int, help='the number of latent variables to learn')
        parser.add_argument('--learning_rate', default=0.001, type=float, help='the learning rate')
        return parser

def main(args):
    model = VariationalAutoEncoder(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

    # Manually call test which runs the test loop and logs accuracy and loss
    trainer.test()


if __name__ == '__main__':
    parser = VariationalAutoEncoder.add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
