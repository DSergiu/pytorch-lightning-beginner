import sys

sys.path.append('../src')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseLightningModule import BaseLightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from torch import optim


class VariationalAutoEncoder(BaseLightningModule):
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
        self.fc_mu = nn.Linear(196, self.hparams.latent_variables)
        self.fc_logvar = nn.Linear(196, self.hparams.latent_variables)
        self.dec_fc = nn.Linear(self.hparams.latent_variables, 196)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.example_input_array = torch.rand(self.hparams.batch_size, 1, 28, 28)

    def encode(self, x):
        nr_batches = x.shape[0]
        x = self.encoder(x)
        x = x.view(nr_batches, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)  # Use a Normal Gaussian to distribute the latent variables evenly
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

    def loss_function(self, x, x_hat, mu, logvar):
        loss = F.binary_cross_entropy(x_hat.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss + kl_divergence

    def _share_step(self, batch, prefix, log_acc=True):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss = self.loss_function(x, x_hat, mu, logvar)
        self.log(f'{prefix}_loss', loss, logger=True, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args():
        parser = BaseLightningModule.add_model_specific_args()
        parser.add_argument('--latent_variables', default=5, type=int, help='the number of latent variables to learn')
        return parser


def main(args):
    model = VariationalAutoEncoder(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd(), name='lightning_logs/')
    trainer.logger = tb_logger
    trainer.fit(model=model)
    trainer.test(model=model)


if __name__ == '__main__':
    parser = VariationalAutoEncoder.add_model_specific_args()
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(accelerator='gpu', devices=1, check_val_every_n_epoch=5, learning_rate=0.001)
    args = parser.parse_args()
    main(args)
