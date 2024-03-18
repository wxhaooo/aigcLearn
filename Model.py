import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.module import T
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import time


# vae for mnist dataset
class VAE(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 400, latent_dim: int = 200, device: str = 'cuda'):
        super(VAE, self).__init__()

        self.device = device

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # encoder产生z
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )

        # 然后产生p(z|x)的均值和方差
        # 图像是二维灰度图，所以方差和均值dim = 2
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.log_var_layer = nn.Linear(latent_dim, 2)

        # decoder根据p(x|z)产生新的采样
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        return self.decoder(x)

    def mean_log_var(self, x):
        mean, log_var = self.mean_layer(x), self.log_var_layer(x)
        return mean, log_var

    def reparameteriztion(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        x = self.encoder(x)
        mean, log_var = self.mean_log_var(x)
        z = self.reparameteriztion(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    @staticmethod
    def loss(x, x_hat, mean, log_var):
        # 因为这个数据集的图形都是0/1，所以用交叉熵做Loss了
        reproductive_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        # reproductive_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
        # reproductive_loss = nn.MSELoss(x_hat,x,reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproductive_loss + kld


# cvae for mnist dataset
class CVAE(nn.Module):
    def __init__(self, class_dim: int = 10, input_dim: int = 28 * 28, hidden_dim: int = 400, latent_dim: int = 200,
                 device: str = 'cuda'):
        super(CVAE, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.input_dim = input_dim + latent_dim

        self.output_dim = input_dim

        self.class_embedding = nn.Embedding(self.class_dim, self.latent_dim)

        # encoder产生z
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.LeakyReLU(0.2),
        )

        # 然后产生p(z|x, y)的均值和方差
        # 图像是二维灰度图，所以方差和均值dim = 2
        self.mean_layer = nn.Linear(self.latent_dim, 2)
        self.log_var_layer = nn.Linear(self.latent_dim, 2)

        # decoder根据p(x|z, y)产生新的采样
        self.decoder = nn.Sequential(
            nn.Linear(2 + latent_dim, self.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        y = self.class_embedding(y).squeeze(dim=1)
        cat_x = torch.cat((x, y), 1)
        x = self.encoder(cat_x)
        return x, cat_x

    def decode(self, x, y):
        y = self.class_embedding(y).squeeze(dim=1)
        x = torch.cat((x, y), 1)
        return self.decoder(x)

    def mean_log_var(self, x):
        mean, log_var = self.mean_layer(x), self.log_var_layer(x)
        return mean, log_var

    def reparameteriztion(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x, y):
        x, cat_x = self.encode(x, y)
        mean, log_var = self.mean_log_var(x)
        z = self.reparameteriztion(mean, log_var)
        x_hat = self.decode(z, y)
        return x_hat, mean, log_var


    @staticmethod
    def loss(x, x_hat, mean, log_var):
        reproductive_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproductive_loss + kld