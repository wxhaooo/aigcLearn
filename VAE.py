import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transforms = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transforms)
device = torch.device("cuda" if torch.cuda.is_available() else "")
print(device)

batch_size = 128
train_loader = DataLoader(dataset=mnist_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
train_data_iter = iter(train_loader)
images = train_data_iter.__next__()
num_sampler = 25
sample_images = [images[0][i,0] for i in range(num_sampler)]
sample_image = sample_images[0]
image_size = sample_image.size()
x_dim = image_size[0] * image_size[1]
print(x_dim)

import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import time

currentTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
writer = SummaryWriter('./log/{currentTime}'.format(currentTime=currentTime))


class VAE(nn.Module):

    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 400, latent_dim: int = 200, device: str = 'cuda'):
        super(VAE, self).__init__()

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

        # 均值和方差与z的维度一致
        # self.mean_layer = nn.Linear(latent_dim, latent_dim)
        # self.log_var_layer = nn.Linear(latent_dim, latent_dim)

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
        mean, log_var = self.mean_layer(x), self.log_var_layer(x)
        return mean, log_var

    def decode(self, x):
        return self.decoder(x)

    def reparameteriztion(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameteriztion(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var


def loss_function(x, x_hat, mean, log_var):
    # 因为这个数据集的图形都是0/1，所以用交叉熵做Loss了
    reproductive_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    # reproductive_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
    # reproductive_loss = nn.MSELoss(x_hat,x,reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproductive_loss + kld


def vae_train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(batch_size, x_dim).to(device)
            x_hat, mean, log_var = model(data)
            loss = loss_function(data, x_hat, mean, log_var)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('vae_loss/train', epoch_loss, epoch)
        print(epoch_loss)

    writer.close()


class CVAE(VAE):
    def __init__(self, num_classes: int, input_dim: int = 28 * 28, hidden_dim: int = 400, latent_dim: int = 200, device: str = 'cuda'):
        super(CVAE, self).__init__(input_dim, hidden_dim, latent_dim, device)
        self.label_projector = nn.Sequential(
            nn.Linear(num_classes, self.latent_dim),
            nn.ReLU())

    def condition_on_label(self, z, y):
        projected_label = self.label_projector(y)
        return z + projected_label

    def forward(self, x, y):
        mean, log_var = self.encode(x)
        z = self.reparameteriztion(mean, log_var)
        z = self.condition_on_label(z, y)
        x_hat = self.decode(z)
        return x_hat, mean, log_var


def cvae_train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(batch_size, x_dim).to(device)
            x_hat, mean, log_var = model(data, target)
            loss = loss_function(data, x_hat, mean, log_var)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('cvae_loss/train', epoch_loss, epoch)
        print(epoch_loss)

    writer.close()


# cvae train
cvae_model: CVAE = CVAE(num_classes = 10).to(device)
optimizer = Adam(cvae_model.parameters(), lr=1e-3)
cvae_train(cvae_model, optimizer, epochs=100, device=device)

# vae train
# vae_model = VAE().to(device)
# optimizer = Adam(vae_model.parameters(), lr = 1e-3)
# vae_train(vae_model, optimizer, epochs=100, device=device)