import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import time
from Model import VAE

transforms = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
device = torch.device("cuda" if torch.cuda.is_available() else "")

batch_size = 128
train_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_data_iter = iter(train_loader)
images = train_data_iter.__next__()
num_sampler = 25
sample_images = [images[0][i, 0] for i in range(num_sampler)]
sample_image = sample_images[0]
image_size = sample_image.size()
x_dim = image_size[0] * image_size[1]

currentTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
writer = SummaryWriter('./log/{currentTime}'.format(currentTime=currentTime))


def vae_train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(batch_size, x_dim).to(device)
            x_hat, mean, log_var = model(data)
            loss = VAE.loss(data, x_hat, mean, log_var)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('vae_loss/train', epoch_loss, epoch)
        print(epoch_loss)

    writer.close()


def vae_plot_latent_space(vae_model, scale=1, n=25, digit_size=28, fig_size=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))
    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = vae_model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size, ] = digit

    plt.figure(figsize=(fig_size, fig_size))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


# vae train
# vae_model = VAE().to(device)
# optimizer = Adam(vae_model.parameters(), lr = 1e-3)
# vae_train(vae_model, optimizer, epochs=100, device=device)

# checkpoints_path = './checkpoints/'
# checkpoint_name = f'{checkpoints_path}/vae_{currentTime}.pt'
# # torch.save(vae_model.state_dict(), checkpoint_name)

vae_model_path = './checkpoints/vae_2024-03-17-19_14_38.pt'
vae_model = VAE().to(device)
vae_model.load_state_dict(torch.load(vae_model_path))
vae_model.eval()

vae_plot_latent_space(vae_model)
