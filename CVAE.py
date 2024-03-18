import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import time
from Model import CVAE

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


def cvae_train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(batch_size, x_dim).to(device)
            # target = target.type(torch.float32)
            target = target.view(batch_size, 1).to(device)

            x_hat, mean, log_var = model(data, target)
            loss = CVAE.loss(data, x_hat, mean, log_var)
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('cvae_loss/train', epoch_loss, epoch)
        print(epoch_loss)

    writer.close()


# cvae_model = CVAE().to(device)
# optimizer = Adam(cvae_model.parameters(), lr=1e-3)
# cvae_train(cvae_model, optimizer, epochs=100, device=device)

# save checkpoints
# checkpoint_name = f'{checkpoints_path}/cvae_embedding_test.pt'
# torch.save(cvae_model.state_dict(), checkpoint_name)

# load checkpoints
cvae_model_path = './checkpoints/cvae_embedding_test.pt'
cvae_model = CVAE().to(device)
cvae_model.load_state_dict(torch.load(cvae_model_path))


# generate new digit
def cvae_generate_single_digit(y):
    with torch.no_grad():
        z = torch.randn(1, 2, device='cuda')
        x_decoded = cvae_model.decode(z, y)
        digit = x_decoded.detach().cpu().reshape(28, 28)
        plt.imshow(digit, cmap='grey')
        plt.axis('off')
        plt.show()
        torch.cuda.empty_cache()


def cvae_plot_latent_space(cvae_model, scale=1, n=25, digit_size=28, fig_size=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))
    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            y = torch.randint(0,9,[1,1], dtype=torch.long,device='cuda')
            # y = torch.tensor([8], device='cuda', dtype=torch.long)
            x_decoded = cvae_model.decode(z_sample, y)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size, ] = digit

    plt.figure(figsize=(fig_size, fig_size))
    plt.title('CVAE Latent Space Visualization')
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


# cvae_generate_single_digit(torch.tensor([9],device='cuda',dtype=torch.long))
cvae_model.eval()
cvae_plot_latent_space(cvae_model)