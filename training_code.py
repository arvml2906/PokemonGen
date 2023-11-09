# %%
import sys
import os
from matplotlib.pyplot import imshow, imsave
from PIL import Image
import glob
import datetime
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from dataloader import POKE
from model import Discriminator, Generator
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
from google.colab import drive
#!pip install torch torchvision

# %%
drive.mount('/content/drive')

# %%
"""
# Importing Libraries
"""

# %%

# %%
IMAGE_DIM = (32, 32, 3)

# %%

# %%
#%matplotlib inline

# %%
MODEL_NAME = 'DCGAN'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%


def get_sample_image(G, n_noise):
    """
        save sample 100 images
    """
    z = torch.randn(10, n_noise).to(DEVICE)
    y_hat = G(z).view(10, 3, 32, 32).permute(0, 2, 3, 1)  # (100, 28, 28)
    result = (y_hat.detach().cpu().numpy() + 1) / 2.
    return result


# %%
#Load dataset from Google Drive
#!unzip - q "drive/My Drive/ZIP_FILES/PokemonImg.zip" - d "Dataset2" | head - n 5

# %%
"""
# Unwanted Images(Grayscale Images)
"""

# %%


def is_grayscale(image_path):
    img = Image.open(image_path)
    return img.mode == 'L'  # 'L' indicates grayscale mode in PIL


def is_4d(image_path):
    img = Image.open(image_path)
    return len(img.getbands()) == 4  # Check if the image has 4 channels


# Replace 'your_dataset_folder' with the path to your dataset folder
dataset_folder = 'Dataset2'
unwanted_images = []

for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        image_path = os.path.join(root, file)
        if is_grayscale(image_path) or is_4d(image_path):
            unwanted_images.append(image_path)

print(f"Number of unwanted images: {len(unwanted_images)}")
print(unwanted_images)


# %%
D = Discriminator(in_channel=IMAGE_DIM[-1]).to(DEVICE)
G = Generator(out_channel=IMAGE_DIM[-1]).to(DEVICE)

# %%


def convert_palette_to_rgba(image):
    if image.mode == 'P':
        return image.convert('RGBA')
    return image


transform = transforms.Compose([
    # transforms.Lambda(convert_palette_to_rgba),  # Convert palette images to
    # RGBA
    transforms.Resize((IMAGE_DIM[0], IMAGE_DIM[1])),
    # transforms.Grayscale(num_output_channels=3),  # Convert to RGB format
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


# %%
dataset = POKE(
    data_path='Dataset2/PokemonImg',
    transform=transform,
    grayscale_images=unwanted_images)

# %%
len(dataset)
remaining_grayscale_images = []
for idx in range(len(dataset)):
    img_path = dataset.fpaths[idx]
    if is_grayscale(img_path):
        remaining_grayscale_images.append(img_path)


# %%
len(remaining_grayscale_images)

# %%
batch_size = 64

# %%
"""
# Ensuring consistent Image dimensions in all batches
"""

# %%
# Define your collate_fn function


def collate_fn(batch):
    max_height = max([img.shape[1] for img in batch])
    max_width = max([img.shape[2] for img in batch])

    padded_images = []
    for img in batch:
        c, h, w = img.shape
        padded_img = torch.zeros(3, max_height, max_width)
        padded_img[:, :h, :w] = img
        padded_images.append(padded_img)

    return torch.stack(padded_images)


# %%
data_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=8,
    collate_fn=collate_fn)


# %%
for batch_idx, batch in enumerate(data_loader):
    print(f"Batch Index: {batch_idx}")
    print(f"Batch Shape: {batch.shape}")  # Check the shape of the batch


# %%
criterion = nn.BCELoss()
D_opt = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))

# %%
max_epoch = 100
step = 0
n_critic = 1  # for training more k steps about Discriminator
n_noise = 100

# %%
D_labels = torch.ones([batch_size, 1]).to(
    DEVICE)  # Discriminator Label to real
D_fakes = torch.zeros([batch_size, 1]).to(
    DEVICE)  # Discriminator Label to fake

# %%
"""
# Model Training
"""

# %%
for epoch in range(max_epoch):
    for idx, images in enumerate(data_loader):
        # Training Discriminator
        x = images.to(DEVICE)
        x_outputs = D(x)
        D_x_loss = criterion(x_outputs, D_labels)

        z = torch.randn(batch_size, n_noise).to(DEVICE)
        z_outputs = D(G(z))
        D_z_loss = criterion(z_outputs, D_fakes)
        D_loss = D_x_loss + D_z_loss

        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        if step % n_critic == 0:
            # Training Generator
            z = torch.randn(batch_size, n_noise).to(DEVICE)
            z_outputs = D(G(z))
            G_loss = criterion(z_outputs, D_labels)

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_opt.step()

        if step % 500 == 0:
            dt = datetime.datetime.now().strftime('%H:%M:%S')
            print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, G Loss: {:.4f}, Time:{}'.format(
                epoch, max_epoch, step, D_loss.item(), G_loss.item(), dt))
            G.eval()
            img = get_sample_image(G, n_noise)
            # create folder by right clicking in the file window and name the folder 'samples'
            # imsave('content/samples/{}_step{:05d}.jpg'.format(MODEL_NAME, step), img[0])
            G.train()
        step += 1

# %%
"""
# Results
"""

# %%
G.eval()  # Generated Image sample
imshow(get_sample_image(G, n_noise)[0])

# %%
t = Image.open(dataset.fpaths[31])  # Actual Pokemon image sample
t = (transform(t).permute(1, 2, 0) + 1) / 2.
imshow(t)
