import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
from PIL import Image

from diffusers import DDPMScheduler, UNet2DModel


PATH = "CHANGE WITH YOUR OWN PATH"


# first we create custom class to loading our images from directory
class CustomImageLoader(Dataset):
    # inital class we set: path, images array and transform
    def __init__(self, path, transform=None):
        self.path = path
        self.images = [
            image
            for image in os.listdir(path)
            if image.endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform

    # take number of images of our directory
    def __len__(self):
        return len(self.images)

    #
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def main():
    # augmenting our data on-the-fly by using our existing data thus we have "more (since it's on-the-fly we can't see them)" data
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = CustomImageLoader(
        path=PATH, transform=transform
    )  # sending our images to our custom class
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True
    )  # loading our images into data loader so we can then pass it into our model

    print(f"There are {len(dataset)} images on dataset")  # len of our image dataset

    # defining our model
    model = UNet2DModel(
        # size of input images
        sample_size=32,
        # number of channels for input images
        in_channels=3,
        # number of channels of output images
        out_channels=3,
        layers_per_block=2,
        # number of filters for per up and down sampling block
        block_out_channels=(64, 128, 256),
        # since we have relatively small dataset we can use simple down blocks
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        # since we have relatively small dataset we can use simple up blocks
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  #

    scheduler = DDPMScheduler(
        # control the noise addition in 2000 time steps e.g. in how many steps we'll have clean image
        num_train_timesteps=1000,
        # it starts with adding 0.0001 level of noise
        beta_start=1e-4,
        # up to 0.02
        beta_end=2e-2,
        # and it increases it linearly
        beta_schedule="linear",
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # setting optimizer
    loss_fn = nn.MSELoss()  # setting loss function MSE loss is most common one
    epochs = 500  # number of epochs

    model.train()  # training model
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"{epoch+1}/{epochs}"):
            images = batch.to(device)  # sending batch of images to device
            batch_size = images.shape[0]  # number of images in batch (16)

            # here we're picking random different timestep for each image in current batch
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (batch_size,), device=device
            ).long()
            # generates random noise images for each image in the batch with the same shape our image (randn_like provides this)
            # e.g. if our batch is 16 and images are RGB 32*32 images then it will shape like => [16, 3, 32, 32]
            noise = torch.randn_like(images)
            # here we're adding noise to our images based on chosen timesteps
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            # here model tries to predict at which timestep of the given image
            noise_pred = model(noisy_images, timesteps).sample

            loss = loss_fn(noise_pred, noise)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "diffusion_model.pth")
    print("Training complete. Model saved as 'diffusion_model.pth'.")


if __name__ == "__main__":
    main()
