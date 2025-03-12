import torch
from torchvision.utils import save_image
import numpy as np

import matplotlib.pyplot as plt

from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    DDPMPipeline,
)


def show_images(images, nrow=2, title="Generated Images"):
    # Convert to CPU if on GPU
    images = images.cpu()

    # Number of images
    batch_size = images.shape[0]

    # Calculate grid size
    ncol = (batch_size + nrow - 1) // nrow  # Ceiling division for columns

    # Create a figure
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 3, ncol * 3))
    fig.suptitle(title)

    # Flatten axes array if batch_size is small
    if batch_size == 1:
        axes = [axes]
    elif ncol == 1 or nrow == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each image
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0)  # [C, H, W] -> [H, W, C] for matplotlib
        axes[i].imshow(img)
        axes[i].axis("off")  # Hide axes

    # Adjust layout and show
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Re-instantiate your model and scheduler with the same architecture/settings used in training
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
    )
    model.to(device)

    # Load the trained weights (adjust the path if the file is in another folder)
    model.load_state_dict(
        torch.load(
            "PATH_TO_TRAINED_WEIGHTS.pth",
            map_location=device,
        )
    )
    model.eval()

    # Create a pipeline for generating images
    pipeline = DDPMPipeline(unet=model, scheduler=scheduler)
    pipeline.to(device)

    with torch.no_grad():
        # Generate images starting from random noise
        generated_images = pipeline(
            batch_size=4,
            generator=torch.manual_seed(42),  # change seed for different images
        ).images

    # Convert the generated images (PIL Images) to a tensor and normalize
    generated_images = (
        torch.stack(
            [
                torch.from_numpy(np.array(img)).permute(2, 0, 1)
                for img in generated_images
            ]
        ).float()
        / 255.0
    )
    generated_images = generated_images.clamp(0, 1)

    # Save the generated images
    save_image(generated_images, "generated_images.png", nrow=2)
    print("Generated images saved as 'generated_images.png'. Here are the images:")
    show_images(generated_images, nrow=2)


if __name__ == "__main__":
    main()
