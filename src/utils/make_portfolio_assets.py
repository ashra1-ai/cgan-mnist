import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from pathlib import Path
import numpy as np

from src.config import *
from src.models.generator import Generator


def plot_losses():
    log_path = Path(RESULTS_DIR) / "training_log.csv"
    df = pd.read_csv(log_path)

    plt.figure()
    plt.plot(df["d_loss"], label="Discriminator Loss")
    plt.plot(df["g_loss"], label="Generator Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    out = Path(RESULTS_DIR) / "loss_curve.png"
    plt.savefig(out, dpi=200)
    print(f"[saved] {out}")


def generate_digit_grid():
    device = DEVICE
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES, img_channels=IMG_SHAPE[0]).to(device)
    ckpt = Path(CHECKPOINTS_DIR) / f"generator_epoch_{EPOCHS:03d}.pth"
    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()

    rows = []
    for digit in range(10):
        z = torch.randn(10, Z_DIM, device=device)
        labels = torch.full((10,), digit, dtype=torch.long, device=device)
        with torch.no_grad():
            imgs = G(z, labels).cpu()
        rows.append(imgs)

    grid = torch.cat(rows, dim=0)
    out = Path(SAMPLES_DIR) / "digits_grid.png"
    save_image(grid, out, nrow=10, normalize=True)
    print(f"[saved] {out}")


def latent_interpolation():
    device = DEVICE
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES, img_channels=IMG_SHAPE[0]).to(device)
    ckpt = Path(CHECKPOINTS_DIR) / f"generator_epoch_{EPOCHS:03d}.pth"
    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()

    z1 = torch.randn(1, Z_DIM, device=device)
    z2 = torch.randn(1, Z_DIM, device=device)
    labels = torch.tensor([7], device=device)  # animate within digit 7
    frames = []

    for alpha in np.linspace(0, 1, 20):
        z = (1-alpha) * z1 + alpha * z2
        with torch.no_grad():
            img = G(z, labels).cpu()
        frames.append(img)

    gif_path = Path(SAMPLES_DIR) / "interpolation.gif"

    import imageio
    imgs = [((f.squeeze() + 1) / 2 * 255).numpy().astype(np.uint8) for f in frames]
    imageio.mimsave(gif_path, imgs, duration=0.12)

    print(f"[saved] {gif_path}")


def main():
    plot_losses()
    generate_digit_grid()
    latent_interpolation()


if __name__ == "__main__":
    main()
