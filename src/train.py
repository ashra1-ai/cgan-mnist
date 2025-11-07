"""
Train a conditional GAN on MNIST (CPU-friendly).
Run from repository root:
    python -m src.train
"""
import os, csv
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from src.config import *
from src.utils import seed_everything
from src.models.generator import Generator
from src.models.discriminator import Discriminator

def make_dirs():
    Path(SAMPLES_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

def sample_grid(generator, epoch, device, sample_count=SAMPLE_COUNT):
    generator.eval()
    labels = torch.arange(0, NUM_CLASSES, device=device).repeat(sample_count // NUM_CLASSES + 1)[:sample_count]
    noise = torch.randn(sample_count, Z_DIM, device=device)
    with torch.no_grad():
        imgs = generator(noise, labels).cpu()
    save_image(imgs, Path(SAMPLES_DIR) / f"epoch_{epoch:03d}.png", nrow=int(sample_count**0.5), normalize=True)
    generator.train()

def main():
    seed_everything(SEED)
    make_dirs()
    device = DEVICE

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # models
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES, img_channels=IMG_SHAPE[0]).to(device)
    D = Discriminator(num_classes=NUM_CLASSES, img_channels=IMG_SHAPE[0]).to(device)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    log_file = Path(RESULTS_DIR) / "training_log.csv"
    with open(log_file, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "step", "d_loss", "g_loss"])

    step = 0
    for epoch in range(1, EPOCHS + 1):
        loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}/{EPOCHS}")
        for i, (real_imgs, labels) in loop:
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            bs = real_imgs.size(0)
            valid = torch.ones(bs, 1, device=device)
            fake_lab = torch.zeros(bs, 1, device=device)

            # -----------------
            # Train Discriminator
            # -----------------
            z = torch.randn(bs, Z_DIM, device=device)
            fake_imgs = G(z, labels)

            D_real = D(real_imgs, labels)
            D_fake = D(fake_imgs.detach(), labels)

            d_loss = (criterion(D_real, valid) + criterion(D_fake, fake_lab)) / 2

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # -----------------
            # Train Generator
            # -----------------
            z = torch.randn(bs, Z_DIM, device=device)
            fake_imgs = G(z, labels)
            out = D(fake_imgs, labels)
            g_loss = criterion(out, valid)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            if step % 10 == 0:
                loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

            with open(log_file, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, step, f"{d_loss.item():.6f}", f"{g_loss.item():.6f}"])

            step += 1

        # end epoch: save sample & checkpoint
        if epoch % SAVE_EVERY == 0:
            sample_grid(G, epoch, device, SAMPLE_COUNT)
            torch.save(G.state_dict(), Path(CHECKPOINTS_DIR) / f"generator_epoch_{epoch:03d}.pth")
            torch.save(D.state_dict(), Path(CHECKPOINTS_DIR) / f"discriminator_epoch_{epoch:03d}.pth")

        print(f"Epoch {epoch} finished | d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}")

if __name__ == "__main__":
    main()
