"""
Generate conditional samples from a saved checkpoint.
Usage:
  python -m src.generate --digit 3 --count 25 --checkpoint checkpoints/generator_epoch_020.pth
"""
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image

from src.config import *
from src.models.generator import Generator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--digit", type=int, required=True)
    p.add_argument("--count", type=int, default=25)
    p.add_argument("--checkpoint", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    device = DEVICE

    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES, img_channels=IMG_SHAPE[0]).to(device)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = Path(CHECKPOINTS_DIR) / f"generator_epoch_{EPOCHS:03d}.pth"

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()

    noise = torch.randn(args.count, Z_DIM, device=device)
    labels = torch.full((args.count,), args.digit, dtype=torch.long, device=device)

    with torch.no_grad():
        samples = G(noise, labels).cpu()

    out_path = Path(SAMPLES_DIR) / f"custom_digit_{args.digit}.png"
    save_image(samples, str(out_path), nrow=int(args.count**0.5), normalize=True)
    print(f"Wrote samples to: {out_path}")

if __name__ == "__main__":
    main()
