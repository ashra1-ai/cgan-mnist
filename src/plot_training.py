import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

log_path = Path("results/training_log.csv")
df = pd.read_csv(log_path)

# Plot raw step-wise loss (noisy but shows training dynamics)
plt.figure(figsize=(10, 5))
plt.plot(df["d_loss"], label="Discriminator Loss")
plt.plot(df["g_loss"], label="Generator Loss")
plt.title("Loss per Training Step")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_per_step.png")
plt.close()

# Plot averaged loss per epoch
df_epoch = df.groupby("epoch").mean()

plt.figure(figsize=(10, 5))
plt.plot(df_epoch["d_loss"], marker="o", label="Discriminator Loss")
plt.plot(df_epoch["g_loss"], marker="o", label="Generator Loss")
plt.title("Loss per Epoch (Averaged)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_per_epoch.png")
plt.close()

print("✅ Saved:\n  results/loss_per_step.png\n  results/loss_per_epoch.png")
