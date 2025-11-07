import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Small CNN/Transposed-CNN generator designed to be CPU-friendly while
    producing clearer images than an MLP. Input: (batch, z_dim) + labels.
    Output: (batch, 1, 28, 28), range [-1,1]
    """
    def __init__(self, z_dim=100, num_classes=10, img_channels=1, feature_maps=64):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_channels = img_channels

        # embed label, combine with noise
        self.label_emb = nn.Embedding(num_classes, num_classes)

        input_dim = z_dim + num_classes

        # A small MLP to expand to feature map vector then reshape to conv feature map
        self.fc = nn.Sequential(
            nn.Linear(input_dim, feature_maps * 4 * 3 * 3),
            nn.BatchNorm1d(feature_maps * 4 * 3 * 3),
            nn.ReLU(True),
        )

        # conv transpose stack to upsample to 28x28
        self.deconv = nn.Sequential(
            # input: (B, feature_maps*4, 3, 3)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1), # 6x6
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1), # 12x12
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1), # 24x24 -> we will crop/pad
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # noise: (B, z_dim), labels: (B,)
        lbl = self.label_emb(labels)             # (B, num_classes)
        x = torch.cat([noise, lbl], dim=1)       # (B, z_dim + num_classes)
        x = self.fc(x)
        B = x.size(0)
        x = x.view(B, -1, 3, 3)
        x = self.deconv(x)                       # may be 24x24 or 48x48 depending kernel choices
        # Resize/crop/pad to 28x28 if necessary
        x = nn.functional.interpolate(x, size=(28,28), mode='bilinear', align_corners=False)
        return x
if __name__ == "__main__":
    # simple test
    G = Generator()
    noise = torch.randn(8, 100)
    labels = torch.randint(0, 10, (8,))
    fake_imgs = G(noise, labels)
    print(fake_imgs.shape)  # should be (8, 1, 28, 28)