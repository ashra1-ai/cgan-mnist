import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Small CNN discriminator conditioned on labels.
    Input: (B, 1, 28, 28) and labels (B,)
    Output: probability (B, 1)
    """
    def __init__(self, num_classes=10, img_channels=1, feature_maps=64):
        super().__init__()
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # small conv stack
        self.conv = nn.Sequential(
            # input: (B, 1 + num_classes_as_channels?, 28, 28)
            nn.Conv2d(img_channels + 0, feature_maps, kernel_size=4, stride=2, padding=1), # 14x14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps*2, kernel_size=4, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps*2, feature_maps*4, kernel_size=3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # final classifier: we'll combine conv features + label embedding
        self.classifier = nn.Sequential(
            nn.Linear(feature_maps*4*4*4 + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        B = img.size(0)
        x = self.conv(img)                       # (B, fm*4, h, w)
        x = x.view(B, -1)
        lbl = self.label_emb(labels)
        x = torch.cat([x, lbl], dim=1)
        out = self.classifier(x)
        return out
if __name__ == "__main__":
    # simple test
    D = Discriminator()
    fake_imgs = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    probs = D(fake_imgs, labels)
    print(probs.shape)  # should be (8, 1)