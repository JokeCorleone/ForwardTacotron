import torch
import torch.nn.functional as F
from models.tacotron import CBHG
import torch.nn as nn


class Discriminator(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
        ])

    def forward(self, x):
        #x = x.transpose(1, 2)
        features = list()
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


if __name__ == '__main__':
    model = Discriminator()

    x = torch.randn(3, 80, 100)
    print(x.shape)

    out = model(x)
    out2 = model(x)
    for (feats_fake, score_fake), (feats_real, _) in zip(out, out2):
        print(feats_fake.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)