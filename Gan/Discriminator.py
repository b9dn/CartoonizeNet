import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.norm_1 = nn.BatchNorm2d(128)

        self.conv_4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.norm_2 = nn.BatchNorm2d(256)

        self.conv_6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.norm_3 = nn.BatchNorm2d(256)

        self.conv_7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv_8 = nn.Conv2d(256, 1, 4, padding=0)

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.norm_1(self.conv_3(F.leaky_relu(self.conv_2(x)))), negative_slope=0.2)
        x = F.leaky_relu(self.norm_2(self.conv_5(F.leaky_relu(self.conv_4(x)))), negative_slope=0.2)
        x = F.leaky_relu(self.norm_3(self.conv_6(x)), negative_slope=0.2)
        x = self.conv_8(self.conv_7(x))

        return sigmoid(x)
