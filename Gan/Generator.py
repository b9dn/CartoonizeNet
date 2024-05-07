import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid


class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.conv_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.norm_1 = nn.BatchNorm2d(256)
        self.norm_2 = nn.BatchNorm2d(256)

    def forward(self, x):
        output = self.norm_2(self.conv_2(F.relu(self.norm_1(self.conv_1(x)))))
        return output + x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, 7, padding=3)
        self.norm_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.norm_2 = nn.BatchNorm2d(128)

        self.conv_4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(256, 256, 3, padding=1)
        self.norm_3 = nn.BatchNorm2d(256)

        residual_blocks = []
        for l in range(8):
            residual_blocks.append(res_block())
        self.res = nn.Sequential(*residual_blocks)

        self.conv_6 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv_7 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.norm_4 = nn.BatchNorm2d(128)

        self.conv_8 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv_9 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.norm_5 = nn.BatchNorm2d(64)

        self.conv_10 = nn.Conv2d(64, 3, 7, padding=3)

    def forward(self, x):
        x = F.relu(self.norm_1(self.conv_1(x)))
        x = F.relu(self.norm_2(self.conv_3(self.conv_2(x))))
        x = F.relu(self.norm_3(self.conv_5(self.conv_4(x))))
        x = self.res(x)
        x = F.relu(self.norm_4(self.conv_7(self.conv_6(x))))
        x = F.relu(self.norm_5(self.conv_9(self.conv_8(x))))

        return sigmoid(self.conv_10(x))