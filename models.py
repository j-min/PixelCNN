import torch.nn as nn
from layers import maskAConv, MaskBConvBlock


class PixelCNN(nn.Module):
    def __init__(self, n_channel=3, h=128, discrete_channel=256):
        """PixelCNN Model"""
        super(PixelCNN, self).__init__()

        self.discrete_channel = discrete_channel

        self.MaskAConv = maskAConv(n_channel, 2 * h, k_size=7, stride=1, pad=3)
        MaskBConv = []
        for i in range(15):
            MaskBConv.append(MaskBConvBlock(h, k_size=3, stride=1, pad=1))
        self.MaskBConv = nn.Sequential(*MaskBConv)

        # 1x1 conv to 3x256 channels
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * h, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, n_channel * discrete_channel, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """
        Args:
            x: [batch_size, channel, height, width]
        Return:
            out [batch_size, channel, height, width, 256]
        """
        batch_size, c_in, height, width = x.size()

        # [batch_size, 2h, 32, 32]
        x = self.MaskAConv(x)

        # [batch_size, 2h, 32, 32]
        x = self.MaskBConv(x)

        # [batch_size, 3x256, 32, 32]
        x = self.out(x)

        # [batch_size, 3, 256, 32, 32]
        x = x.view(batch_size, c_in, self.discrete_channel, height, width)

        # [batch_size, 3, 32, 32, 256]
        x = x.permute(0, 1, 3, 4, 2)

        return x
