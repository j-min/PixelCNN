import torch.nn as nn
from layers import *


class PixelCNN(nn.Module):
    def __init__(self, c_in=3, dim=128, c_out=256, k_size=3, stride=1, pad=1):
        """PixelCNN Model"""

        self.MaskAConv = MaskedConv2d('A', 1, dim, k_size=7, stride=1, pad=3)
        self.MaskBConv = []
        for i in range(15):
            self.MaskBConv.append(MaskedConvBlock('B', dim, k_size, stride, pad))
        self.

            # 1x1 conv to 256 channels
            nn.Conv2d(dim, c_out, k_size=1, stride=1, pad=0)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, channel=1, height, width]
        Return:
            out [batch_size, channel_out=256, height, width]
        """

        x = self.MaskAConv(x)
        for conv_block in self.conv_layers:


        return self.net(x)

# class PixelRNN(nn.Module):
#
#     def __init__(self):
#         super(PixelRNN, self).__init__(self)
#
#     def forward(self, x):
#         pass
