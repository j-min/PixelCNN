import torch
import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, c_in, c_out, k_size, stride, pad):
        """2D Convolution with masked weight for Autoregressive connection"""
        super(MaskedConv2d, self).__init__(
            c_in, c_out, k_size, stride, pad, bias=False)
        assert mask_type in ['A', 'B']
        self.mask_type = mask_type
        ch_out, ch_in, height, width = self.weight.size()

        # Mask
        #         -------------------------------------
        #        |  1       1       1       1       1 |
        #        |  1       1       1       1       1 |
        #        |  1       1    1 if B     0       0 |   H // 2
        #        |  0       0       0       0       0 |   H // 2 + 1
        #        |  0       0       0       0       0 |
        #         -------------------------------------
        #  index    0       1     W//2    W//2+1

        mask = torch.ones(ch_out, ch_in, height, width)
        if mask_type == 'A':
            # First Convolution Only
            # => Restricting connections to
            #    already predicted neighborhing channels in current pixel
            mask[:, :, height // 2, width // 2:] = 0
            mask[:, :, height // 2 + 1:] = 0
        else:
            mask[:, :, height // 2, width // 2 + 1:] = 0
            mask[:, :, height // 2] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


def maskAConv(c_in=3, c_out=256, k_size=7, stride=1, pad=3):
    """2D Masked Convolution (type A)"""
    return nn.Sequential(
        MaskedConv2d('A', c_in, c_out, k_size, stride, pad),
        nn.BatchNorm2d(c_out))


class MaskBConvBlock(nn.Module):
    def __init__(self, h=128, k_size=3, stride=1, pad=1):
        """1x1 Conv + 2D Masked Convolution (type B) + 1x1 Conv"""
        super(MaskBConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * h, h, 1),  # 1x1
            nn.BatchNorm2d(h),
            nn.ReLU(),
            MaskedConv2d('B', h, h, k_size, stride, pad),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            nn.Conv2d(h, 2 * h, 1),  # 1x1
            nn.BatchNorm2d(2 * h)
        )

    def forward(self, x):
        """Residual connection"""
        return self.net(x) + x
