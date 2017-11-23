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

        mask = torch.ones(ch_in, ch_out, height, width)
        if mask_type == 'A':
            # First Convolution Only
            # => Restricting connections to
            #    already predicted neighborhing channels in current pixel
            mask[:, :, height//2, width//2:] = 0
            mask[:, :, height//2+1:] = 0
        else:
            mask[:, :, height//2, width//2+1:] = 0
            mask[:, :, height//2]
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskedConvBlock(nn.Module):
    def __init__(self, mask_type='B', dim=128, k_size=3, stride=1, pad=1):
        """1x1 Conv + 2D Masked Convolution + 1x1 Conv"""
        self.net = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            MaskedConv2d(mask_type, dim, dim, k_size, stride, pad),
            nn.Conv2d(dim, 2*dim, 1)
        )

    def forward(self, x):
        return self.next(x)
