from __future__ import annotations

import torch
import torch.nn as nn

from duckietown_seg.models.blocks import ASPPLite, DoubleConvBlock, DownBlock, OutputProjection, UpBlock, initialize_weights


class ConfigurableUNet(nn.Module):
    """U-Net with depthwise, SE decoder, and ASPP-lite toggles for fair ablations."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 32,
        bilinear: bool = True,
        use_depthwise: bool = False,
        use_se_decoder: bool = False,
        use_aspp: bool = False,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        factor = 2 if bilinear else 1
        bottleneck_channels = base_channels * 16 // factor

        self.inc = DoubleConvBlock(in_channels, c1, use_depthwise=use_depthwise, use_se=False)
        self.down1 = DownBlock(c1, c2, use_depthwise=use_depthwise)
        self.down2 = DownBlock(c2, c3, use_depthwise=use_depthwise)
        self.down3 = DownBlock(c3, c4, use_depthwise=use_depthwise)
        self.down4 = DownBlock(c4, bottleneck_channels, use_depthwise=use_depthwise)
        self.aspp = ASPPLite(bottleneck_channels) if use_aspp else nn.Identity()
        self.up1 = UpBlock(
            base_channels * 16,
            c4 // factor,
            bilinear=bilinear,
            use_depthwise=use_depthwise,
            use_se=use_se_decoder,
        )
        self.up2 = UpBlock(
            base_channels * 8,
            c3 // factor,
            bilinear=bilinear,
            use_depthwise=use_depthwise,
            use_se=use_se_decoder,
        )
        self.up3 = UpBlock(
            base_channels * 4,
            c2 // factor,
            bilinear=bilinear,
            use_depthwise=use_depthwise,
            use_se=use_se_decoder,
        )
        self.up4 = UpBlock(
            base_channels * 2,
            c1,
            bilinear=bilinear,
            use_depthwise=use_depthwise,
            use_se=use_se_decoder,
        )
        self.outc = OutputProjection(c1, num_classes)

        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.aspp(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

