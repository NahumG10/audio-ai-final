import torch
import torch.nn as nn
from .config import ModelConfig


class ConvBlock(nn.Module):
    """Conv2D -> BatchNorm -> LeakyReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch_in = in_channels
        for ch_out in channels:
            self.blocks.append(ConvBlock(ch_in, ch_out, kernel_size))
            self.pools.append(nn.MaxPool2d(2))
            ch_in = ch_out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_connections = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skip_connections.append(x)
            x = pool(x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(
        self,
        bottleneck_ch: int,
        channels: list[int],
        skip_channels: list[int],
        kernel_size: int,
    ):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()

        ch_in = bottleneck_ch
        for ch_out, skip_ch in zip(channels, skip_channels):
            self.upconvs.append(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
            )
            self.blocks.append(ConvBlock(ch_out + skip_ch, ch_out, kernel_size))
            ch_in = ch_out

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        for upconv, block, skip in zip(self.upconvs, self.blocks, reversed(skip_connections)):
            x = upconv(x)
            dh = skip.size(2) - x.size(2)
            dw = skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [0, dw, 0, dh])
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """
    U-Net for vocal isolation.

    Takes a log-magnitude spectrogram of a mix as input and predicts
    a soft mask that isolates the vocal component.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        enc_channels = config.encoder_channels
        dec_channels = config.decoder_channels
        self.num_enc_blocks = len(enc_channels)

        # Skip channels in reverse order (matching decoder steps)
        skip_channels_reversed = list(reversed(enc_channels))[:len(dec_channels)]

        self.encoder = Encoder(config.in_channels, enc_channels, config.kernel_size)
        self.bottleneck = ConvBlock(enc_channels[-1], enc_channels[-1], config.kernel_size)
        self.dropout = nn.Dropout2d(config.dropout)
        self.decoder = Decoder(
            enc_channels[-1], dec_channels, skip_channels_reversed, config.kernel_size
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(dec_channels[-1], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_h, original_w = x.shape[2], x.shape[3]
        factor = 2 ** self.num_enc_blocks
        pad_h = (factor - original_h % factor) % factor
        pad_w = (factor - original_w % factor) % factor
        x = nn.functional.pad(x, [0, pad_w, 0, pad_h])

        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.dropout(x)
        x = self.decoder(x, skips)
        mask = self.final_conv(x)

        mask = mask[:, :, :original_h, :original_w]
        return mask


def build_model(config: ModelConfig) -> UNet:
    return UNet(config)
