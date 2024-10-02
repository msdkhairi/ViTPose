
from typing import Tuple, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.sam2 import MultiScaleBlock

from ..builder import NECKS


class LSTEmbed(nn.Module):
    """
    Patch to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Union[Tuple[int, int], int] = 1,
        stride: Union[Tuple[int, int], int] = 1,
        padding: Union[Tuple[int, int], int] = 0,
        in_chans: int = 256,
        out_chans: int = 256,
        flatten: bool = False,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.flatten:
            x = x.flatten(1, 2)
        return x


@NECKS.register_module()
class SAM2PESimpleLST(nn.Module):
    def __init__(self,
                 input_feature_size: int = 64,
                 feature_dim=256,
                 decoder_embed_dim=768,
                 transformer_depth=12,
                 drop_path_rate=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 use_abs_pos=True,
                 num_transformer_heads=8,
                 window_size=0,
                ):
        super().__init__()

        # self.embed = nn.Conv2d(feature_dim, decoder_embed_dim, kernel_size=1, stride=1, padding=0)
        self.embed = LSTEmbed(kernel_size=1, 
                              stride=1, 
                              padding=0, 
                              in_chans=feature_dim, 
                              out_chans=decoder_embed_dim,
                              flatten=False)

        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_feature_size, input_feature_size, decoder_embed_dim))
        
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, transformer_depth)
        ]

        self.blocks = nn.ModuleList()
        for i in range(transformer_depth):
            block = MultiScaleBlock(
                dim=decoder_embed_dim,
                dim_out=decoder_embed_dim,
                num_heads=num_transformer_heads,
                drop_path=dpr[i],
                q_stride=None,
                window_size=window_size,
            )
            self.blocks.append(block)
    
    def init_weights(self):
        pass
    
    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        pos_embed = self.pos_embed
        pos_embed = F.interpolate(
            pos_embed.permute(0, 3, 1, 2).contiguous(), 
            size=(h, w), 
            mode="bicubic")
        
        pos_embed = pos_embed.permute(0, 2, 3, 1).contiguous()
        return pos_embed
    
    def forward_blocks(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x
    
    def forward_features(self, x):
        B, C, H, W = x.shape # (B, 256, 64, 64)
        
        x = self.embed(x) # (B, 64, 64, decoder_embed_dim)

        x = x + self._get_pos_embed((H, W))

        output = self.forward_blocks(x)

        return output

    def forward(self, x):
        x = self.forward_features(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x