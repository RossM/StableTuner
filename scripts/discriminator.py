import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops, einops.layers.torch
import diffusers
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from typing import Tuple

def Downsample(dim, dim_out):
    return nn.Conv2d(dim, dim_out, 4, 2, 1)

class Residual(nn.Sequential):
    def forward(self, input):
        x = input
        for module in self:
            x = module(x)
        return x + input

def Block(dim, dim_out, *, kernel_size=3, groups=8):
    return nn.Sequential(
        nn.GroupNorm(groups, dim_out),
        nn.SiLU(),
        nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size//2),
    )

def ResnetBlock(dim, *, kernel_size=3, groups=8):
    return Residual(
        Block(dim, dim, kernel_size=kernel_size, groups=groups),
        Block(dim, dim, kernel_size=kernel_size, groups=groups),
    )
    
class SelfAttention(nn.Module):
    def __init__(self, dim, out_dim, *, heads=4, key_dim=32, value_dim=32):
        super().__init__()
        self.dim = dim
        self.out_dim = dim
        self.heads = heads
        self.key_dim = key_dim

        self.to_k = nn.Linear(dim, key_dim)
        self.to_v = nn.Linear(dim, value_dim)
        self.to_q = nn.Linear(dim, key_dim * heads)
        self.to_out = nn.Linear(value_dim * heads, out_dim)

    def forward(self, x):
        shape = x.shape
        x = einops.rearrange(x, 'b c ... -> b (...) c')

        k = self.to_k(x)
        v = self.to_v(x)
        q = self.to_q(x)
        q = einops.rearrange(q, 'b n (h c) -> b (n h) c', h=self.heads)
        if hasattr(nn.functional, "scaled_dot_product_attention"):
            result = F.scaled_dot_product_attention(q, k, v)
        else:
            attention_scores = torch.bmm(q, k.transpose(-2, -1))
            attention_probs = torch.softmax(attention_scores.float() / math.sqrt(self.key_dim), dim=-1).type(attention_scores.dtype)
            result = torch.bmm(attention_probs, v)
        result = einops.rearrange(result, 'b (n h) c -> b n (h c)', h=self.heads)
        out = self.to_out(result)

        out = einops.rearrange(out, 'b n c -> b c n')
        out = torch.reshape(out, (shape[0], self.out_dim, *shape[2:]))
        return out

def SelfAttentionBlock(dim, attention_dim, *, heads=8, groups=8):
    return Residual(
        nn.GroupNorm(groups, dim),
        SelfAttention(dim, dim, heads=heads, key_dim=attention_dim, value_dim=attention_dim),
    )
    
class Discriminator2D(ModelMixin, ConfigMixin):
    """
    This is a very simple discriminator architecture. It doesn't take any conditioning,
    not even the time step.
    """

    @register_to_config
    def __init__(
        self, 
        in_channels: int = 8,
        out_channels: int = 1,
        block_out_channels: Tuple[int] = (128, 256, 512, 1024, 1024, 1024),
        block_repeats: Tuple[int] = (2, 2, 2, 2, 2),
        downsample_blocks: Tuple[int] = (0, 1, 2),
        attention_blocks: Tuple[int] = (1, 2, 3, 4),
        hidden_channels: int = 4096,
        attention_dim: int = 128,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([])
        
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 7, padding=3)
        
        for i in range(0, len(block_out_channels) - 1):
            block_in = block_out_channels[i]
            block_out = block_out_channels[i + 1]
            block = nn.Sequential()
            for j in range(0, block_repeats[i]):
                if i in attention_blocks:
                    block.append(SelfAttentionBlock(block_in, attention_dim))
                block.append(ResnetBlock(block_in))
            if i in downsample_blocks:
                block.append(Downsample(block_in, block_out))
            elif block_in != block_out:
                block.append(nn.Conv2d(block_in, block_out, 1))
            self.blocks.append(block)

        # A simple MLP to make the final decision based on statistics from
        # the output of every block
        self.to_out = nn.Sequential(
            nn.Linear(2 * sum(block_out_channels[1:]), hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        
    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        
    def forward(self, x):
        x = self.conv_in(x)
        d = torch.zeros([x.shape[0], 0], device=x.device, dtype=x.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
            x_mean = einops.reduce(x, 'b c w h -> b c', 'mean')
            x_max = einops.reduce(x, 'b c w h -> b c', 'max')
            d = torch.cat([d, x_mean, x_max], dim=-1)
        return self.to_out(d)

