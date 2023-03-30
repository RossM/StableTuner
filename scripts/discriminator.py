import math
import torch
import torch.nn.functional as F
import einops, einops.layers.torch
import diffusers
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from typing import Tuple

def Downsample(dim, dim_out):
    return torch.nn.Conv2d(dim, dim_out, 4, 2, 1)

class Residual(torch.nn.Sequential):
    def forward(self, input):
        x = input
        for module in self:
            x = module(x)
        return x + input

def Block(dim, dim_out, *, kernel_size=3, groups=8):
    return torch.nn.Sequential(
        torch.nn.GroupNorm(groups, dim_out),
        torch.nn.SiLU(),
        torch.nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size//2),
    )

def ResnetBlock(dim, *, kernel_size=3, groups=8):
    return Residual(
        Block(dim, dim, kernel_size=kernel_size, groups=groups),
        Block(dim, dim, kernel_size=kernel_size, groups=groups),
    )
    
class SelfAttention(torch.nn.Module):
    def __init__(self, dim, out_dim, *, heads=4, key_dim=32, value_dim=32):
        super().__init__()
        self.dim = dim
        self.out_dim = dim
        self.heads = heads
        self.key_dim = key_dim

        self.to_k = torch.nn.Linear(dim, key_dim)
        self.to_v = torch.nn.Linear(dim, value_dim)
        self.to_q = torch.nn.Linear(dim, key_dim * heads)
        self.to_out = torch.nn.Linear(value_dim * heads, out_dim)

    def forward(self, x):
        shape = x.shape
        x = einops.rearrange(x, 'b c ... -> b (...) c')

        k = self.to_k(x)
        v = self.to_v(x)
        q = self.to_q(x)
        q = einops.rearrange(q, 'b n (h c) -> b (n h) c', h=self.heads)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            result = torch.nn.functional.scaled_dot_product_attention(q, k, v)
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
        torch.nn.GroupNorm(groups, dim),
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
        d_channels: int = 1024,
        hidden_channels: int = 1024,
        attention_dim: int = 64
    ):
        super().__init__()
        
        self.blocks = torch.nn.ModuleList([])
        self.linear_outs = torch.nn.ModuleList([])
        
        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], 7, padding=3)

        self.blocks.append(torch.nn.Sequential(
            ResnetBlock(block_out_channels[0]),
            ResnetBlock(block_out_channels[0]),
            Downsample(block_out_channels[0], block_out_channels[1]),
        ))
        self.linear_outs.append(torch.nn.Linear(block_out_channels[1] * 2, d_channels))
        self.blocks.append(torch.nn.Sequential(
            SelfAttentionBlock(block_out_channels[1], attention_dim),
            ResnetBlock(block_out_channels[1]),
            ResnetBlock(block_out_channels[1]),
            Downsample(block_out_channels[1], block_out_channels[2]),
        ))
        self.linear_outs.append(torch.nn.Linear(block_out_channels[2] * 2, d_channels))
        self.blocks.append(torch.nn.Sequential(
            SelfAttentionBlock(block_out_channels[2], attention_dim),
            ResnetBlock(block_out_channels[2]),
            ResnetBlock(block_out_channels[2]),
            Downsample(block_out_channels[2], block_out_channels[3]),
        ))
        self.linear_outs.append(torch.nn.Linear(block_out_channels[3] * 2, d_channels))
        self.blocks.append(torch.nn.Sequential(
            SelfAttentionBlock(block_out_channels[3], attention_dim),
            ResnetBlock(block_out_channels[3]),
            ResnetBlock(block_out_channels[4]),
        ))
        self.linear_outs.append(torch.nn.Linear(block_out_channels[4] * 2, d_channels))
        self.blocks.append(torch.nn.Sequential(
            SelfAttentionBlock(block_out_channels[4], attention_dim),
            ResnetBlock(block_out_channels[4]),
            ResnetBlock(block_out_channels[5]),
        ))
        self.linear_outs.append(torch.nn.Linear(block_out_channels[5] * 2, d_channels))
        
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(5 * d_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )
        
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        
    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        
    def forward(self, x):
        x = self.conv_in(x)
        d = torch.zeros([1, 0], device=x.device, dtype=x.dtype)
        for block, linear_out in zip(self.blocks, self.linear_outs):
            x = block(x)
            x_mean = x.mean([-2, -1])
            x_max, _ = einops.rearrange(x, 'b c h w -> b c (h w)').max(-1)
            out = linear_out(torch.cat([x_mean, x_max], dim=-1))
            d = torch.cat([d, out], dim=-1)
        return self.to_out(d)

