import math
import torch
import torch.nn.functional as F
import einops, einops.layers.torch
import diffusers
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin

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
    
class Discriminator(ModelMixin, ConfigMixin):
    """
    This is a very simple discriminator architecture. It doesn't take any conditioning,
    not even the time step.
    """

    def __init__(self, dim=128, attention_dim=64):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(8, dim, 7, padding=3),
            ResnetBlock(dim),
            ResnetBlock(dim),
            Downsample(dim, dim*2),
            SelfAttentionBlock(dim*2, attention_dim),
            ResnetBlock(dim*2),
            ResnetBlock(dim*2),
            Downsample(dim*2, dim*4),
            SelfAttentionBlock(dim*4, attention_dim),
            ResnetBlock(dim*4),
            ResnetBlock(dim*4),
            Downsample(dim*4, dim*8),
            SelfAttentionBlock(dim*8, attention_dim),
            ResnetBlock(dim*8),
            ResnetBlock(dim*8),
            SelfAttentionBlock(dim*8, attention_dim),
            ResnetBlock(dim*8),
            ResnetBlock(dim*8),
            SelfAttentionBlock(dim*8, attention_dim),
            torch.nn.Conv2d(dim*8, dim*8, 1),
            torch.nn.GroupNorm(8, dim*8),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim*8, 1, 1),
            #einops.layers.torch.Reduce('b c h w -> b', 'mean'),
        )
        
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        
    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        
    def forward(self, x):
        if self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint_sequential(self.model, 10, x)
        else:
            return self.model(x)

