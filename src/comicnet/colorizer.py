import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from dataclasses import dataclass, asdict
from einops import rearrange

class UpSample(nn.Module):
    """ UpSampling block using PixelShuffle """
    def __init__(self, filters=64):
        super().__init__()
        self.conv = nn.Conv2d(filters, filters * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

## DownSampling block
class DownSample(nn.Module):
    """ DownSampling block using PixelUnshuffle """
    def __init__(self, filters=64):
        super().__init__()
        self.conv = nn.Conv2d(filters, filters // 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)

    def forward(self, x):
        """ SHAPE (B, C, H, W) -> SHAPE (B, C/4, H/2, W/2) """
        x = self.conv(x)
        x = self.pixel_unshuffle(x)
        return x

# Custom LayerNormalization
class BiasFree_LayerNorm(nn.Module):
    """ Bias-Free Layer Normalization """
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        x = x.contiguous() 
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """ With-Bias Layer Normalization """
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        x = x.contiguous() 
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    """ Layer Normalization supporting two types: BiasFree and WithBias """
    def __init__(self, dim, LayerNorm_type, out_4d=True):
        super().__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
        self.out_4d = out_4d

    def to_3d(self, x):
        # Convert (B, C, H, W) to (B, H*W, C)
        if len(x.shape) == 3:
            return x
        elif len(x.shape) == 4:
            return rearrange(x, 'b c h w -> b (h w) c')
        else:
            raise ValueError("Input must be a 3D or 4D tensor")
    
    def to_4d(self, x, h, w):
        # Convert (B, H*W, C) to (B, C, H, W)
        if len(x.shape) == 4:
            return x
        elif len(x.shape) == 3:
            return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        else:
            raise ValueError("Input must be a 3D or 4D tensor")

    def forward(self, x):
        if self.out_4d:
            h, w = x.shape[-2:]
            return self.to_4d(self.body(self.to_3d(x)), h, w)
        else:
            return self.body(x)
        
class RepConv3(nn.Module):
    def __init__(self, in_channels, out_channels, groups, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.deploy = deploy
        self.reparam = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        if not deploy:
            self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
            self.conv_1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), groups=groups)
            self.conv_3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=groups)
            self.conv_1x1_branch = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)
            self.conv_3x3_branch = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False)
        else:
            self._delete_branches()

    def _delete_branches(self):
        for name in ['conv_3x3','conv_1x1','conv_1x3','conv_3x1', 'conv_1x1_branch', 'conv_3x3_branch']:
            if hasattr(self, name):
                delattr(self, name)

    def fuse(self, delete_branches=True):
        if self.deploy:
            return
        # Extract weights and biases
        conv_3x3_w, conv_3x3_b = self.conv_3x3.weight, self.conv_3x3.bias
        conv_1x1_w, conv_1x1_b = self.conv_1x1.weight, self.conv_1x1.bias
        conv_1x3_w, conv_1x3_b = self.conv_1x3.weight, self.conv_1x3.bias
        conv_3x1_w, conv_3x1_b = self.conv_3x1.weight, self.conv_3x1.bias
        conv_1x1_branch_w, conv_3x3_branch_w = self.conv_1x1_branch.weight, self.conv_3x3_branch.weight
        # Pad the smaller kernels to 3x3
        conv_1x1_w_pad = F.pad(conv_1x1_w, [1, 1, 1, 1])
        conv_1x3_w_pad = F.pad(conv_1x3_w, [0, 0, 1, 1])
        conv_3x1_w_pad = F.pad(conv_3x1_w, [1, 1, 0, 0])
        if self.groups == 1:
            conv_1x1_3x3_w_pad = F.conv2d(conv_3x3_branch_w, conv_1x1_branch_w.permute(1, 0, 2, 3))
        else:
            w_slices = []
            conv_1x1_branch_w_T = conv_1x1_branch_w.permute(1, 0, 2, 3)
            in_channels_per_group = self.in_channels // self.groups
            out_channels_per_group = self.out_channels // self.groups
            for g in range(self.groups):
                # Slice the transposed 1x1 weights for this group's channels
                conv_1x1_branch_w_T_slice = conv_1x1_branch_w_T[:, g*in_channels_per_group:(g+1)*in_channels_per_group, :, :]
                # Slice the 3x3 weights for this group's output channels
                conv_3x3_branch_w_slice = conv_3x3_branch_w[g*out_channels_per_group:(g+1)*out_channels_per_group, :, :, :]
                w_slices.append(F.conv2d(conv_3x3_branch_w_slice, conv_1x1_branch_w_T_slice))
            conv_1x1_3x3_w_pad = torch.cat(w_slices, dim=0)
        # Fuse weights and biases
        conv_w = conv_3x3_w + conv_1x1_w_pad + conv_1x3_w_pad + conv_3x1_w_pad + conv_1x1_3x3_w_pad
        if conv_3x3_b is None:
            conv_3x3_b = torch.zeros(self.out_channels, device=conv_w.device)
        conv_b = conv_3x3_b + conv_1x1_b + conv_1x3_b + conv_3x1_b
        self.reparam.weight.data.copy_(conv_w)
        self.reparam.bias.data.copy_(conv_b)
        # Delete the original branches
        if delete_branches:
            self._delete_branches()
        # Set deploy flag
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.reparam(x)
        else:
            return self.conv_3x3(x) + self.conv_1x1(x) + self.conv_1x3(x) + self.conv_3x1(x) + self.conv_3x3_branch(self.conv_1x1_branch(x))

from .monarch_attn import MonarchAttention

@dataclass
class RepAttnConfig:
    dim: int
    num_heads: int = 8
    block_size: int = 16
    num_steps: int = 2
    pad_type: str = "pre"
    impl: str = "torch"
    deploy: bool = False

class RepAttn(nn.Module):
    """ Re-parameterizable Attention Block using MonarchAttention as the core attention mechanism."""
    def __init__(self, dim, num_heads=8, block_size=14, num_steps=1, pad_type="pre", impl="torch", deploy=False):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.monarch_attn = MonarchAttention(
            block_size=block_size,
            num_steps=num_steps,
            pad_type=pad_type,
            impl=impl
        )
        if deploy:
            self.attn_fn = self.monarch_attn
        else:
            self.attn_fn = self.common_attn
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.deploy = deploy

    def common_attn(self, q, k, v):
        """ Scaled Dot-Product Attention """
        scale = (q.shape[-1]) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out

    @torch.no_grad()
    def fuse(self):
        if not self.deploy:
            self.attn_fn = self.monarch_attn
            self.deploy = True

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        attn_out = self.attn_fn(q, k, v)
        attn_out = rearrange(attn_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        out = self.proj(attn_out)
        return out
    
@dataclass
class FFNConfig:
    dim: int
    expansion_factor: int = 1
    deploy: bool = False

class RepFFN(nn.Module):
    def __init__(self, dim, expansion_factor=1, deploy=False):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = RepConv3(dim, hidden_features, groups=1, deploy=deploy)
        self.dwconv = RepConv3(hidden_features, hidden_features*2, groups=hidden_features, deploy=deploy)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    @torch.no_grad()
    def fuse(self):
        self.project_in.fuse()
        self.dwconv.fuse()  


    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class SkipConnection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim*2, dim, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x
    
class RepTransformerBlock(nn.Module):
    def __init__(self, rep_attn_cfg: RepAttnConfig, ffn_cfg: FFNConfig, norm_type='WithBias'):
        super().__init__()
        self.rep_attn = RepAttn(**asdict(rep_attn_cfg))
        self.rep_ffn = RepFFN(**asdict(ffn_cfg))
        self.norm1 = LayerNorm(rep_attn_cfg.dim, norm_type)
        self.norm2 = LayerNorm(rep_attn_cfg.dim, norm_type)

    @torch.no_grad()
    def fuse(self):
        self.rep_attn.fuse()
        self.rep_ffn.fuse()

    def forward(self, x):
        x = x + self.rep_attn(self.norm1(x))
        x = x + self.rep_ffn(self.norm2(x))
        return x
    
class Block(nn.Module):
    def __init__(self, num_block, rep_attn_cfg: RepAttnConfig, ffn_cfg: FFNConfig, norm_type='WithBias'):
        super().__init__()
        self.num_block = num_block
        self.blocks = nn.ModuleList([
            RepTransformerBlock(rep_attn_cfg, ffn_cfg, norm_type) for _ in range(num_block)
        ])

    @torch.no_grad()
    def fuse(self):
        for block in self.blocks:
            block.fuse()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class ColorComicNet(nn.Module):
    """ Main model implementation """
    def __init__(self, input_shape=(3, 1024, 1024), output_channels=3, deploy=False, dims=[48, 96, 192, 384], num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 2, 4], bias=True, last_act=None):
        super().__init__()
        assert len(dims) == len(num_blocks) == len(num_heads), "Length of dims, num_blocks and num_heads must be the same"
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.deploy = deploy
        self.dims = dims
        self.num_blocks = num_blocks
        self.bias = bias
        self.num_heads = num_heads

        # Extractor
        self.stem = nn.Conv2d(input_shape[0], dims[0], kernel_size=7, stride=4, padding=3, bias=bias)
        
        # Encoder
        layers = []
        down_convs = []
        for idx in range(len(dims)):
            attn_cfg, ffn_cfg = self.build_cfg(dims[idx], num_heads[idx])
            block = Block(num_blocks[idx], attn_cfg, ffn_cfg, norm_type='WithBias')
            if idx < len(dims) - 1:
                down_convs.append(DownSample(dims[idx]))
            layers.append(block)
        self.bottleneck = layers[-1] # Last encoder layer as bottleneck
        self.encoder = nn.ModuleList(layers[:-1])
        self.downsample = nn.ModuleList(down_convs)
        
        # Decoder
        layers = []
        up_convs = []
        skip_connections = []
        for idx in range(len(dims)-2, -1, -1):
            attn_cfg, ffn_cfg = self.build_cfg(dims[idx], num_heads[idx])
            # print(f"Decoder layer {idx}: shape {l_shape}")
            up_conv = UpSample(dims[idx+1])
            block = Block(num_blocks[idx], attn_cfg, ffn_cfg, norm_type='WithBias')
            layers.append(block)
            up_convs.append(up_conv)
            skip_connections.append(SkipConnection(dims[idx]))
        self.decoder = nn.ModuleList(layers)
        self.up_sample = nn.ModuleList(up_convs)
        self.skip = nn.ModuleList(skip_connections)

        # Head
        self.head = nn.Sequential(
            RepConv3(dims[0], dims[0]//2, 1, deploy=deploy),
            nn.GELU(),
            nn.Conv2d(dims[0]//2, output_channels, kernel_size=1, bias=bias),
        )
        self.last_act = last_act if last_act is not None else nn.Identity()

    @torch.no_grad()
    def fuse(self):
        for block in self.encoder:
            block.fuse()
        self.bottleneck.fuse()
        for block in self.decoder:
            block.fuse()
        for conv in self.head:
            if isinstance(conv, RepConv3):
                conv.fuse()
    
    def build_cfg(self, dim, head):
        # RepAttn config
        attn_cfg = RepAttnConfig(
            dim=dim,
            num_heads=head,
            block_size=12,
            num_steps=2,
            pad_type="pre",
            impl="torch",
            deploy=self.deploy
        )
        ## FFN config
        ffn_cfg = FFNConfig(
            dim=dim,
            expansion_factor=1,
        )
        return attn_cfg, ffn_cfg

    def forward(self, x):
        """ 
        x: (B, C, H, W)
        """
        res = x
        x = self.stem(x)
        feats = []
        for blk, down in zip(self.encoder, self.downsample):
            x = blk(x)
            feats.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for blk, up, skip in zip(self.decoder, self.up_sample, self.skip):
            x = up(x)
            cur_feat = feats.pop()
            x = skip(x, cur_feat)
            x = blk(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.head(x) + res
        x = self.last_act(x)
        return x
    
# Example model configuration
MODEL_CFG = {
    'input_shape': (3, 512, 512),
    'dims': [24, 48, 96, 192],
    'num_blocks': [1, 2, 2, 4],
    'num_heads': [1, 2, 4, 8],
    'bias': True,
    'last_act': nn.Tanh(),
    'deploy': False
}