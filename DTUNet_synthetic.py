import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.utils.checkpoint as checkpoint
import os
from timm.models import create_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from functools import partial
from timm.models.vision_transformer import _cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seed = 3407
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load DATA
data = sio.loadmat(r'./data/PPNMM30dB75pure.mat')
abundance_GT = torch.from_numpy(data["A"])  # true abundance
original_HSI = torch.from_numpy(data["Y"])  # HSI data
b_true = torch.from_numpy(data["b"])        # nonlinear coefficients

# VCA_endmember and GT
VCA_endmember = data["M1"]
GT_endmember = data["M"]

endmember_init = torch.from_numpy(VCA_endmember).unsqueeze(2).unsqueeze(3).float()
GT_init = torch.from_numpy(GT_endmember).unsqueeze(2).unsqueeze(3).float()

band_Number = original_HSI.shape[0]
endmember_number, pixel_number = abundance_GT.shape

# Hyperparameters
col, C, gamma = 100, 108, 0.8
batch_size = 1
EPOCH = 600
alpha = 0.1
drop_out = 0.
learning_rate = 0.003

# Define original_HSI and abundance_GT
original_HSI = torch.reshape(original_HSI, (band_Number, col, col))
abundance_GT = torch.reshape(abundance_GT, (endmember_number, col, col))

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.mul(self.shared_MLP(x)*x, 1)
        return x



# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=1))
    abundance_input = torch.reshape(
        abundance_input.squeeze(0), (endmember_number, col, col)
    )
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input


# endmember normalization
def norm_endmember(endmember_input, endmember_GT):
    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT


# plot abundance
def plot_abundance(abundance_input, abundance_GT_input):
    plt.figure(figsize=(60, 25))
    for i in range(0, endmember_number):
        plt.subplot(2, endmember_number, i + 1)
        plt.pcolor(abundance_input[i, :, :], cmap='jet')
        plt.colorbar(shrink=.83)

        plt.subplot(2, endmember_number, endmember_number + i + 1)
        plt.pcolor(abundance_GT_input[i, :, :], cmap='jet')
        plt.colorbar(shrink=.83)

    plt.show()


# plot endmember
def plot_endmember(endmember_input, endmember_GT):
    plt.figure(figsize=(13, 2.5), dpi=150)
    for i in range(0, endmember_number):
        plt.subplot(1, endmember_number, i + 1)
        plt.plot(endmember_input[:, i], label="Extracted")
        plt.plot(endmember_GT[:, i], label="GT")
    plt.legend()
    plt.show()


# change the index of abundance and endmember
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT):
    RMSE_matrix = np.zeros((endmember_number, endmember_number))
    SAD_matrix = np.zeros((endmember_number, endmember_number))
    RMSE_index = np.zeros(endmember_number).astype(int)
    SAD_index = np.zeros(endmember_number).astype(int)
    RMSE_abundance = np.zeros(endmember_number)
    SAD_endmember = np.zeros(endmember_number)

    for i in range(0, endmember_number):
        for j in range(0, endmember_number):
            RMSE_matrix[i, j] = AbundanceRmse(
                abundance_input[i, :, :], abundance_GT_input[j, :, :]
            )
            SAD_matrix[i, j] = SAD_distance(endmember_input[:, i], endmember_GT[:, j])

        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])

    abundance_input[np.arange(endmember_number), :, :] = abundance_input[
        RMSE_index, :, :
    ]
    endmember_input[:, np.arange(endmember_number)] = endmember_input[:, SAD_index]

    return abundance_input, endmember_input, RMSE_abundance, SAD_endmember


class load_data(torch.utils.data.Dataset):
    def __init__(self, img, gt, transform=None):
        self.img = img.float()
        self.gt = gt.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img, self.gt

    def __len__(self):
        return 1


# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
    return rmse



# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim

'''Dilate'''
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,q,k,v):
        #B, C//3, H, W
        B,d,H,W = q.shape
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1 ,H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 2, 3)  #B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)# B, C, H, W
        # qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C//self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        qkv = self.qkv(x.clone()).reshape(B, 3, self.num_dilation, C//self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        #num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C//self.num_dilation, H, W).permute(1, 0, 3, 4, 2 )
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])# B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DilateBlock(nn.Module):
    "Implementation of Dilate-attention block"
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        #B, C, H, W
        return x


class GlobalAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim,  num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalBlock(nn.Module):
    """
    Implementation of Transformer
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_block=False):
        super().__init__()
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    """
    def __init__(self, img_size=224, in_chans=3, hidden_dim=16,
                 patch_size=4, embed_dim=96, patch_way=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.img_size = img_size
        assert patch_way in ['overlaping', 'nonoverlaping', 'pointconv'],\
            "the patch embedding way isn't exist!"
        if patch_way == "nonoverlaping":
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif patch_way == "overlaping":
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 224x224
                nn.BatchNorm2d(hidden_dim),
                nn.GELU( ),
                nn.Conv2d(hidden_dim, int(hidden_dim*2), kernel_size=3, stride=2,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim*2)),
                nn.GELU( ),
                nn.Conv2d(int(hidden_dim*2), int(hidden_dim*4), kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim*4)),
                nn.GELU( ),
                nn.Conv2d(int(hidden_dim*4), embed_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),  # 56x56
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.GELU( ),
                nn.Conv2d(hidden_dim, int(hidden_dim*2), kernel_size=1, stride=1,
                          padding=0, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim*2)),
                nn.GELU( ),
                nn.Conv2d(int(hidden_dim*2), int(hidden_dim*4), kernel_size=3, stride=2,
                          padding=1, bias=False),  # 56x56
                nn.BatchNorm2d(int(hidden_dim*4)),
                nn.GELU( ),
                nn.Conv2d(int(hidden_dim*4), embed_dim, kernel_size=1, stride=1,
                          padding=0, bias=False),   # 56x56
            )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # B, C, H, W
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """
    def __init__(self, in_channels, out_channels, merging_way, cpe_per_satge, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert merging_way in ['conv3_2', 'conv2_2', 'avgpool3_2', 'avgpool2_2'], \
            "the merging way is not exist!"
        self.cpe_per_satge = cpe_per_satge
        if merging_way == 'conv3_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        elif merging_way == 'conv2_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        elif merging_way == 'avgpool3_2':
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        else:
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        if self.cpe_per_satge:
            self.pos_embed = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)

    def forward(self, x):
        #x: B, C, H ,W
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
        return x


class Dilatestage(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """
    def __init__(self, dim, depth, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):

        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            DilateBlock(dim=dim, num_heads=num_heads,
                        kernel_size=kernel_size, dilation=dilation,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


class Globalstage(nn.Module):
    """ A basic Transformer layer for one stage."""
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):

        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            GlobalBlock(dim=dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim*2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


class Dilateformer(nn.Module):
    def __init__(self, img_size=100, patch_size=1, in_chans=224, num_classes=1000, embed_dim=108,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], kernel_size=3, dilation=[1, 2, 3],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 merging_way='conv3_2',
                 patch_way='overlaping',
                 dilate_attention=[True, True, False, False],
                 downsamples=[True, True, True, False],
                 cpe_per_satge=False, cpe_per_block=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        '''add'''
        self.patch_norm = True
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        #patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, patch_way=patch_way)
        # self.patch_embed = PatchEmbed(
        #     patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if dilate_attention[i_layer]:
                stage = Dilatestage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    kernel_size=kernel_size,
                                    dilation=dilation,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
            else:
                stage = Globalstage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
            self.stages.append(stage)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

@register_model
def dilateformer_tiny(pretrained=True, **kwargs):
    model = Dilateformer(depths=[2, 1, 2, 1], embed_dim=C, num_heads=[3, 6, 12, 24], **kwargs)
    model.default_cfg = _cfg()
    return model


# DTU-Net
class multiStageUnmixing(nn.Module):
    def __init__(self):
        super(multiStageUnmixing, self).__init__()

        self.encodelayer = nn.Sequential(nn.Softmax())

        self.decoderlayer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )


        """Spectral Branch"""
        self.spectral = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),

            nn.Dropout(0.2),

            ChannelAttention(in_planes=band_Number-2, ratio=2),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),

            ChannelAttention(in_planes=band_Number-4, ratio=2),

            nn.MaxPool3d((2, 1, 1)),

            nn.Dropout(0.2),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),

            ChannelAttention(in_planes=(band_Number-4)//2-2, ratio=2),

            nn.MaxPool3d((2, 1, 1)),

            nn.Dropout(0.2),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),
            ChannelAttention(in_planes=((band_Number-4)//2-2)//2-2, ratio=2),
            nn.MaxPool3d((2, 1, 1)),

            nn.Dropout(0.2),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),

            ChannelAttention(in_planes=(((band_Number-4)//2-2)//2-2)//2-2, ratio=2),

            nn.MaxPool3d((2, 1, 1)),

            nn.Dropout(0.2),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 1), stride=1,
                      padding=0),

            nn.LeakyReLU(0.2),

            ChannelAttention(in_planes=((((band_Number-4)//2-2)//2-2)//2-2)//2-2, ratio=2),

            nn.MaxPool3d((2, 1, 1)),

            nn.Dropout(0.2),

            nn.Conv2d(in_channels=(((((band_Number-4)//2-2)//2-2)//2-2)//2-2)//2, out_channels=endmember_number,
                      kernel_size=3, stride=1, padding=1)
        )
        self.upscale = nn.Sequential(
            nn.Linear((C*8)//endmember_number, col*col),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(endmember_number*2, endmember_number, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(endmember_number, endmember_number*2, kernel_size=1, stride=1, padding=0)
        )
        self.spectral_attention = nn.Sequential(
            nn.Conv2d(endmember_number*2, endmember_number, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(endmember_number, endmember_number*2, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=endmember_number*2, out_channels=endmember_number, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(endmember_number),
            nn.Dropout(drop_out),

            nn.Conv2d(in_channels=endmember_number, out_channels=endmember_number, kernel_size=3,
                      stride=1, padding=1),

        )

        self.dilate = create_model(
        'dilateformer_tiny',
        pretrained=False,
        num_classes=endmember_number,
        drop_block=None
        )

        # decoder2
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # decoder2
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(5, 1, 1), stride=1, padding=0),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(5, 1, 1), stride=1, padding=0),
            nn.Hardtanh(),
            nn.MaxPool3d((2, 1, 1)),
            nn.Dropout(0.2),

            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(5, 1, 1), stride=1, padding=0),
            nn.Hardtanh(),
            nn.MaxPool3d((2, 1, 1)),
            nn.Dropout(0.2)
        )
        self.upscale2 = nn.Sequential(
            nn.Linear(((band_Number*3-8)//2-4)//2, col*col)
        )

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        cls_emb = self.dilate(x)
        cls_emb = cls_emb.view(1, endmember_number, -1)
        abu_est = self.upscale(cls_emb).view(1, endmember_number, col, col)
        spectral_feature = self.spectral(x)

        """Spatial-Spectral Feature Fusion"""
        spectral_spatial_feature = torch.cat((abu_est, spectral_feature), 1)
        spectral_spatial_feature2 = self.conv(spectral_spatial_feature)
        feature_attetion_weight = self.spectral_attention(spectral_spatial_feature2)
        spectral_attetion_feature = torch.mul((spectral_spatial_feature2 * feature_attetion_weight), 1)
        attetion_feature = spectral_attetion_feature + spectral_spatial_feature
        layer1out = self.layer4(attetion_feature)
        en_result1 = self.encodelayer(gamma * layer1out)
        de_result1 = self.decoderlayer4(en_result1)
        de_result2 = torch.multiply(de_result1, de_result1)
        de_result3 = torch.cat((x, de_result1, de_result2), 1)

        middle = self.decoder2(de_result3)
        middle1 = middle.view(1, ((band_Number*3-8)//2-4)//2, col, col)
        middle2 = middle1.flatten(2).transpose(1, 2)
        middle3 = self.avgpool(middle2.transpose(1, 2))
        middle4 = middle3.flatten(1)
        b1 = middle4.view(1, 1, -1)
        b2 = self.upscale2(b1)
        b = b2.view(1, 1, col, col)
        de_result = de_result1 + torch.multiply(b, de_result2)

        return en_result1, de_result, b


# SAD loss of reconstruction
def reconstruction_SADloss(output, target):

    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


MSE = torch.nn.MSELoss(size_average=True)

# load data
train_dataset = load_data(
    img=original_HSI, gt=abundance_GT, transform=transforms.ToTensor()
)
# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False
)

net = multiStageUnmixing().cuda()
# weight init
net.apply(net.weights_init)

# decoder weight init by VCA
model_dict = net.state_dict()
model_dict["decoderlayer4.0.weight"] = endmember_init


net.load_state_dict(model_dict)


# optimizer
def set_optimizer(model, lr_base, decay):
    slow_params = map(id, model.decoderlayer4.parameters())
    else_params = filter(lambda addr: id(addr) not in slow_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': model.decoderlayer4.parameters(), 'lr': 0.02, 'weight_decay': 1e-5},
        {'params': else_params}], lr=lr_base, weight_decay=decay
    )
    return optimizer

optimizer = set_optimizer(model=net, lr_base=learning_rate, decay=8e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)
apply_clamp_inst1 = NonZeroClipper()

train_losses = []
abundance_losses = []
mse_losses = []

'''Train the model'''
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        scheduler.step()
        x = x.cuda()
        net.train().cuda()

        en_abundance, reconstruction_result, b_result = net(x)

        abundanceLoss = reconstruction_SADloss(x, reconstruction_result)

        MSELoss = MSE(x, reconstruction_result)

        ALoss = abundanceLoss
        BLoss = MSELoss

        total_loss = ALoss + (alpha * BLoss)

        optimizer.zero_grad()
        net.decoderlayer4.apply(apply_clamp_inst1)
        total_loss.backward()
        optimizer.step()

        train_losses.append(total_loss.item())
        abundance_losses.append(ALoss.item())
        mse_losses.append(BLoss.item())

        if epoch % 100 == 0:
            print(
                "Epoch:",
                epoch,
                "| Abundanceloss: %.4f" % ALoss.cpu().data.numpy(),
                "| MSEloss: %.4f" % (alpha * BLoss).cpu().data.numpy(),
                "| total_loss: %.4f" % total_loss.cpu().data.numpy(),
            )

net.eval()


en_abundance, reconstruction_result, b_result = net(x)

decoder_para = net.state_dict()["decoderlayer4.0.weight"].cpu().numpy()
decoder_para = np.mean(np.mean(decoder_para, -1), -1)

en_abundance, abundance_GT = norm_abundance_GT(en_abundance, abundance_GT)
decoder_para, GT_endmember = norm_endmember(decoder_para, GT_endmember)

en_abundance, decoder_para, RMSE_abundance, SAD_endmember = arange_A_E(
    en_abundance, abundance_GT, decoder_para, GT_endmember
)
print("RMSE", RMSE_abundance)
print("mean_RMSE", RMSE_abundance.mean())
print("endmember_SAD", SAD_endmember)
print("mean_SAD", SAD_endmember.mean())

plot_abundance(en_abundance, abundance_GT)
plot_endmember(decoder_para, GT_endmember)


'''evaluate the nonlinear coefficient estimation'''
b_predict = torch.flatten(b_result, 1)
b_true = b_true.cuda()
b_true = b_true.view(1, col, col)
b_true = torch.flatten(b_true, 1)
MSE_b = torch.sum((b_predict - b_true) ** 2, 1)/10000
MSE_b = MSE_b.item()
RMSE_b = pow(MSE_b, 0.5)
print("RMSE_b", RMSE_b)
print(b_result)
b_predict = b_predict.cpu().detach().numpy()

# save the results
sio.savemat('DTU_PPNMM30dB75pure_results.mat', {'M0': decoder_para, 'A0': en_abundance, 'b0': b_predict})

