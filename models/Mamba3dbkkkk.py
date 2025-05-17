##############################################################
# % Author: Hibiki
# % Date:09/08/2024
###############################################################

import torch
import sys
import torch.nn as nn
import numpy as np
from functools import partial, reduce
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1
from .build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils import misc
from knn_cuda import KNN
import timm

### Mamba import start ###
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import _load_weights
from collections import namedtuple

from utils.bimamba_ssm.modules.mamba_simple import Mamba
from utils.bimamba_ssm.utils.generation import GenerationMixin
from utils.bimamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from utils.bimamba_ssm.rope import *

from utils.serialization import Point

try:
    from utils.bimamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

### Mamba import end ###

###ordering
import math
from utils.bimamba_ssm.z_order import *
import spconv.pytorch as spconv


######################################## Grouper ########################################
class DGCNN_Grouper(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )
        self.num_features = 128

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            # idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            # idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128)
        '''
        x = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class SimpleEncoder(nn.Module):
    def __init__(self, k=32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k
        self.knn = KNN(k=k, transpose_mode=False)
        self.num_features = embed_dims

    def forward(self, xyz, n_group):
        # 2048 divide into 128 * 32, overlap is needed
        if isinstance(n_group, list):
            n_group = n_group[-1]

        center = misc.fps(xyz, n_group)  # B G 3
        assert center.size(1) == n_group, f'expect center to be B {n_group} 3, but got shape {center.shape}'

        batch_size, num_points, _ = xyz.shape

        # _, idx = self.knn(xyz, center)
        idx = knn_point(self.group_size, xyz, center)

        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()

        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size

        features = self.embedding(neighborhood)  # B G C

        return center, features


######################################## Fold ########################################
class Fold(nn.Module):
    def __init__(self, in_channel, step, hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature

        patch_feature = torch.cat([
            g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
            token_feature
        ], dim=-1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step, 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc


######################################## Mamba ########################################
class GroupFeature(nn.Module):  # FPS + KNN
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # the first is the point itself
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, feat):
        '''
            input:
                xyz: B N 3
                feat: B N C
            ---------------------------
            output:
                neighborhood: B N K 3
                feature: B N K C
        '''
        batch_size, num_points, _ = xyz.shape  # B N 3 : 1 128 3
        C = feat.shape[-1]

        center = xyz
        # knn to get the neighborhood
        _, idx = self.knn(xyz, xyz)  # B N K : get K idx for every center
        assert idx.size(1) == num_points  # N center
        assert idx.size(2) == self.group_size  # K knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]  # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous()  # 1 128 8 3
        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :]  # BxNxK C 128x8 384   128*26*8
        assert neighborhood_feat.shape[-1] == feat.shape[-1]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size, feat.shape[-1]).contiguous()  # 1 128 8 384
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        return neighborhood, neighborhood_feat


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# Local Geometry Aggregation
class K_Norm(nn.Module):
    def __init__(self, out_dim, k_group_size, alpha, beta):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz, lc_x):
        # get knn xyz and feature
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x)  # B G K 3, B G K C

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)  # B G 1 C
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)  # B G 1 3

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)  # B G K 3

        B, G, K, C = knn_x.shape

        # Feature Expansion
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # B G K 2C

        # Affine
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat

        # Geometry Extraction
        knn_x_w = knn_x.permute(0, 3, 1, 2)  # B 2C G K

        return knn_x_w


# Max Pooling
class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0]  # B 2C G K -> B 2C G
        return lc_x


# Pooling
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)[0]  # B 2C G K -> B 2C G
        return lc_x


# Pooling
class K_Pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        e_x = torch.exp(knn_x_w)  # B 2C G K
        up = (knn_x_w * e_x).mean(-1)  # # B 2C G
        down = e_x.mean(-1)
        lc_x = torch.div(up, down)
        # lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1) # B 2C G K -> B 2C G
        return lc_x


# shared MLP
class Post_ShareMLP(nn.Module):
    def __init__(self, in_dim, out_dim, permute=True):
        super().__init__()
        self.share_mlp = torch.nn.Conv1d(in_dim, out_dim, 1)
        self.permute = permute

    def forward(self, x):
        # x: B 2C G mlp-> B C G  permute-> B G C
        if self.permute:
            return self.share_mlp(x).permute(0, 2, 1)
        else:
            return self.share_mlp(x)


## MLP
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


# K_Norm + K_Pool + Shared MLP
class LNPBlock(nn.Module):
    def __init__(self, lga_out_dim, k_group_size, alpha, beta, mlp_in_dim, mlp_out_dim, act_layer=nn.SiLU, drop_path=0., norm_layer=nn.LayerNorm, ):
        super().__init__()
        '''
        lga_out_dim: 2C
        mlp_in_dim: 2C
        mlp_out_dim: C
        x --->  (lga -> pool -> mlp -> act) --> x

        '''
        self.lga_out_dim = lga_out_dim

        self.lga = K_Norm(self.lga_out_dim, k_group_size, alpha, beta)
        self.kpool = K_Pool()
        self.mlp = Post_ShareMLP(mlp_in_dim, mlp_out_dim)
        self.pre_norm_ft = norm_layer(self.lga_out_dim)

        self.act = act_layer()

    def forward(self, center, feat):
        # feat: B G+1 C
        B, G, C = feat.shape
        cls_token = feat[:, 0, :].view(B, 1, C)
        feat = feat[:, 1:, :]  # B G C

        lc_x_w = self.lga(center, feat)  # B 2C G K

        lc_x_w = self.kpool(lc_x_w)  # B 2C G : 1 768 128

        # norm([2C])
        lc_x_w = self.pre_norm_ft(lc_x_w.permute(0, 2, 1))  # pre-norm B G 2C
        lc_x = self.mlp(lc_x_w.permute(0, 2, 1))  # B G C : 1 128 384

        lc_x = self.act(lc_x)

        lc_x = torch.cat((cls_token, lc_x), dim=1)  # B G+1 C : 1 129 384
        return lc_x


class Mamba3DBlock(nn.Module):
    def __init__(self,
                 num_query,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.SiLU,
                 norm_layer=nn.LayerNorm,
                 k_group_size=8,
                 alpha=100,
                 beta=1000,
                 bimamba_type="v2",
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(num_query + 1)

        self.k_group_size = k_group_size

        self.lfa = LNPBlock(lga_out_dim=dim * 2,
                            k_group_size=self.k_group_size,
                            alpha=alpha,
                            beta=beta,
                            mlp_in_dim=dim * 2,
                            mlp_out_dim=dim,
                            act_layer=act_layer,
                            drop_path=drop_path,
                            norm_layer=norm_layer,
                            )

        self.mixer = Mamba(num_query, dim, bimamba_type=bimamba_type)

    def shuffle_x(self, x, shuffle_idx):
        pos = x[:, None, 0, :]
        feat = x[:, 1:, :]
        shuffle_feat = feat[:, shuffle_idx, :]
        x = torch.cat([pos, shuffle_feat], dim=1)
        return x

    def mamba_shuffle(self, x):
        G = x.shape[1] - 1  #
        shuffle_idx = torch.randperm(G)
        # shuffle_idx = torch.randperm(int(0.4*self.num_group+1)) # 1-mask
        x = self.shuffle_x(x, shuffle_idx)  # shuffle

        x = self.mixer(self.norm2(x))  # layernorm->mamba

        x = self.shuffle_x(x, shuffle_idx)  # un-shuffle
        return x

    def forward(self, x):
        # x + norm(x)->lfa(x)->dropout

        # x = x + self.drop_path(self.lfa(center, self.norm1(x))) # x: 32 129 384. center: 32 128 3

        # x + norm(x)->mamba(x)->dropout
        # x = x + self.drop_path(self.mamba_shuffle(x))
        x = x + self.drop_path(self.mixer(self.norm2(x)))

        return x


class encoderBlock(nn.Module):
    def __init__(self,
                 num_query=512,
                 trans_dim=384,
                 drop_path_rate=0,
                 depth=6,
                 k_group_size=4,
                 bimamba_type="v4",
                 encoder_type='graph',
                 centernum=[1024, 512, 256], ):
        super().__init__()

        self.encoder_type = encoder_type
        self.centernum = centernum
        self.num_features = 128

        if self.encoder_type == 'graph':
            self.k = 8
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, trans_dim)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.SiLU(),
            nn.Linear(512, trans_dim)
        )
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Mamba3DBlock(
                num_query=num_query,
                dim=trans_dim,  #
                k_group_size=k_group_size,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,  #
                bimamba_type=bimamba_type,
            )
            for i in range(depth)])

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        self.knn = KNN(k=self.k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )
        self.layer5 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer6 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.lowerdim1 = nn.Sequential(
            nn.Linear(trans_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.num_features)
        )
        self.lowerdim2 = nn.Sequential(
            nn.Linear(trans_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.num_features)
        )
        self.alpha0 = nn.Parameter(torch.ones([1, 1, 1, 64]))
        self.alpha1 = nn.Parameter(torch.ones([1, 1, 1, 16]))
        self.alpha = nn.Parameter(torch.ones([4, 1, 1, self.num_features * 2]))

        self.beta0 = nn.Parameter(torch.zeros([1, 1, 1, 64]))
        self.beta1 = nn.Parameter(torch.zeros([1, 1, 1, 16]))
        self.beta = nn.Parameter(torch.zeros([4, 1, 1, self.num_features * 2]))

        self.kpool = K_Pool()

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature1(self, coor_q, x_q, coor_k, x_k, i):
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            # idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_k.transpose(-1, -2).contiguous()) # B G M
            # idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)

        x_k = x_k.transpose(2, 1).contiguous()
        x_q = x_q.transpose(2, 1).contiguous()
        coor_q = coor_q.transpose(-1, -2).contiguous()
        coor_k = coor_k.transpose(-1, -2).contiguous()

        neighborhood = coor_k.view(batch_size * num_points_k, -1)[idx, :]  # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points_q, k, 3).contiguous()  # 1 128 8 3
        neighborhood = neighborhood - coor_q.unsqueeze(2)

        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()

        x_q_bk = x_q.contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        # feature = torch.cat((feature - x_q, x_q), dim=1)

        knn_xyz = neighborhood
        knn_x = (feature - x_q).permute(0, 2, 3, 1).contiguous()
        mean_x = x_q_bk.unsqueeze(dim=-2)  # B G 1 C
        std_x = torch.std(knn_x - mean_x)
        mean_xyz = coor_q.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)  # B G 1 3
        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)  # B G K 3
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, x_q_bk.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # B G K 2C
        if i == 0:
            knn_x = self.alpha1 * knn_x + self.beta1
        else:
            knn_x = self.alpha[i + 1] * knn_x + self.beta[i + 1]
        knn_x_w = knn_x.permute(0, 3, 1, 2)  # B 2C G K

        return knn_x_w


    def get_graph_feature2(self, coor_q, x_q, coor_k, x_k, i):
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            # idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_k.transpose(-1, -2).contiguous()) # B G M
            # idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)

        x_k = x_k.transpose(2, 1).contiguous()
        x_q = x_q.transpose(2, 1).contiguous()
        coor_q = coor_q.transpose(-1, -2).contiguous()
        coor_k = coor_k.transpose(-1, -2).contiguous()

        neighborhood = coor_k.view(batch_size * num_points_k, -1)[idx, :]  # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points_q, k, 3).contiguous()  # 1 128 8 3
        neighborhood = neighborhood - coor_q.unsqueeze(2)

        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()

        x_q_bk = x_q.contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        # feature = torch.cat((feature - x_q, x_q), dim=1)

        knn_xyz = neighborhood
        knn_x = (feature - x_q).permute(0, 2, 3, 1).contiguous()
        mean_x = x_q_bk.unsqueeze(dim=-2)  # B G 1 C
        std_x = torch.std(knn_x - mean_x)
        mean_xyz = coor_q.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)  # B G 1 3
        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)  # B G K 3
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, x_q_bk.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # B G K 2C
        if i == 0:
            knn_x = self.alpha0 * knn_x + self.beta0
        else:

            knn_x = self.alpha[i - 1] * knn_x + self.beta[i - 1]
        knn_x_w = knn_x.permute(0, 3, 1, 2)  # B 2C G K

        return knn_x_w

    def mamba(self, center, x, i):
        bs = x.size(0)
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        cls_pos = self.cls_pos.expand(bs, -1, -1)

        centerpt_p = self.pos_embed(center)
        feature_x = self.input_proj(x)

        feature_x = torch.cat((cls_tokens, feature_x), dim=1)
        pos = torch.cat((cls_pos, centerpt_p), dim=1)

        x = self.blocks[i](feature_x + pos)

        return x

    def localfeature(self, coor, f, layer1, layer2, num, lowerdim, i, islast):
        f = self.get_graph_feature1(coor, f, coor, f, i)
        f = layer1(f)
        f = self.kpool(f)
        coor_q, f_q = self.fps_downsample(coor, f, num)
        f = self.get_graph_feature2(coor_q, f_q, coor, f, i)
        f = layer2(f)
        f = self.kpool(f)
        coor = coor_q
        f = self.mamba(coor.transpose(-1, -2).contiguous(), f.transpose(-1, -2).contiguous(), i)
        if islast == False:
            f = f[:, 1:, ]
            f = lowerdim(f).transpose(-1, -2).contiguous()
        return coor, f


    def forward(self, x):
        x = x.transpose(-1, -2).contiguous()
        coor = x
        f = self.input_trans(x)

        coor, f = self.localfeature(coor, f, self.layer1, self.layer2, self.centernum[0], self.lowerdim1, 0, False)

        coor, f = self.localfeature(coor, f, self.layer3, self.layer4, self.centernum[1], self.lowerdim2, 1, False)

        coor, f = self.localfeature(coor, f, self.layer5, self.layer6, self.centernum[2], self.lowerdim1, 2, True)

        return f


class decoderBlock(nn.Module):
    def __init__(self,
                 num_query=512,
                 trans_dim=384,
                 drop_path_rate=0,
                 depth=6,
                 k_group_size=4,
                 bimamba_type="v4",
                 order=["hilbert", "z", "z-trans"],
                 encoder_type='graph', ):
        super().__init__()

        self.encoder_type = encoder_type
        self.order = order
        self.num_features = 128
        self.depth = depth

        if self.encoder_type == 'graph':
            self.k = 8
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, trans_dim)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(trans_dim, 512),
            nn.SiLU(),
            nn.Linear(512, trans_dim)
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Mamba3DBlock(
                num_query=num_query,
                dim=trans_dim,  #
                k_group_size=k_group_size,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,  #
                bimamba_type=bimamba_type,
            )
            for i in range(depth)])

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        self.knn = KNN(k=self.k, transpose_mode=False)

        self.layer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(trans_dim * 2, trans_dim, kernel_size=1, bias=False),
                nn.GroupNorm(4, trans_dim),
                nn.LeakyReLU(negative_slope=0.2)
            )
            for i in range(depth)])

        self.alpha = nn.Parameter(torch.ones([3, 1, 1, trans_dim * 2]))
        self.beta = nn.Parameter(torch.zeros([3, 1, 1, trans_dim * 2]))
        self.kpool = K_Pool()

    def serialization(self, pos, feat, order="z", grid_size=0.02):
        bs, n_p, _ = pos.size()
        if not isinstance(order, list):
            order = [order]

        scaled_coord = pos / grid_size
        grid_coord = torch.floor(scaled_coord).to(torch.int64)
        min_coord = grid_coord.min(dim=1, keepdim=True)[0]
        grid_coord = grid_coord - min_coord

        batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

        point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
        point_dict = Point(**point_dict)
        point_dict.serialization(order=order)

        order = point_dict.serialized_order

        pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
        feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
        '''
        batch_idx = point_dict.batch
        grid_coord = point_dict.grid_coord

        sparse_shape = torch.add(
            torch.max(grid_coord, dim=0).values, 12
        ).tolist()

        sparse_conv_feat = spconv.SparseConvTensor(
            features = feat.reshape(bs*n_p, -1),

            indices=torch.cat(
                [batch_idx.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape = sparse_shape,
            batch_size = batch_idx[-1].tolist() + 1,
        )
        x = self.spnet(sparse_conv_feat)

        x = x.features.reshape(bs, n_p , -1)
        '''
        return pos, feat

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k, i):
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            # idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_k.transpose(-1, -2).contiguous()) # B G M
            # idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)

        x_k = x_k.transpose(2, 1).contiguous()
        x_q = x_q.transpose(2, 1).contiguous()
        coor_q = coor_q.transpose(-1, -2).contiguous()
        coor_k = coor_k.transpose(-1, -2).contiguous()

        neighborhood = coor_k.view(batch_size * num_points_k, -1)[idx, :]  # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points_q, k, 3).contiguous()  # 1 128 8 3
        neighborhood = neighborhood - coor_q.unsqueeze(2)

        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()

        x_q_bk = x_q.contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        # feature = torch.cat((feature - x_q, x_q), dim=1)

        knn_xyz = neighborhood
        knn_x = (feature - x_q).permute(0, 2, 3, 1).contiguous()
        mean_x = x_q_bk.unsqueeze(dim=-2)  # B G 1 C
        std_x = torch.std(knn_x - mean_x)
        mean_xyz = coor_q.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)  # B G 1 3
        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)  # B G K 3
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, x_q_bk.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # B G K 2C
        knn_x = self.alpha[i] * knn_x + self.beta[i]
        knn_x_w = knn_x.permute(0, 3, 1, 2)  # B 2C G K

        return knn_x_w

    def mamba(self, center, x, i):
        bs = x.size(0)
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        cls_pos = self.cls_pos.expand(bs, -1, -1)

        centerpt_p = self.pos_embed(center)
        feature_x = self.input_proj(x)

        feature_x = torch.cat((cls_tokens, feature_x), dim=1)
        pos = torch.cat((cls_pos, centerpt_p), dim=1)

        x = self.blocks[i](feature_x + pos)

        return x

    def forward(self, center, q):
        coor, f = center, q

        for i in range(self.depth):
            coor, f = self.serialization(coor, f, self.order[i])

            f = f.transpose(-1, -2).contiguous()
            f = self.get_graph_feature(coor.transpose(-1, -2).contiguous(), f, coor.transpose(-1, -2).contiguous(), f, i)
            f = self.layer[i](f)
            f = self.kpool(f)

            f = self.mamba(coor, f.transpose(-1, -2).contiguous(), i)
            f = f[:, 1:, ]

        return coor, f

    ######################################## PoinTr ########################################


@MODELS.register_module()
class Mamba3d(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        global_feature_dim = config.global_feature_dim
        self.num_query = query_num = config.num_query
        self.trans_dim = config.transformer_config.trans_dim
        self.encoder_dims = config.transformer_config.encoder_dims
        self.num_points = getattr(config, 'num_points', None)
        self.center_num = getattr(config, 'center_num', [512, 128])

        self.decoder_type = config.decoder_type

        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.k_group_size = config.center_local_k  # default=8
        self.bimamba_type = config.bimamba_type
        self.drop_path_rate = config.drop_path_rate
        self.depth = config.depth
        self.order = config.order
        self.numorder = config.numorder

        self.encoder = encoderBlock(num_query=self.center_num[-1], trans_dim=self.trans_dim, drop_path_rate=self.drop_path_rate,
                                    depth=self.depth, k_group_size=self.k_group_size, bimamba_type=self.bimamba_type, centernum=self.center_num, )

        self.decoder = decoderBlock(num_query=self.num_query, trans_dim=self.trans_dim, drop_path_rate=self.drop_path_rate,
                                    depth=self.depth, k_group_size=self.k_group_size, bimamba_type=self.bimamba_type, order=self.order)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_query)
        )

        if self.num_points is not None:
            self.factor = self.num_points // self.num_query
            assert self.num_points % self.num_query == 0
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point

        self.increase_dim1 = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim)
        )

        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, self.encoder_dims)
        )
        self.increase_dim2 = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt, epoch=1):
        pred_coarse, pred_fine = ret
        # recon loss
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)

        return loss_coarse, loss_fine

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz):
        bs = xyz.size(0)

        x = self.encoder(xyz)
        # x_t = self.encoder(centerpt, feature)

        global_feature = self.increase_dim1(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        # global_feature = torch.cat([global_feature[:, 0], global_feature[:, 1:].max(1)[0]+ global_feature[:, 1:].mean(1)[0]], dim=-1)

        coarse_point_cloud = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        coarse_inp = misc.fps(xyz, self.num_query // 2)  # B 128 3
        coarse_point_cloud = torch.cat([coarse_point_cloud, coarse_inp], dim=1)  # B 224+128 3?
        query_ranking = self.query_ranking(coarse_point_cloud)  # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True)  # b n 1
        coarse_point_cloud = torch.gather(coarse_point_cloud, 1, idx[:, :self.num_query].expand(-1, -1, coarse_point_cloud.size(-1)))

        q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse_point_cloud.size(1), -1),
                coarse_point_cloud], dim=-1))  # b n c

        # coarse_point_cloud, q = self.serialization(coarse_point_cloud, q, self.order)

        coarse_point_cloud, q = self.decoder(coarse_point_cloud, q)
        # q = q[:,1:,]

        B, M, C = q.shape

        global_feature = self.increase_dim2(q.transpose(1, 2)).transpose(1, 2)  # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024
        # global_feature = torch.cat([global_feature[:, 0], global_feature[:, 1:].max(1)[0] ], dim=-1)

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        rebuild_feature = self.reduce_map(rebuild_feature)  # B M C
        relative_xyz = self.decode_head(rebuild_feature)  # B M S 3
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3
        rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

        assert rebuild_points.size(1) == self.num_points
        assert coarse_point_cloud.size(1) == self.num_query

        ret = (coarse_point_cloud, rebuild_points)
        return ret