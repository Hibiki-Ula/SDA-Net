##############################################################
# % Author: Hibiki
# % Date:09/08/2024
###############################################################

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial, reduce
from extensions.chamfer_dist import ChamferDistanceL1
from .build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils import misc
from utils.hypercdloss import hyperV2
import sys


from .mamba_layer import MambaBlock
from utils.PCM_utils import MLP, serialization, _init_weights, index_points
from utils.PointMLP_layers import ConvBNReLU1D, LocalGrouper, PreExtraction, PreExtraction_Replace, LocalGrouperWithSer,\
    PosExtraction, get_activation, PointNetFeaturePropagation

#####################################################################################
from timm.models.layers import DropPath, trunc_normal_

class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block 
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'onebyone'    
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'  
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'  
            self_attn_combine_style = 'concat'    
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()        
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()      

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
  
        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        
        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q
    
class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
        cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
        k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length)
        return q

class PointTransformerDecoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q
    

class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

#####################################################################################
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
            ], dim = -1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc
#####################################################################################
class block(nn.Module):
    def __init__(self, 
                 pre_blocks = [ 1, 1, 1, 1 ],
                 mamba_blocks = [ 1, 1, 1, 1 ],
                 k_neighbors = [ 12, 12, 12, 12 ],
                 reducers = [ 2, 2, 2, 1 ],
                 use_xyz = True,
                 encoder_dims  = 384, 
                 bimamba_type = 'v2', 
                 drop_path_rate = 0.1, 
                 mamba_layers_orders = 'z', 
                 grid_size = 0.02,
                 en_dims = [ 384, 384, 384, 384, 384 ],
                 de_dims = [ 384, 384, 384, 384, 384 ],

                 **kwargs):
        super(block, self).__init__()
        self.pre_blocks = pre_blocks
        self.mamba_blocks = mamba_blocks
        self.k_neighbors = k_neighbors
        self.kstride = [1, 1, 1, 1,1,1]
        self.reducers = reducers
        self.use_xyz = use_xyz
        self.embed_dim = encoder_dims
        self.stages = len(self.pre_blocks)
        self.bimamba_type = bimamba_type
        self.drop_path_rate = drop_path_rate
        self.mamba_layers_orders = mamba_layers_orders  
        self.grid_size = grid_size
        last_channel = self.embed_dim
        out_channel = last_channel
        

        self.order = "original"

        self.local_grouper_list = nn.ModuleList()
        self.blocks_list = nn.ModuleList()
        self.mamba_block = nn.ModuleList()
        self.encode_list = nn.ModuleList()
        self.pos_proj = nn.ModuleList()
        self.reduce_map = nn.ModuleList()
        self.decode_head = nn.ModuleList()


        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.mamba_blocks))]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        mamba_layer_idx = 0

        for i in range(len(self.pre_blocks)):
            pre_block_num = self.pre_blocks[i]
            mamba_block_num = self.mamba_blocks[i]
            kneighbor = self.k_neighbors[i]
            reduce = self.reducers[i]
            # append local_grouper_list
            self.encode_list.append(
                    PointNetFeaturePropagation(de_dims[i] + en_dims[i + 1], de_dims[i + 1],
                                            blocks=pre_block_num, res_expansion=1.0,
                                            bias=True, activation= 'relu')
            )


            local_grouper = LocalGrouper(last_channel, reduce, kneighbor, self.use_xyz, normalize="anchor",
                                                k_stride = self.kstride[i])  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)

                # append pre_block_list
            if pre_block_num == 0:
                        # only max pooling
                pre_block_module = PreExtraction_Replace(last_channel, out_channel, pre_block_num,
                                                                groups= 1,
                                                                res_expansion= 'relu',
                                                                bias=False, activation='relu', use_xyz= self.use_xyz) 
            else:
                pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num,
                                                        groups = 1,
                                                        res_expansion = 1.0,
                                                        bias=False, activation='relu', use_xyz= self.use_xyz) 
            self.blocks_list.append(pre_block_module)


            mamba_block = nn.Sequential()

            for n_mamba in range(mamba_block_num):
                mamba_block_module = MambaBlock(dim=out_channel, layer_idx=mamba_layer_idx, bimamba_type = self.bimamba_type, drop_path=inter_dpr[mamba_layer_idx])
                mamba_block.append(mamba_block_module)
                mamba_layer_idx += 1
            self.mamba_block.append(mamba_block)

            self.pos_proj.append(
                nn.Linear(3, self.embed_dim, bias=False)
            )  
  
        self.mamba_block.apply(
            partial(_init_weights, n_layer=mamba_layer_idx, )
        )



    def serialization_func(self, p, x, x_res, order, layers_outputs=[]):
        if order == self.order:
            return p, x, x_res
        else:
            p, x, x_res = serialization(p, x, x_res=x_res, order=order,
                                        layers_outputs=layers_outputs,
                                        grid_size=self.grid_size)
            self.order = order
            return p, x, x_res
        
        
    def unet_encoder(self, xyz_list, fea_list):
        fea = fea_list[0]
        for i in range(len(self.encode_list)):
            fea = self.encode_list[i](xyz_list[i + 1], xyz_list[i], fea_list[i + 1], fea).permute(0, 2, 1).contiguous()

        return xyz_list, fea_list, fea
        

    def encoder_forward(self, xyz, fea):
        xyz_list, fea_list = [xyz], [fea]
        mamba_layer_idx = 0
        fea_res = None

        for i in range(self.stages):
            xyz, fea, fea_res = self.serialization_func(xyz, fea, fea_res,self.mamba_layers_orders[mamba_layer_idx])

            xyz, fea, fea_res = self.local_grouper_list[i](xyz, fea,fea_res)  # [b,g,3]  [b,g,k,d]

            # 动态池化接cnn1d
            fea = self.blocks_list[i](fea)  # [b,d,g]
            fea = fea.permute(0, 2, 1).contiguous()

            for layer in self.mamba_block[i]:
                fea = fea + self.pos_proj[i](xyz)
                fea, fea_res = layer(fea, fea_res)
                mamba_layer_idx += 1

            xyz_list.append(xyz)
            fea_list.append(fea)

        xyz_list.reverse()
        fea_list.reverse()

        return self.unet_encoder(xyz_list, fea_list)
    

    
    def forward(self, xyz, fea, globalfea = None):
            return self.encoder_forward(xyz, fea)



#####################################################################################


@MODELS.register_module()
class pointcloudmamba(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.Dataname = config.Dataname
        self.num_points = config.num_points
        self.num_query = config.num_query
        self.pre_blocks = config.pre_blocks
        self.en_mamba_blocks = config.en_mamba_blocks
        self.en_k_neighbors = config.en_k_neighbors
        self.reducers = config.reducers
        self.use_xyz = config.use_xyz
        self.embed_dim = config.encoder_dims
        self.embedding = ConvBNReLU1D(3, self.embed_dim, bias=False, activation='relu')
        self.stages = len(self.pre_blocks)
        self.bimamba_type = config.bimamba_type
        self.drop_path_rate = config.drop_path_rate
        self.en_mamba_layers_orders = config.en_mamba_layers_orders    
        self.grid_size = config.grid_size
        self.en_dims = config.encoder_channel_list
        self.de_dims = config.decoder_channel_list

        self.decoder = PointTransformerDecoderEntry(config.decoder_config)



        self.encoder = block(
            pre_blocks = self.pre_blocks,
            mamba_blocks = self.en_mamba_blocks,
            k_neighbors = self.en_k_neighbors,
            reducers = self.reducers,
            use_xyz = self.use_xyz,
            encoder_dims  = self.embed_dim, 
            bimamba_type = self.bimamba_type, 
            drop_path_rate = self.drop_path_rate, 
            mamba_layers_orders = self.en_mamba_layers_orders, 
            grid_size = self.grid_size,
            en_dims = self.en_dims,
            de_dims = self.de_dims,
        )

        self.increase_dim = nn.Sequential(
            nn.Linear(self.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024))
        
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, self.num_query * 3)
        )
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(1024  + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, self.embed_dim)
        )

        self.increase_dim2 = nn.Sequential(
            nn.Conv1d(self.embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.reduce_map = nn.Linear(1027+ self.embed_dim, self.embed_dim)

        assert self.num_points % self.num_query == 0
        self.decode_head = SimpleRebuildFCLayer(self.embed_dim * 2, step=self.num_points // self.num_query) 
        self.build_loss_func()


    def build_loss_func(self):
        if self.Dataname == 'PCN' :
            self.loss_func = ChamferDistanceL1()
        elif self.Dataname == 'ShapeNet':
            self.loss_func = hyperV2()
        else:
            raise NotImplementedError(f'Loss do not support {self.Dataname}')

    
    def get_loss(self, ret, gt, epoch=1):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret
        
        assert pred_fine.size(1) == gt.size(1)

        # denoise loss
        idx = knn_point(self.num_points // self.num_query, gt, denoised_coarse) # B n k 
        denoised_target = index_points(gt, idx) # B n k 3 
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        # recon loss
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

    def forward(self, xyz):
        fea = self.embedding(xyz.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # B,D,N

        xyz_list, fea_list, fea = self.encoder(xyz, fea)
        B, _ ,C = fea.shape

        global_feature = self.increase_dim(fea) # B 1024 N 
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        coarse = self.coarse_pred(global_feature).reshape(B, -1, 3)

        coarse_inp = misc.fps(xyz, self.num_query//2) # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3?
        query_ranking = self.query_ranking(coarse) # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            # add denoise task
            # first pick some point : 64?
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1) # B 256+64 3?
            denoise_length = 64     

            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c

            # forward decoder

            q = self.decoder(q=q, v=fea_list[0], q_pos=coarse, v_pos=xyz_list[0], denoise_length=denoise_length)


        else:
            denoise_length =0
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c
            q = self.decoder(q=q, v=fea_list[0], q_pos=coarse, v_pos=xyz_list[0])

        B, M ,C = q.shape
        global_feature = self.increase_dim2(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse], dim=-1)  # B M 1027 + C
        
        rebuild_feature = self.reduce_map(rebuild_feature)  # B M C
        relative_xyz = self.decode_head(rebuild_feature)  # B M S 3
        rebuild_points = (relative_xyz + coarse.unsqueeze(-2))  # B M S 3

        
        if self.training:
            # split the reconstruction and denoise task
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse[:, -denoise_length:].contiguous()

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            ret = (coarse, rebuild_points)
            return ret
