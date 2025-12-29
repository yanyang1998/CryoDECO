# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

# from transformers import ViTConfig, ViTModel,ViTForImageClassification
from timm.models.vision_transformer import VisionTransformer, _cfg, Block
from timm.models.swin_transformer import swin_s3_tiny_224, _cfg
from timm.layers.helpers import to_2tuple
from timm.layers import PatchEmbed, format

# from blitz.modules import BayesianLinear
# from blitz.utils import variational_estimator
import torch.nn.functional as F
import numpy as np
import random
from safetensors.torch import load_file
import os

__all__ = [
    'vit_tiny',
    'vit_small',
    'swin_vit_tiny',
    'vit_base',
    'vit_huge',
    'vit_giant',
    'vit_hMLP_base',
    'vit_conv_small',
    'vit_conv_base',
]


class CryosolverEncoder(nn.Module):
    def __init__(self, encoder_type, feature_dim, std_z_init, conf_table_dim=None, pretrained_model_path=None,
                 fuse_only_table=0.0,
                 finetune_strategy='all', finetune_layer_num=3, use_fused_encoder=False,
                 fuse_type='concat', gradient_checkpointing=False, warmup_epochs=10, min_fuse_only_table=0.0,
                 update_ratio_type='cosine',feature_take_indices=None):
        super(CryosolverEncoder, self).__init__()
        self.use_fused_encoder = use_fused_encoder
        self.feature_dim = feature_dim
        self.conf_table_dim = conf_table_dim
        self.fuse_type = fuse_type
        self.std_z_init = std_z_init
        self.feature_take_indices = feature_take_indices

        if encoder_type == 'vit_small':
            self.vit_backbone = vit_small(
                gradient_checkpointing=gradient_checkpointing,
                dynamic_img_size=True, patch_size=14, use_bn=True,
                stop_grad_conv1=True,feature_take_indices=self.feature_take_indices)
        else:
            self.vit_backbone = vit_base(gradient_checkpointing=gradient_checkpointing,
                                         dynamic_img_size=True, patch_size=14, use_bn=True,
                                         stop_grad_conv1=True,feature_take_indices=self.feature_take_indices)
        in_channels = self.vit_backbone.head.in_features
        self.vit_backbone.head = Classifier(input_dim=in_channels, output_dim=self.feature_dim,
                                            num_linear_layers=1,
                                            add_vit_blocks_num=0)
        self.vit_backbone.norm = nn.Identity()
        if use_fused_encoder:
            # self.conftable = conftable
            if fuse_type == 'gate':
                self.fuse = GatingMechanism(self.feature_dim, self.conf_table_dim, ratio_only_B=fuse_only_table,
                                            warmup_epochs=warmup_epochs, min_ratio_B=min_fuse_only_table,
                                            update_ratio_type=update_ratio_type)

        if finetune_strategy != 'random_initial':
            if finetune_strategy == 'vit_block':
                block_len = len(self.vit_backbone.blocks)
                block_finetune_id = [str(block_len - 1 - bi) for bi in
                                     range(finetune_layer_num)]
                for name, param in self.vit_backbone.named_parameters():
                    '''finetune the last fc layer and the last transformer layer'''
                    if name.startswith('block'):
                        if name.split('.')[1] in block_finetune_id:
                            continue

                    elif name.startswith('head'):
                        continue
                    param.requires_grad = False
            if finetune_strategy == 'mlp_layer':
                for name, param in self.vit_backbone.named_parameters():
                    '''finetune the last fc layer'''
                    if not name.startswith('head'):
                        param.requires_grad = False

            pretrained_state_dict = load_file(os.path.join(pretrained_model_path, 'model.safetensors'),
                                              device='cpu')
            my_state_dict = {}
            for k in list(pretrained_state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('base_encoder') and not k.startswith(
                        'base_encoder.%s' % 'head'):
                    # remove prefix
                    my_state_dict[k[len("base_encoder."):]] = pretrained_state_dict[k]
                if k.startswith('base_encoder.head.'):
                    my_state_dict['projector' + k[len('base_encoder.head'):]] = pretrained_state_dict[k]
                # delete renamed or unused k
                # del state_dict[k]
            msg = self.vit_backbone.load_state_dict(my_state_dict, strict=False)
            # print(msg)

    def forward(self, in_dict, pose_only=False, table_features=None):

        y_real_resized = in_dict['y_real_resized']
        batch_size = y_real_resized.shape[0]
        pfm_features,intermediates = self.vit_backbone(y_real_resized)
        conf_dict = {'z': pfm_features,'intermediates':intermediates if self.feature_take_indices is not None else None}

        if self.use_fused_encoder:
            if pose_only:
                table_features = self.std_z_init * torch.randn((batch_size, self.conf_table_dim),
                                                               dtype=torch.float32,
                                                               device=pfm_features.device
                                                               )
            # else:
            #     table_features = conftable(in_dict)['z']
            if self.fuse_type == 'gate':
                conf_features = self.fuse(pfm_features, table_features)
            else:
                conf_features = torch.cat([table_features, pfm_features], dim=1)

            # conf_dict = {'z': conf_features, 'z_table': table_features}
            conf_dict['z_table'] = table_features
            conf_dict['z'] = conf_features
            # conf_dict = {'z': table_features, 'z_table': table_features}

        return conf_dict

    def update_fuse_only_table(self, epoch):
        if self.use_fused_encoder:
            if self.fuse_type == 'gate':
                self.fuse.update_ratio_only_B(epoch)


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=True, use_bn=False, pretrain_mode=False,feature_take_indices=None, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.pretrain_mode = pretrain_mode
        self.build_2d_sincos_position_embedding()
        self.feature_take_indices=feature_take_indices

        # weight initialization
        for name, m in self.named_modules():
            if use_bn and isinstance(m, Block):
                # module.norm1 = BN_bnc(module.norm1.normalized_shape)
                m.norm2 = BN_bnc(m.norm2.normalized_shape)

            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def forward_features(self, x):
        intermediates = []
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:

            x,intermediates = my_checkpoint_seq(self.blocks, x,feature_take_indices=self.feature_take_indices)
            # intermediates = [self.forward_head(feature) for feature in intermediates] if self.feature_take_indices is not None else intermediates
            # intermediates = [self.forward_head(feature) for feature in intermediates] if self.feature_take_indices is not None else intermediates
        else:
            if self.feature_take_indices is not None:
                for i in range(len(self.blocks)):
                    x = self.blocks[i](x)
                    if i in self.feature_take_indices:
                        intermediates.append(x[:, 0])
                        # intermediates.append(self.forward_head(x))
            else:
                x = self.blocks(x)
        x = self.norm(x)

        return x,intermediates
    def forward(self, x: torch.Tensor) :
        x,intermediates = self.forward_features(x)
        x = self.forward_head(x)
        return x,intermediates



class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, num_linear_layers=3, mlp_dim=1024, add_vit_blocks_num=0):
        super().__init__()
        self.add_vit_blocks_num = add_vit_blocks_num
        if add_vit_blocks_num > 0:
            self.vit_blocks = nn.Sequential(*[
                Block(
                    dim=input_dim,
                    num_heads=12,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_norm=False,
                    init_values=None,
                    proj_drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    # act_layer=nn.GELU,
                    # mlp_layer=Mlp,
                )
                for i in range(add_vit_blocks_num)])

        mlp = []
        for l in range(num_linear_layers):
            if l == 0:
                dim1 = input_dim
            else:
                dim1 = mlp_dim
                mlp_dim = mlp_dim // 2
            dim2 = output_dim if l == num_linear_layers - 1 else mlp_dim

            # mlp.append(nn.Linear(dim1, dim2, bias=False))
            mlp.append(nn.Linear(dim1, dim2))

            if l < num_linear_layers - 1:
                mlp.append(nn.SyncBatchNorm(dim2))
                mlp.append(nn.ReLU(inplace=True))
        self.mlp_classifier = nn.Sequential(*mlp)
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        if self.add_vit_blocks_num > 0:
            x_ = self.vit_blocks(x)
        else:
            x_ = x
        x_ = self.norm(x_)
        x_ = self.mlp_classifier(x_)
        return x_


class Classifier_2linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 2)


    def forward(self, x):
        x_ = self.linear1(x)
        x_ = F.relu(x_)
        x_ = self.linear2(x_)
        return x_


class Classifier_3linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.SyncBatchNorm(1024)
        self.linear2 = nn.Linear(1024, 256)
        self.bn2 = nn.SyncBatchNorm(256)
        self.linear3 = nn.Linear(256, output_dim)


    def forward(self, x):
        x_ = self.linear1(x)
        x_ = self.bn1(x_)
        x_ = F.relu(x_)
        x_ = self.linear2(x_)
        x_ = self.bn2(x_)
        x_ = F.relu(x_)
        x_ = self.linear3(x_)
        return x_


class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=False,
                 **kwargs):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        if in_chans == 3:
            embed_dim = embed_dim // 2
            self.proj2 = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))
            nn.init.uniform_(self.proj2.weight, -val, val)
            nn.init.zeros_(self.proj2.bias)

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 1, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if C == 3:
            x_f = x[:, 1:, :, :]
            x = x[:, 0:1, :, :]

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = format.nchw_to(x, 'NHWC')
        x = self.norm(x)
        if C == 3:
            x_f = self.proj2(x_f)
            if self.flatten:
                x_f = x_f.flatten(2).transpose(1, 2)  # BCHW -> BNC
            else:
                x_f = format.nchw_to(x_f, 'NHWC')
            x_f = self.norm(x_f)
            x = torch.cat([x, x_f], dim=-1)

        return x


class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.SyncBatchNorm,
                 flatten=False, **kwargs):
        super().__init__()
        assert patch_size == 16, 'hMLP_Stem only supports patch size of 16'
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.flatten = flatten
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim // 4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim // 4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim // 4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                          ])

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = format.nchw_to(x, 'NHWC')
        return x


def vit_tiny(patch_size=14, in_chans=1, gradient_checkpointing=False, use_bn=True, stop_grad_conv1=False,
             pretrain_mode=False,feature_take_indices=None, **kwargs):
    model = VisionTransformerMoCo(
        in_chans=in_chans,
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_bn=use_bn, stop_grad_conv1=stop_grad_conv1,
        pretrain_mode=pretrain_mode,feature_take_indices=feature_take_indices, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_small(patch_size=14, in_chans=1, gradient_checkpointing=False, use_bn=True, stop_grad_conv1=False,
              pretrain_mode=False,feature_take_indices=None, **kwargs):
    model = VisionTransformerMoCo(
        in_chans=in_chans,
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_bn=use_bn, stop_grad_conv1=stop_grad_conv1,
        pretrain_mode=pretrain_mode,feature_take_indices=feature_take_indices, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_base(patch_size=14, in_chans=1, gradient_checkpointing=False, use_bn=True, stop_grad_conv1=False,
             pretrain_mode=False,feature_take_indices=None, **kwargs):
    model = VisionTransformerMoCo(in_chans=in_chans,
                                  patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                  qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), use_bn=use_bn,
                                  stop_grad_conv1=stop_grad_conv1, pretrain_mode=pretrain_mode,feature_take_indices=feature_take_indices, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_large(patch_size=14, in_chans=1, gradient_checkpointing=False, use_bn=True, stop_grad_conv1=False,
              pretrain_mode=False,feature_take_indices=None, **kwargs):
    model = VisionTransformerMoCo(in_chans=in_chans,
                                  patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                                  qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), use_bn=use_bn,
                                  stop_grad_conv1=stop_grad_conv1, pretrain_mode=pretrain_mode,feature_take_indices=feature_take_indices, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_huge(patch_size=14, in_chans=1, gradient_checkpointing=False, use_bn=True, stop_grad_conv1=False, **kwargs):
    model = VisionTransformerMoCo(in_chans=in_chans,
                                  patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
                                  qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), use_bn=use_bn,
                                  stop_grad_conv1=stop_grad_conv1, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_giant(patch_size=14, in_chans=1, gradient_checkpointing=False, use_bn=True, stop_grad_conv1=False, **kwargs):
    model = VisionTransformerMoCo(in_chans=in_chans,
                                  patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48 / 11,
                                  qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), use_bn=use_bn,
                                  stop_grad_conv1=stop_grad_conv1, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_conv_small(patch_size=14, in_chans=1, gradient_checkpointing=False, use_bn=True, stop_grad_conv1=False,
                   **kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(in_chans=in_chans, patch_size=patch_size, use_bn=use_bn,
                                  stop_grad_conv1=stop_grad_conv1,
                                  embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_conv_base(patch_size=14, in_chans=1, gradient_checkpointing=False, use_bn=True, stop_grad_conv1=False,
                  **kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(in_chans=in_chans, patch_size=patch_size, use_bn=use_bn,
                                  stop_grad_conv1=stop_grad_conv1,
                                  embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def vit_hMLP_base(patch_size=16, in_chans=1, gradient_checkpointing=False, use_bn=True, stop_grad_conv1=False,
                  **kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(in_chans=in_chans, patch_size=patch_size, use_bn=use_bn,
                                  stop_grad_conv1=stop_grad_conv1,
                                  embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=hMLP_stem, **kwargs)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
    model.default_cfg = _cfg()
    return model


def swin_vit_tiny(**kwargs):
    # minus one ViT block
    model = swin_s3_tiny_224(**kwargs)
    model.default_cfg = _cfg()
    return model


class BN_bnc(nn.SyncBatchNorm):
    """
    BN_bnc: BatchNorm1d on hidden feature with (B,N,C) dimension
    """

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B * N, C)  # (B,N,C) -> (B*N,C)
        x = super().forward(x)  # apply batch normalization
        x = x.reshape(B, N, C)  # (B*N,C) -> (B,N,C)
        return x


def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)




from itertools import chain
from torch.utils.checkpoint import checkpoint


def my_checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True,
feature_take_indices=None
):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """

    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)
    intermediates=[]
    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state,
                       use_reentrant=False)
        if feature_take_indices is not None and end in feature_take_indices:
            # intermediates.append(x)
            intermediates.append(x[:, 0])
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x,intermediates


class SharedCNN(nn.Module):
    def __init__(self, resolution, depth, channels, kernel_size, in_channels, nl=nn.ReLU, coord_conv=False,
                 dropout=False, radial_average=False, conf_regressor_settings=None):
        """
        resolution: int
        depth: int
        channels: int
        kernel_size: int
        in_channels: int
        coord_conv: bool
        dropout: bool
        radial_average: bool
        """
        super(SharedCNN, self).__init__()

        cnn = []

        if radial_average:
            cnn.append(RadialAverager())
            final_size = resolution // 2
        else:
            final_size = resolution
        if coord_conv:
            cnn.append(AddCoords(final_size))
            in_channels = in_channels + 3
        else:
            in_channels = in_channels
        out_channels = channels
        for i in range(depth):
            ks = min(kernel_size, final_size)
            if dropout and i > 0:
                cnn.append(nn.Dropout2d())
            cnn.append(nn.Conv2d(in_channels, out_channels, ks, padding='same', padding_mode='reflect'))
            in_channels = out_channels
            cnn.append(nl())
            if 2 * in_channels <= 2048:
                out_channels = 2 * in_channels
            else:
                out_channels = in_channels
            if dropout:
                cnn.append(nn.Dropout2d())
            cnn.append(nn.Conv2d(in_channels, out_channels, ks, padding='same', padding_mode='reflect'))
            in_channels = out_channels
            cnn.append(nn.GroupNorm(channels, in_channels))
            if i < depth - 1:
                cnn.append(nl())
            else:
                cnn.append(nn.Tanh())
            if final_size // 2 > 0:
                cnn.append(nn.AvgPool2d(2))
                final_size = final_size // 2

        self.cnn = nn.Sequential(*cnn)

        self.final_size = final_size
        self.final_channels = in_channels

        if conf_regressor_settings is not None:
            self.conf_regressor = ConfRegressor(self.final_channels, self.final_size, conf_regressor_settings['z_dim'],
                                                conf_regressor_settings['std_z_init'],
                                                conf_regressor_settings['variational'],
                                                # n_classes=conf_regressor_settings['n_classes']
                                                )
        else:
            self.conf_regressor = None

    def forward(self, y_real):
        """
        y_real: [..., d, D - 1, D - 1]

        output: [..., final_channels, final_size, final_size]
        """
        in_dims = y_real.shape[:-3]
        d = y_real.shape[-3]
        res = y_real.shape[-2]
        result = self.cnn(y_real.reshape(np.prod(in_dims), d, res, res)).reshape(
            *in_dims, self.final_channels, self.final_size, self.final_size
        )
        if self.conf_regressor is not None:
            result = self.conf_regressor(result)
        return result


class ConfTable(nn.Module):
    def __init__(self, n_imgs, conf_table_dim, variational, std_z_init, conf_init=None, use_generated_features=False):
        """
        n_imgs: int
        z_dim: int
        variational: bool
        """
        super(ConfTable, self).__init__()
        self.variational = variational
        if conf_init is not None:
            self.conf_init = torch.tensor(conf_init).float() * std_z_init
            self.projection_head = nn.Linear(conf_init.shape[-1], conf_table_dim)
        else:
            self.conf_init = torch.tensor(
                std_z_init * np.random.randn(n_imgs, conf_table_dim)
            ).float()
            self.projection_head = None
        if use_generated_features:
            self.table_conf = nn.Parameter(self.conf_init, requires_grad=False)
        else:
            self.table_conf = nn.Parameter(self.conf_init, requires_grad=True)
        if variational:
            logvar_init = torch.tensor(np.ones((n_imgs, conf_table_dim))).float()
            self.table_logvar = nn.Parameter(logvar_init, requires_grad=True)

    def initialize(self, conf):
        """
        conf: [n_imgs, z_dim] (numpy)
        """
        state_dict = self.state_dict()
        state_dict['table_conf'] = torch.tensor(conf).float()
        self.load_state_dict(state_dict)

    def forward(self, in_dict):
        """
        in_dict: dict
            index: [batch_size]
            y: [batch_size(, n_tilts), D, D]
            y_real: [batch_size(, n_tilts), D - 1, D - 1]
            R: [batch_size(, n_tilts), 3, 3]
            t: [batch_size(, n_tilts), 2]
            tilt_index: [batch_size( * n_tilts)]

        output: dict
            z: [batch_size, z_dim]
            z_logvar: [batch_size, z_dim] if variational and not pose_only
        """
        conf = self.table_conf[in_dict['index_p']]
        if self.projection_head is not None:
            conf = self.projection_head(conf)
        conf_dict = {'z': conf}
        if self.variational:
            logvar = self.table_logvar[in_dict['index_p']]
            conf_dict['z_logvar'] = logvar
        return conf_dict

    def reset(self):
        state_dict = self.state_dict()
        state_dict['table_conf'] = self.conf_init / 10.0
        self.load_state_dict(state_dict)


class ConfRegressor(nn.Module):
    def __init__(self, channels, kernel_size, z_dim, std_z_init, variational):
        """
        channels: int
        kernel_size: int
        z_dim: int
        std_z_init: float
        variational: bool
        """
        super(ConfRegressor, self).__init__()
        self.z_dim = z_dim
        self.variational = variational
        self.std_z_init = std_z_init
        if variational:
            out_features = 2 * z_dim
        else:
            out_features = z_dim
        self.out_features = out_features
        self.regressor = nn.Conv2d(channels, out_features, kernel_size, padding='valid')

    def forward(self, shared_features):
        """
        shared_features: [..., channels, kernel_size, kernel_size]

        output: dict
            z: [..., z_dim]
            z_logvar: [..., z_dim] if variational and not pose_only
        """
        in_dim = shared_features.shape[:-3]
        c = shared_features.shape[-3]
        ks = shared_features.shape[-2]
        z_full = self.regressor(shared_features.reshape(-1, c, ks, ks)).reshape(np.prod(in_dim), self.out_features)
        if self.variational:
            conf_dict = {
                'z': z_full[:, :self.z_dim],
                'z_logvar': nn.Tanh()(z_full[:, self.z_dim:] / 10.) * 10.
            }
        else:
            conf_dict = {
                'z': z_full
            }
        return conf_dict


class AddCoords(nn.Module):
    def __init__(self, resolution, radius_channel=True):
        """
        resolution: int
        radius_channel: bool
        """
        super(AddCoords, self).__init__()
        self.radius_channel = radius_channel

        xx_ones = torch.ones([1, resolution], dtype=torch.int32)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(resolution, dtype=torch.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, resolution], dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(resolution, dtype=torch.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 1, 2)
        yy_channel = yy_channel.permute(0, 3, 1, 2)

        xx_channel = xx_channel.float() / (resolution - 1)
        yy_channel = yy_channel.float() / (resolution - 1)

        xx_channel = xx_channel - 0.5
        yy_channel = yy_channel - 0.5

        self.xx_channel = nn.Parameter(xx_channel, requires_grad=False)
        self.yy_channel = nn.Parameter(yy_channel, requires_grad=False)

        self.radius_calc = None
        if radius_channel:
            self.radius_calc = nn.Parameter(
                torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2)),
                requires_grad=False
            )

    def forward(self, x):
        """
        x: [batch_size, d, D - 1, D - 1]

        output: [batch_size, d + 2/3, D - 1, D - 1]
        """
        batch_size = x.shape[0]

        xx_channel = self.xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = self.yy_channel.repeat(batch_size, 1, 1, 1)

        out = torch.cat([x, xx_channel, yy_channel], dim=1)

        if self.radius_channel:
            out = torch.cat([out, self.radius_calc.repeat(batch_size, 1, 1, 1)], dim=1)

        return out


class RadialAverager(nn.Module):
    def __init__(self):
        super(RadialAverager, self).__init__()

    @staticmethod
    def forward(y_real):
        """
        y_real: [batch_size, d, D - 1, D - 1]

        output: [batch_size, d, (D - 1) // 2, (D - 1) // 2]
        """
        res = y_real.shape[-1]
        y_real_avg = torch.mean(torch.cat([
            y_real[..., None],
            torch.flip(y_real, [-1, -2])[..., None],
            torch.flip(torch.transpose(y_real, -2, -1), [-2])[..., None],
            torch.flip(torch.transpose(y_real, -2, -1), [-1])[..., None]
        ], -1), -1)
        return y_real_avg[..., :res // 2, :res // 2]


class GatingMechanism(nn.Module):

    def __init__(self, feature_dim: int, in_dim: int, ratio_only_B=0.0, update_ratio_type=None,
                 warmup_epochs=10, min_ratio_B=0.0):
        """
        初始化门控机制。

        Args:
            feature_dim (int): 输入特征的维度。两个特征的维度必须相同。
        """
        super(GatingMechanism, self).__init__()
        self.feature_dim = feature_dim
        self.ratio_only_B = ratio_only_B
        self.max_ratio_B = ratio_only_B

        # 门控网络：它接收拼接后的两个特征，因此输入维度是 feature_dim * 2
        # 输出维度是 feature_dim，与单个特征的维度相同
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        if in_dim != feature_dim:
            self.projection_head = nn.Linear(in_dim, feature_dim)
        else:
            self.projection_head = nn.Identity()
        self.update_ratio_type = update_ratio_type
        self.warmup_epochs = warmup_epochs
        self.min_ratio_B = min_ratio_B

    def update_ratio_only_B(self, epoch):
        if self.update_ratio_type is not None:
            if epoch < self.warmup_epochs:
                if epoch<0:
                    pass
                elif self.update_ratio_type == 'linear':
                    self.ratio_only_B = self.max_ratio_B * (
                                self.warmup_epochs - epoch) / self.warmup_epochs + self.min_ratio_B
                elif self.update_ratio_type == 'cosine':
                    self.ratio_only_B = self.min_ratio_B + 0.5 * (
                                1 + math.cos(math.pi * epoch / self.warmup_epochs)) * (
                                                    self.max_ratio_B - self.min_ratio_B)
            else:
                self.ratio_only_B = self.min_ratio_B
        return self.ratio_only_B

    def forward(self, feature_pfm: torch.Tensor, feature_in: torch.Tensor, ) -> torch.Tensor:
        """
        前向传播。

        Args:
            feature_A (torch.Tensor): 第一个特征张量，形状为 (batch_size, feature_dim)。
            feature_B (torch.Tensor): 第二个特征张量，形状为 (batch_size, feature_dim)。

        Returns:
            torch.Tensor: 融合后的特征张量，形状为 (batch_size, feature_dim)。
        """
        feature_in = self.projection_head(feature_in)
        if self.ratio_only_B > 0 and random.random() < self.ratio_only_B:
            # 如果 ratio_only_B 大于随机数，则只返回 feature_B
            return feature_in
        else:
            # if feature_pfm.shape != feature_B.shape:
            #     raise ValueError("两个特征张量的形状必须相同")

            # 1. 将两个特征在最后一个维度上拼接
            # 形状变为 (batch_size, feature_dim * 2)
            concatenated_features = torch.cat([feature_pfm, feature_in], dim=-1)

            # 2. 将拼接后的特征输入门控网络，生成门控信号 g
            # g 的形状为 (batch_size, feature_dim)，其值在 (0, 1) 之间
            gate_signal = self.gate_network(concatenated_features)

            # 3. 应用门控机制进行融合
            # g * feature_A: 保留 feature_A 的部分
            # (1 - g) * feature_B: 保留 feature_B 的部分
            fused_feature = gate_signal * feature_pfm + (1 - gate_signal) * feature_in

            return fused_feature
