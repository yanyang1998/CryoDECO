import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.vision_transformer import PatchEmbed, Block
import timm.models.vision_transformer as timm_vit
from timm.models import efficientvit_mit
from einops import rearrange
from typing import Optional, Type
import numpy as np
from Model.encoder import GatingMechanism


class StructuralZGate(nn.Module):
    """Dataset-level, per-dimension gate for decoder-visible structural latents."""

    def __init__(self, z_dim, init=0.95, init_noise=0.02):
        super().__init__()
        init = float(np.clip(init, 1.0e-4, 1.0 - 1.0e-4))
        values = torch.full((int(z_dim),), init, dtype=torch.float32)
        if init_noise > 0:
            values.add_(torch.empty_like(values).uniform_(-init_noise, init_noise))
        values.clamp_(1.0e-4, 1.0 - 1.0e-4)
        self.raw_gate = nn.Parameter(torch.logit(values))
        self.register_buffer("initial_gate", values.clone())
        self.register_buffer("locked_mask", torch.ones_like(values))
        self.register_buffer("mask_locked", torch.tensor(False))
        self.register_buffer("anneal_alpha", torch.tensor(0.0))
        self.register_buffer("gate_active", torch.tensor(True))
        self.register_buffer("prune_order", torch.full((int(z_dim),), -1, dtype=torch.long))
        self.register_buffer("prune_importance", torch.zeros(int(z_dim), dtype=torch.float32))
        self.register_buffer("prune_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("actual_lock_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("transition_start_epoch", torch.tensor(-1.0))
        self.register_buffer("auto_advance_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("auto_advance_active_dim", torch.tensor(-1, dtype=torch.long))
        self.curriculum_mode = "progressive_dimension"

    def gate(self, z=None):
        gate = torch.sigmoid(self.raw_gate)
        return gate if z is None else gate.to(dtype=z.dtype, device=z.device)

    @torch.no_grad()
    def lock_mask(self, threshold=0.5, min_active_dim=4, epoch=None, auto_advanced=False):
        gate = self.gate()
        mask = gate >= float(threshold)
        keep = min(max(int(min_active_dim), 0), gate.numel())
        if int(mask.sum()) < keep:
            mask[torch.topk(gate, keep).indices] = True
        self.locked_mask.copy_(mask.to(self.locked_mask.dtype))
        self.mask_locked.fill_(True)
        inactive = torch.nonzero(~mask, as_tuple=False).flatten()
        inactive = inactive[torch.argsort(gate[inactive], stable=True)]
        count = inactive.numel()
        self.prune_order.fill_(-1)
        self.prune_importance.zero_()
        if count:
            self.prune_order[:count].copy_(inactive)
            self.prune_importance[:count].copy_(gate[inactive])
        self.prune_count.fill_(count)
        if epoch is not None:
            self.actual_lock_epoch.fill_(int(epoch))
            self.transition_start_epoch.fill_(float(epoch))
            if auto_advanced:
                self.auto_advance_epoch.fill_(int(epoch))
                self.auto_advance_active_dim.fill_(int(mask.sum().item()))

    def _progressive_scale(self, gate):
        count = int(self.prune_count.item())
        scale = torch.ones_like(gate)
        if count == 0:
            return scale
        order = self.prune_order[:count].to(device=gate.device)
        importance = self.prune_importance[:count].to(dtype=torch.float64, device=gate.device)
        if not bool(torch.all(torch.isfinite(importance) & (importance > 0)).item()):
            importance = torch.ones_like(importance)
        weights = importance / importance.sum()
        cumulative = torch.cumsum(weights, dim=0)
        alpha = min(max(float(self.anneal_alpha.item()), 0.0), 1.0)
        if alpha >= 1.0:
            scale[order] = 0.0
            return scale
        alpha_tensor = torch.tensor(alpha, dtype=cumulative.dtype, device=cumulative.device)
        fully_pruned = int(torch.searchsorted(cumulative, alpha_tensor, right=True).item())
        if fully_pruned:
            scale[order[:fully_pruned]] = 0.0
        if fully_pruned < count and alpha > 0.0:
            before = 0.0 if fully_pruned == 0 else float(cumulative[fully_pruned - 1].item())
            duration = float(weights[fully_pruned].item())
            fade = min(max((alpha - before) / duration, 0.0), 1.0)
            scale[order[fully_pruned]] = 0.5 * (1.0 + float(np.cos(np.pi * fade)))
        return scale

    def effective_gate(self, z=None):
        gate = self.gate(z)
        if not bool(self.gate_active.item()):
            # Keep raw_gate in the autograd graph during the warm-up bypass.
            # DDP otherwise treats it as an unused parameter and fails on the
            # following iteration.  The zero-valued term preserves the exact
            # all-open behavior while producing a zero (not missing) gradient.
            return torch.ones_like(gate) + 0.0 * gate
        if not bool(self.mask_locked.item()):
            return gate
        if self.curriculum_mode == "progressive_dimension":
            return gate * self._progressive_scale(gate)
        mask = self.locked_mask.to(dtype=gate.dtype, device=gate.device)
        alpha = self.anneal_alpha.to(dtype=gate.dtype, device=gate.device)
        return gate * (mask + (1.0 - alpha) * (1.0 - mask))

    def forward(self, z):
        return z * self.effective_gate(z)

    @property
    def active_dim(self):
        if bool(self.mask_locked.item()):
            return int(self.locked_mask.sum().item())
        return int((self.gate() >= 0.5).sum().item())


class AdaptiveLayerNorm(nn.Module):
    """LayerNorm with zero-initialized structural-z scale and shift modulation."""

    def __init__(self, normalized_shape, cond_dim, eps=1.0e-5):
        super().__init__()
        self.normalized_shape = (int(normalized_shape),)
        self.cond_dim = int(cond_dim)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.modulator = nn.Linear(self.cond_dim, 2 * self.normalized_shape[0], bias=False)
        nn.init.zeros_(self.modulator.weight)
        self.register_buffer("runtime_scale", torch.tensor(1.0))

    def set_runtime_scale(self, scale):
        self.runtime_scale.fill_(float(scale))

    def forward(self, x, condition=None):
        y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if condition is None:
            return y
        if condition.ndim != 2 or condition.shape[-1] != self.cond_dim:
            raise ValueError(f"AdaLN condition must have shape [batch, {self.cond_dim}]")
        if y.shape[0] % condition.shape[0] != 0:
            raise ValueError("AdaLN condition batch cannot be broadcast over decoder queries")
        gamma, beta = self.modulator(condition).chunk(2, dim=-1)
        repeat = y.shape[0] // condition.shape[0]
        y_shape = y.shape
        y = y.reshape(condition.shape[0], repeat, *y_shape[1:])
        gamma = gamma[:, None]
        beta = beta[:, None]
        scale = self.runtime_scale.to(dtype=y.dtype, device=y.device)
        return (y * (1.0 + scale * gamma) + scale * beta).reshape(y_shape)


class VitDecoder(nn.Module):
    def __init__(self, in_features, hidden_dim, resolution, dim_attn=256, vit_layers=1, out_dim=1,
                 decoder_num_heads=8,
                 mlp_ratio=2., sample_size=512, type='vit', ):
        super(VitDecoder, self).__init__()
        self.type = type
        self.resolution = resolution
        self.in_features = in_features

        self.projector_i = nn.Linear(in_features, dim_attn)
        self.projector_pe = nn.Linear(in_features, dim_attn)
        self.attn_block = Block(dim_attn, decoder_num_heads, sample_size=sample_size, resolution=resolution,
                                mlp_ratio=mlp_ratio)

        self.projector_o = ResidualLinearMLPDecoder(hidden_dim=hidden_dim, out_dim=out_dim, n_layers=3,
                                                    add_attn_layer=False,
                                                    in_dim=dim_attn, use_attn_layer=True)

    def forward(self, x_all, kv=None):

        x_pe = x_all[:, :, 0:self.in_features]
        x = x_pe + x_all[:, :, self.in_features:]

        x = self.projector_i(x)

        x_pe = self.projector_pe(x_pe)

        if kv is not None:
            kv = kv[:, :, 0:self.in_features] + kv[:, :, self.in_features:]
            kv = self.projector_i(kv)
        x = self.attn_block(x, kv=kv)
        if self.type.startswith('vit_efficient'):
            x = self.projector_x(x)
            x = self.projector_o(x + x_pe)
        else:
            x = self.projector_o(x + x_pe)
        return x


class EfficientVitBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            stride=1,
            kernel_size=1,
            heads_ratio=1.0,
            head_dim=32,
            expand_ratio=4,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
    ):
        super(EfficientVitBlock, self).__init__()
        self.context_module = efficientvit_mit.ResidualBlock(
            efficientvit_mit.LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=head_dim,
                norm_layer=(None, norm_layer),
            ),
            nn.Identity(),
        )
        self.local_module = efficientvit_mit.ResidualBlock(
            efficientvit_mit.MBConv(
                stride=stride,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm_layer=(None, None, norm_layer),
                act_layer=(act_layer, act_layer, None),
            ),
            nn.Identity(),
        )

    def forward(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x


@torch.jit.script
def dynamic_pad(x: torch.Tensor, window_size: int) -> tuple:
    """
    动态填充输入至 window_size 的整数倍
    返回填充后的张量及填充量 (pad_h, pad_w)
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 填充顺序: (C左, C右, W左, W右, H左, H右)
    return x, (pad_h, pad_w)


# @torch.jit.script
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    窗口划分（兼容填充后的尺寸）
    输入: x (B, H_pad, W_pad, C)
    输出: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = rearrange(x, 'B (H_win h) (W_win w) C -> (B H_win W_win) h w C',
                  h=window_size, w=window_size)
    return x


# @torch.jit.script
def window_reverse(windows: torch.Tensor, window_size: int, H_pad: int, W_pad: int) -> torch.Tensor:
    """
    窗口合并（兼容填充后的尺寸）
    输入: windows (B*num_windows, window_size, window_size, C)
    输出: (B, H_pad, W_pad, C)
    """
    B_win = windows.shape[0]
    B = B_win // ((H_pad // window_size) * (W_pad // window_size))
    x = rearrange(windows, '(B H_win W_win) h w C -> B (H_win h) (W_win w) C',
                  B=B, H_win=H_pad // window_size, W_win=W_pad // window_size,
                  h=window_size, w=window_size)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, use_flash=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_flash = use_flash

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H_orig, W_orig, C = x.shape

        # 动态填充
        x_padded, (pad_h, pad_w) = dynamic_pad(x, self.window_size)
        H_pad, W_pad = H_orig + pad_h, W_orig + pad_w

        # 窗口划分
        x_windows = window_partition(x_padded, self.window_size)  # (B*num_win, ws, ws, C)
        N = self.window_size ** 2
        x_windows = x_windows.view(-1, N, C)  # (B_win*N, C)

        # 生成 QKV
        qkv = self.qkv(x_windows).view(-1, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_win, num_heads, N, head_dim)

        # Flash Attention 或原生实现
        # if self.use_flash and flash_attn_available:
        #     attn_output = flash_attn_qkvpacked_func(qkv, softmax_scale=1.0, causal=False)
        # else:
        #     q, k, v = qkv[0], qkv[1], qkv[2]
        #     attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        #     attn = attn.softmax(dim=-1)
        #     attn_output = attn @ v

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            # 0.,
        )

        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).reshape(-1, self.window_size, self.window_size, C)

        # 窗口合并与裁剪
        x = window_reverse(attn_output, self.window_size, H_pad, W_pad)
        x = x[:, :H_orig, :W_orig, :]  # 移除填充部分

        return self.proj(x)


class MultiHeadSparseSelfAttention(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads, sample_size, resolution=31, dropout=0.0, sample_mode="uniform"):
        """
        embed_dim: 输入的embedding维度
        num_heads: 多头注意力头数
        sample_size: 每个查询采样的键数量
        dropout: dropout概率
        sample_mode: 采样模式，可选 "random"（随机采样）或 "uniform"（固定均匀采样）
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        # 生成均匀采样索引，形状 [sample_size]
        self.resolution = resolution
        self.indices = torch.linspace(0, pow(resolution, 2) - 1, steps=sample_size).long()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sample_size = sample_size
        self.sample_mode = sample_mode

        self.q_proj = nn.Linear(in_dim, embed_dim)
        self.k_proj = nn.Linear(in_dim, embed_dim)
        self.v_proj = nn.Linear(in_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_masked, kv):
        """
        x: 输入 tensor，形状 [batch_size, seq_len, embed_dim]
        """
        # if mask is not None:
        #     q=x[:,mask]
        #     kv=x
        # else:
        #     q=kv=x
        q = x_masked

        batch_size, seq_len, _ = q.size()
        seq_len_kv = kv.size(1)
        # if seq_len != 961:
        #     print("seq_len is not 961")

        # 计算 Q
        q = self.q_proj(q).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q 形状: [batch_size, num_heads, seq_len, head_dim]

        if self.sample_mode == "uniform":

            # 仅计算采样的 K/V，减少计算量
            # if seq_len_kv<pow(self.resolution,2):
            if seq_len < pow(self.resolution, 2):
                if seq_len_kv < self.sample_size:
                    kv_sampled = kv
                    sample_size = seq_len_kv
                else:
                    # self.indices=self.indices[self.indices < seq_len]
                    # kv_sampled = kv[:, self.indices, :]
                    kv_sampled = kv[:, self.indices[self.indices < seq_len], :]

                    sample_size = kv_sampled.shape[1]
            else:
                kv_sampled = kv[:, self.indices, :]
                sample_size = self.sample_size
            # kv_sampled = kv[:, self.indices, :]
            # sample_size = self.sample_size
            k = self.k_proj(kv_sampled).reshape(batch_size, sample_size, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(kv_sampled).reshape(batch_size, sample_size, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            raise ValueError("sample_mode must be 'random' or 'uniform'")

        # 计算注意力，使用 PyTorch 高效实现
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0)

        # 形状恢复 [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class MoEDecoder(nn.Module):
    def __init__(self, in_dim, n_layers, hidden_dim, out_dim, moe_num,
                 top_k=1, num_shared_experts=1, use_clustering_route=True):
        super(MoEDecoder, self).__init__()
        self.num_shared_experts = num_shared_experts

        self.projector_i = self.get_mlp_layers(n_layers=1, dim_in=in_dim, dim_out=hidden_dim, hidden_dim=hidden_dim)

        moe_layers = []
        # 中间层加入带共享专家的MoE
        for _ in range(n_layers):
            moe_layers.append(
                MoELayer(
                    hidden_dim,
                    num_experts=moe_num,
                    top_k=top_k,
                    num_shared_experts=num_shared_experts,
                    use_clustering_route=use_clustering_route
                )
            )
            # moe_layers.append(nl())

        self.moe_layers = nn.Sequential(*moe_layers)

        # 输出层
        self.projector_o = MyLinear(hidden_dim, out_dim)

    def forward(self, x, route_id=None):
        flat = x.view(-1, x.shape[-1])
        route_id = route_id.unsqueeze(-1).repeat(1, flat.shape[0] // route_id.shape[0]).view(-1, 1)
        # ret_flat = self.main(flat)
        ret_flat = self.projector_i(flat)
        for moe_layer in self.moe_layers:
            ret_flat = moe_layer(ret_flat, route_id=route_id)
        ret_flat = self.projector_o(ret_flat)
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1])
        return ret

    def get_mlp_layers(self, n_layers, dim_in, dim_out, hidden_dim, nl=nn.ReLU):
        expert_layers = []
        for i in range(n_layers):
            if i == 0:
                expert_layers.append(
                    ResidualLinear(dim_in, hidden_dim) if dim_in == hidden_dim else nn.Linear(dim_in, hidden_dim))
            elif i == n_layers - 1:
                expert_layers.append(ResidualLinear(hidden_dim, dim_out))
            else:
                expert_layers.append(ResidualLinear(hidden_dim, hidden_dim))
            expert_layers.append(nl())
        return nn.Sequential(*expert_layers)


class ResidualLinearMLPDecoder(nn.Module):
    def __init__(self, in_dim, n_layers, hidden_dim, out_dim, add_attn_layer=False,
                 use_attn_layer=False, nl=nn.ReLU, num_experts=0, top_k=1, num_shared_experts=1,
                 feature_fuse_indices=None, feature_fuse_dim=384, layers_num_block=1, MLP_ln=True,
                 use_adaln=False, adaln_cond_dim=None, adaln_condition_norm=True):
        super(ResidualLinearMLPDecoder, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.feature_fuse_indices = feature_fuse_indices
        self.use_adaln = bool(use_adaln)
        self.adaln_cond_dim = adaln_cond_dim
        self.adaln_condition_norm = (nn.LayerNorm(adaln_cond_dim, elementwise_affine=False)
                                     if self.use_adaln and adaln_condition_norm else nn.Identity())
        if use_attn_layer:
            if in_dim != hidden_dim and not add_attn_layer:
                layers = [nn.Linear(in_dim, hidden_dim), nl()]
            else:
                layers = []
        else:
            layers = [
                ResidualLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim),
                self._make_norm(hidden_dim) if MLP_ln else nn.Identity(),
                nl()
            ]
        # gate_layers=[]
        # 中间层加入带共享专家的MoE
        for i in range(n_layers):
            if num_experts > 0:
                layers.append(
                    MoELayer(
                        hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        num_shared_experts=num_shared_experts
                    )
                )
            else:
                layers.append(ResidualLinear(hidden_dim, hidden_dim, fuse_intermediate_features=(
                            feature_fuse_indices is not None and i in feature_fuse_indices),
                                             intermediate_features_dim=feature_fuse_dim, layer_norm_before=True,
                                             MLP_ln=MLP_ln, layer_num=layers_num_block, hidden_dim=hidden_dim//2,
                                             use_adaln=self.use_adaln, adaln_cond_dim=self.adaln_cond_dim))

            # layers.append(nn.LayerNorm(hidden_dim) if MLP_ln else nn.Identity())
            layers.append(nl())
            # if feature_fuse_indices is not None and i in feature_fuse_indices:
            #     gate_layers.append(GatingMechanism(hidden_dim, hidden_dim))

        # 输出层
        layers.append(
            ResidualLinear(hidden_dim, out_dim) if out_dim == hidden_dim
            else MyLinear(hidden_dim, out_dim)
        )

        self.main_layers = nn.Sequential(*layers)
        # self.gate_layers = nn.ModuleList(gate_layers) if len(gate_layers) > 0 else None

    def _make_norm(self, dim):
        return AdaptiveLayerNorm(dim, self.adaln_cond_dim) if self.use_adaln else nn.LayerNorm(dim)

    def set_adaln_scale(self, scale):
        for module in self.modules():
            if isinstance(module, AdaptiveLayerNorm):
                module.set_runtime_scale(scale)

    @staticmethod
    def _apply_layer(layer, x, condition=None, intermediate_features=None):
        if isinstance(layer, AdaptiveLayerNorm):
            return layer(x, condition)
        if isinstance(layer, ResidualLinear):
            return layer(x, intermediate_features=intermediate_features, adaln_condition=condition)
        return layer(x)

    def forward(self, x, intermediate_features=None, adaln_condition=None, **args):
        flat = x.view(-1, x.shape[-1])
        ii = -1
        condition = self.adaln_condition_norm(adaln_condition) if adaln_condition is not None else None
        if self.feature_fuse_indices is not None and intermediate_features is not None:
            for i in range(len(self.main_layers)):
                # if i / 2 - 1 in self.feature_fuse_indices:
                if i / 3 - 1 in self.feature_fuse_indices:
                    intermediate_features_flat = intermediate_features[ii].contiguous().view(-1, intermediate_features[
                        ii].shape[-1])
                    flat = self._apply_layer(self.main_layers[i], flat, condition, intermediate_features_flat)
                    ii -= 1
                else:
                    flat = self._apply_layer(self.main_layers[i], flat, condition)
            ret_flat = flat
        else:
            for layer in self.main_layers:
                flat = self._apply_layer(layer, flat, condition)
            ret_flat = flat
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1])
        return ret


class MoELayer(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=1, num_shared_experts=1, use_clustering_route=False, nl=nn.ReLU):
        super(MoELayer, self).__init__()

        # self.num_experts = num_experts + num_shared_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_experts
        self.top_k = min(top_k, self.num_routed_experts)
        self.use_clustering_route = use_clustering_route

        # 专家定义（共享 + 路由）
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                nn.Linear(input_dim, input_dim) for _ in range(num_shared_experts)
            ])
        self.routed_experts = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(self.num_routed_experts)
        ])

        if not use_clustering_route:
            # 路由门控
            self.gate = nn.Linear(input_dim, self.num_routed_experts)
            self.gate_bias = nn.Parameter(torch.zeros(self.num_routed_experts))
        self.nl = nl()
        # self.gate_bias = nn.Parameter(torch.randn(self.num_routed_experts) * 0.2)

        # 新增：专家选择统计
        # self.register_buffer("expert_counts", torch.zeros(self.num_routed_experts, dtype=torch.long))
        # self.register_buffer("total_samples", torch.tensor(0, dtype=torch.long))

    def forward(self, x, route_id=None):
        # 共享专家处理
        if self.num_shared_experts > 0:
            shared_outputs = [expert(x) for expert in self.shared_experts]
            shared_combined = sum(shared_outputs) / self.num_shared_experts
        else:
            shared_combined = 0
        if not self.use_clustering_route:
            # 路由专家处理
            gate_scores = self.gate(x) + self.gate_bias
            topk_values, topk_indices = torch.topk(gate_scores, k=self.top_k, dim=-1)
            topk_weights = torch.softmax(topk_values, dim=-1)

            # 更新专家选择统计（新增）
            # if self.training:  # 仅在训练时统计
            #     with torch.no_grad():
            #         flat_indices = topk_indices.view(-1)  # 展平所有样本的选择
            #         for idx in flat_indices:
            #             self.expert_counts[idx] += 1
            #         self.total_samples += x.size(0)  # 累计样本数

            # 计算路由输出
            gate_weights = torch.zeros_like(gate_scores, dtype=topk_weights.dtype)
            gate_weights.scatter_(-1, topk_indices, topk_weights)
            routed_outputs = torch.stack([expert(x) for expert in self.routed_experts], dim=-2)
            routed_combined = torch.einsum('...e,...ed->...d', gate_weights, routed_outputs)

        else:
            if self.training:
                gate_weights = torch.zeros([x.shape[0], self.num_routed_experts], dtype=x.dtype).to(x.device)
                gate_weights.scatter_(-1, route_id, 1.0)
                routed_outputs = torch.stack([expert(x) for expert in self.routed_experts], dim=-2)
                routed_combined = torch.einsum('...e,...ed->...d', gate_weights, routed_outputs)
            else:
                routed_combined = self.sparse_route_forward(x, route_id)
            # routed_combined = self.sparse_route_forward(x, route_id)

        return self.nl(shared_combined + routed_combined + x)

    # def sparse_route_forward(self, x, topk_indices):
    #     # 获取top-k专家索引和权重
    #     # topk_values, topk_indices = torch.topk(gate_scores, k=self.top_k, dim=-1)
    #     # topk_weights = torch.softmax(topk_values, dim=-1)
    #
    #     # 展平处理
    #     flat_indices = topk_indices.view(-1)
    #     # flat_weights = topk_weights.view(-1)
    #     # batch_size = x.size(0)
    #
    #     # 找出所有被选中的唯一专家
    #     unique_experts, inverse_indices = torch.unique(flat_indices, return_inverse=True)
    #
    #     # 准备累加的输出张量
    #     routed_combined = torch.zeros_like(x)
    #
    #     # 对每个被选中的专家进行批量计算
    #     for idx, expert_id in enumerate(unique_experts):
    #         # 获取需要该专家的样本掩码
    #         mask = (flat_indices == expert_id)
    #         if not mask.any():
    #             continue
    #
    #         # 获取对应样本的原始索引
    #         sample_indices = torch.div(torch.nonzero(mask).squeeze(-1), self.top_k, rounding_mode='floor')
    #
    #         # 收集对应样本的输入和权重
    #         selected_x = x[sample_indices]
    #         # selected_weights = flat_weights[mask]
    #
    #         # 专家计算
    #         expert_out = self.routed_experts[expert_id](selected_x)
    #
    #         # 加权并累加到输出
    #         # routed_combined.index_add_(0, sample_indices, expert_out * selected_weights.unsqueeze(-1))
    #         routed_combined.index_add_(0, sample_indices, expert_out)
    #
    #     return routed_combined

    def sparse_route_forward(self, x, topk_indices, topk_weights=None):
        flat_indices = topk_indices.view(-1)
        unique_experts = torch.unique(flat_indices)
        routed_combined = torch.zeros_like(x)

        # 支持权重（若未提供则默认为均匀权重）
        if topk_weights is None:
            topk_weights = torch.ones_like(topk_indices, dtype=x.dtype) / self.top_k
        flat_weights = topk_weights.view(-1)

        for expert_id in unique_experts:
            mask = (flat_indices == expert_id)
            if not mask.any():
                continue

            # 批量收集输入和权重
            sample_indices = torch.div(torch.nonzero(mask).squeeze(-1), self.top_k, rounding_mode='floor')
            selected_x = x[sample_indices]
            selected_weights = flat_weights[mask]

            # 专家计算 + 加权累加
            expert_out = self.routed_experts[expert_id](selected_x)
            routed_combined.index_add_(0, sample_indices, expert_out * selected_weights.unsqueeze(-1))

        return routed_combined

    # def get_expert_freq(self, reset=False):
    #     """返回路由专家的选择频率，并可选是否重置统计"""
    #     freq = self.expert_counts / max(1, self.total_samples * self.top_k)
    #     if reset:
    #         self.expert_counts.zero_()
    #         self.total_samples.zero_()
    #     return freq.cpu().numpy()  # 残差连接


class ResidualLinear(nn.Module):
    def __init__(self, n_in, n_out, fuse_intermediate_features=False, intermediate_features_dim=None,
                 layer_norm_before=False, MLP_ln=True, layer_num=2, hidden_dim=128,
                 use_adaln=False, adaln_cond_dim=None):
        super(ResidualLinear, self).__init__()
        if layer_num == 1:
            hidden_dim= n_out
        layers=[]
        for i in range(layer_num):
            if i == layer_num - 1:
                if MLP_ln:
                    layers += [nn.Linear(hidden_dim, n_out), AdaptiveLayerNorm(n_out, adaln_cond_dim)
                               if use_adaln else nn.LayerNorm(n_out)]
                else:
                    layers += [nn.Linear(hidden_dim, n_out)]


            elif i == 0:
                if MLP_ln:
                    layers += [nn.Linear(n_in, hidden_dim), AdaptiveLayerNorm(hidden_dim, adaln_cond_dim)
                               if use_adaln else nn.LayerNorm(hidden_dim), nn.ReLU()]
                else:
                    layers += [nn.Linear(n_in, hidden_dim), nn.ReLU()]

            else:
                if MLP_ln:
                    layers += [nn.Linear(hidden_dim, hidden_dim), AdaptiveLayerNorm(hidden_dim, adaln_cond_dim)
                               if use_adaln else nn.LayerNorm(hidden_dim), nn.ReLU()]
                else:
                    layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.layers = nn.Sequential(*layers)
        # self.linear = nn.Linear(n_in, n_out)
        self.fuse_intermediate_features = fuse_intermediate_features
        if self.fuse_intermediate_features:
            self.gate_layer = GatingMechanism(n_out, intermediate_features_dim)
            if layer_norm_before:
                self.norm_layer = nn.LayerNorm(intermediate_features_dim)
            else:
                self.norm_layer = nn.Identity()

    def forward(self, x, intermediate_features=None, adaln_condition=None):
        feature = x
        for layer in self.layers:
            feature = layer(feature, adaln_condition) if isinstance(layer, AdaptiveLayerNorm) else layer(feature)
        if self.fuse_intermediate_features and intermediate_features is not None:
            feature = self.gate_layer(feature, self.norm_layer(intermediate_features))
        return feature + x


class MyLinear(nn.Linear):
    def forward(self, x):
        if x.dtype == torch.half:
            return F.linear(x, self.weight.half(), self.bias.half())
        else:
            return F.linear(x, self.weight, self.bias)


def half_linear(x, weight, bias):
    return F.linear(x, weight.half(), bias.half())


def single_linear(x, weight, bias):
    return F.linear(x, weight, bias)


def trim_tensor(x, d, e):
    a, b, c = x.shape
    assert d < b and e < d, "Invalid d or e"
    assert b % d == 0, "b must be divisible by d"

    # 1. 分组：(a, b, c) -> (a, num_groups, d, c)
    num_groups = b // d
    x_grouped = x.view(a, num_groups, d, c)

    # 2. 裁剪每组的最后 e 个向量：(a, num_groups, d, c) -> (a, num_groups, d - e, c)
    x_trimmed_groups = x_grouped[:, :, :d - e, :]

    # 3. 裁剪最后的 e 个组：(a, num_groups, d - e, c) -> (a, num_groups - e, d - e, c)
    x_trimmed_groups = x_trimmed_groups[:, :num_groups - e, :, :]

    # 4. 合并：(a, num_groups - e, d - e, c) -> (a, (num_groups - e) * (d - e), c)
    new_b = (num_groups - e) * (d - e)
    x_trimmed = x_trimmed_groups.reshape(a, new_b, c)

    return x_trimmed


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = timm_vit.Mlp,
            sample_size: int = 96,
            resolution: int = 31,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_norm=qk_norm,
        #     proj_bias=proj_bias,
        #     attn_drop=attn_drop,
        #     proj_drop=proj_drop,
        #     norm_layer=norm_layer,
        # )
        self.attn = MultiHeadSparseSelfAttention(in_dim=dim, embed_dim=dim, num_heads=num_heads,
                                                 sample_size=sample_size, resolution=resolution)

        self.ls1 = timm_vit.LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = timm_vit.DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = timm_vit.LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = timm_vit.DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, kv=None) -> torch.Tensor:
        x = self.norm1(x)
        if kv is not None:
            kv = self.norm1(kv)
        else:
            kv = x
        x = x + self.drop_path1(self.ls1(self.attn(x, kv)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class SharedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, nl=nn.ReLU):
        """
        in_dim: int
        out_dim: int
        depth: int
        hidden_dim: int
        """
        super(SharedLinear, self).__init__()

        linear = []
        for dim in hidden_dim:
            linear.append(nn.Linear(in_dim, dim))
            in_dim = dim
            linear.append(nn.SyncBatchNorm(dim))
            linear.append(nl())
        linear.append(nn.Linear(in_dim, out_dim))
        linear.append(nn.Tanh())
        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        """
        x: [batch_size, in_dim]

        output: [batch_size, out_dim]
        """
        return self.linear(x)


class HyperVolume(nn.Module):
    def __init__(self, resolution, z_dim, n_layers, hidden_dim, pe_type, pe_dim, feat_sigma, domain, pe_type_conf=None,
                 decoder_type='mlp', moe_num=6, num_shared_experts=1, use_clustering_route=True,
                 feature_fuse_indices=None, decoder_ln=True,
                 structural_z_gate_enabled=True, structural_z_gate_init=0.95,
                 structural_z_gate_init_noise=0.02, decoder_adaln_enabled=True,
                 decoder_adaln_condition_norm=True, decoder_adaln_scale=0.5,
                 structural_z_hard_gate_curriculum_mode='progressive_dimension'):
        """
        resolution: int
        z_dim: int
        n_layers: int
        hidden_dim: int
        pe_type: str
        pe_dim: int
        feat_sigma: float
        domain: str
        """
        super(HyperVolume, self).__init__()
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        self.decoder_ln = decoder_ln
        self.structural_z_gate_enabled = bool(structural_z_gate_enabled and z_dim > 0)
        self.structural_z_gate = (StructuralZGate(z_dim, structural_z_gate_init, structural_z_gate_init_noise)
                                  if self.structural_z_gate_enabled else None)
        self.structural_z_hard_gate_curriculum_mode = structural_z_hard_gate_curriculum_mode
        if self.structural_z_gate is not None:
            self.structural_z_gate.curriculum_mode = structural_z_hard_gate_curriculum_mode
        self.decoder_adaln_enabled = bool(decoder_adaln_enabled and z_dim > 0)
        self.decoder_adaln_condition_norm = bool(decoder_adaln_condition_norm)
        self.decoder_adaln_scale = float(decoder_adaln_scale)
        if pe_type == 'gaussian':
            rand_freqs = torch.randn((3 * pe_dim, 3), dtype=torch.float) * feat_sigma
            self.rand_freqs = nn.Parameter(rand_freqs, requires_grad=False)
            x_pe_dim = 3 * 2 * pe_dim
        else:
            raise NotImplementedError
        self.pe_type_conf = pe_type_conf
        if pe_type_conf is None:
            z_pe_dim = z_dim
        elif pe_type_conf == 'geom':
            min_freq = -4
            n_freqs = 4
            geom_freqs_conf = 2.0 ** torch.arange(min_freq, min_freq + n_freqs, dtype=torch.float) * np.pi
            self.geom_freqs_conf = nn.Parameter(geom_freqs_conf, requires_grad=False)
            z_pe_dim = z_dim * 2 * n_freqs
        else:
            raise NotImplementedError

        self.D = resolution
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.feat_sigma = feat_sigma
        self.domain = domain
        self.decoder_type = decoder_type
        self.moe_num = moe_num
        self.num_shared_experts = num_shared_experts

        in_features = x_pe_dim + z_pe_dim
        if domain == 'hartley':
            # self.mlp = ResidualLinearMLP(in_features, n_layers, hidden_dim, 1)
            if decoder_type == 'moe':
                self.decoder = MoEDecoder(in_features, n_layers, hidden_dim, 1, moe_num=moe_num,
                                          num_shared_experts=num_shared_experts,
                                          use_clustering_route=use_clustering_route)
            else:
                self.decoder = ResidualLinearMLPDecoder(in_features, n_layers, hidden_dim, 1,
                                                        feature_fuse_indices=feature_fuse_indices,
                                                        feature_fuse_dim=x_pe_dim, MLP_ln=self.decoder_ln,
                                                        use_adaln=self.decoder_adaln_enabled,
                                                        adaln_cond_dim=z_dim,
                                                        adaln_condition_norm=self.decoder_adaln_condition_norm)
                self.decoder.set_adaln_scale(self.decoder_adaln_scale)
        else:
            raise NotImplementedError

    def forward(self, x, z, route_labels=None, intermediate_features=None):
        """
        x: [batch_size(, n_tilts), n_pts, 3]
        z: [batch_size, z_dim] or None

        output: [batch_size(, n_tilts), n_pts]
        """
        batch_size_in = x.shape[0]
        n_pts = x.shape[-2]
        subtomogram_averaging = x.dim() == 4
        if self.pe_type == 'gaussian':
            x = self.random_fourier_encoding(x)
        decoder_z = self.apply_structural_z(z)
        adaln_condition = decoder_z
        if decoder_z is not None:
            if self.pe_type_conf == 'geom':
                decoder_z = self.geom_fourier_encoding_conf(decoder_z)
            if subtomogram_averaging:
                n_tilts = x.shape[1]
                z_expand = decoder_z[:, None, None].expand(-1, n_tilts, n_pts, -1)
            else:
                z_expand = decoder_z[:, None].expand(-1, n_pts, -1)
            x = torch.cat([x, z_expand], -1)
        if intermediate_features is not None:
            if self.pe_type_conf == 'geom':
                intermediate_features = [self.geom_fourier_encoding_conf(z) for z in intermediate_features]
            if subtomogram_averaging:
                n_tilts = x.shape[1]
                intermediate_features = [z[:, None, None].expand(-1, n_tilts, n_pts, -1) for z in intermediate_features]
            else:
                intermediate_features = [z[:, None].expand(-1, n_pts, -1) for z in intermediate_features]

        if subtomogram_averaging:
            n_tilts = x.shape[1]
            out_shape = (batch_size_in, n_tilts, n_pts)
        else:
            out_shape = (batch_size_in, n_pts)

        if self.decoder_type == 'moe':
            y_pred = self.decoder(x, route_id=route_labels)
        else:
            y_pred = self.decoder(x, route_id=route_labels, intermediate_features=intermediate_features,
                                  adaln_condition=adaln_condition)
        return y_pred.reshape(*out_shape)

    def apply_structural_z(self, z):
        if z is None or self.structural_z_gate is None:
            return z
        return self.structural_z_gate(z)

    def get_structural_z_gate(self):
        return None if self.structural_z_gate is None else self.structural_z_gate.gate()

    def configure_structural_z(self, epoch, warmup_epochs=5, hard_start_epoch=50,
                               hard_anneal_epochs=5, threshold=0.5, min_active_dim=4,
                               curriculum_mode='progressive_dimension', auto_advance_enabled=True):
        if self.structural_z_gate is None:
            return
        gate = self.structural_z_gate
        gate.curriculum_mode = curriculum_mode
        self.structural_z_hard_gate_curriculum_mode = curriculum_mode
        gate.gate_active.fill_(int(epoch) >= int(warmup_epochs))
        auto_advance = (auto_advance_enabled and bool(gate.gate_active.item())
                        and int(epoch) < int(hard_start_epoch)
                        and gate.active_dim <= int(min_active_dim))
        if ((int(epoch) >= int(hard_start_epoch) or auto_advance)
                and not bool(gate.mask_locked.item())):
            gate.lock_mask(threshold, min_active_dim, epoch=epoch,
                           auto_advanced=auto_advance)
        if bool(gate.mask_locked.item()):
            lock_epoch = int(gate.actual_lock_epoch.item())
            if lock_epoch < 0:
                lock_epoch = int(hard_start_epoch)
            alpha = 1.0 if hard_anneal_epochs <= 0 else min(
                max((float(epoch) - lock_epoch) / hard_anneal_epochs, 0.0), 1.0)
            gate.anneal_alpha.fill_(alpha)
        return auto_advance

    def update_structural_z_anneal(self, progress, hard_anneal_epochs=5):
        """Advance a locked structural-z curriculum at batch-level resolution."""
        gate = self.structural_z_gate
        if gate is None or not bool(gate.mask_locked.item()):
            return
        lock_epoch = int(gate.actual_lock_epoch.item())
        if lock_epoch < 0:
            return
        alpha = 1.0 if hard_anneal_epochs <= 0 else min(
            max((float(progress) - lock_epoch) / hard_anneal_epochs, 0.0), 1.0)
        gate.anneal_alpha.fill_(alpha)

    def random_fourier_encoding(self, x):
        """
        x: [batch_size(, n_tilts), n_pts, 3]

        output: [batch_size(, n_tilts), n_pts, 3 * 2 * pe_dim]
        """
        freqs = self.rand_freqs.reshape(1, 1, -1, 3) * (self.D // 2)
        kx_ky_kz = x[..., None, :] * freqs
        k = kx_ky_kz.sum(-1)
        s = torch.sin(k)
        c = torch.cos(k)
        x_encoded = torch.cat([s, c], -1)
        return x_encoded

    def geom_fourier_encoding_conf(self, z):
        """
        z: [batch_size, z_dim]

        output: [batch_size, z_dim * 2 * pe_dim]
        """
        in_dims = z.shape[:-1]
        s = torch.sin(z[..., None] * self.geom_freqs_conf)  # [..., z_dim, pe_dim]
        c = torch.cos(z[..., None] * self.geom_freqs_conf)  # [..., z_dim, pe_dim]
        z_encoded = torch.cat([s, c], -1).reshape(*in_dims, -1)
        return z_encoded

    def get_building_params(self):
        building_params = {
            'resolution': self.D,
            'z_dim': self.z_dim,
            'n_layers': self.n_layers,
            'hidden_dim': self.hidden_dim,
            'pe_type': self.pe_type,
            'pe_dim': self.pe_dim,
            'feat_sigma': self.feat_sigma,
            'domain': self.domain,
            'pe_type_conf': self.pe_type_conf,
            'decoder_type': self.decoder_type,
            'moe_num': self.moe_num,
            'num_shared_experts': self.num_shared_experts,
            'feature_fuse_indices': self.decoder.feature_fuse_indices if hasattr(self.decoder,
                                                                                 'feature_fuse_indices') else None,
            'decoder_ln': self.decoder_ln,
            'structural_z_gate_enabled': self.structural_z_gate_enabled,
            'decoder_adaln_enabled': self.decoder_adaln_enabled,
            'decoder_adaln_condition_norm': self.decoder_adaln_condition_norm,
            'decoder_adaln_scale': self.decoder_adaln_scale,
            'structural_z_hard_gate_curriculum_mode': self.structural_z_hard_gate_curriculum_mode,
        }
        return building_params


class VolumeExplicit(nn.Module):
    def __init__(self, resolution, domain, extent):
        """
        resolution: int
        domain: str
        extent: float
        """
        super(VolumeExplicit, self).__init__()
        assert domain == 'hartley'
        self.D = resolution
        self.domain = domain
        self.extent = extent

        self.volume = nn.Parameter(1e-5 * torch.tensor(
            np.random.randn(resolution, resolution, resolution)
        ).float(), requires_grad=True)

    def forward(self, x, z):
        """
        x: [batch_size, n_pts, 3] in [-extent, extent]
        z: None

        output: [batch_size, n_pts]
        """
        assert z is None, "Explicit volume(s) do not support heterogeneous reconstruction."
        batch_size_in = x.shape[0]
        out = torch.nn.functional.grid_sample(
            1e2 * self.volume[None, None].repeat(batch_size_in, 1, 1, 1, 1),
            x[:, None, None, :, :] / (2. * self.extent) * 2,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return out.reshape(batch_size_in, -1)

    def get_building_params(self):
        building_params = {
            "resolution": self.D,
            "domain": self.domain,
            "extent": self.extent,
        }

        return building_params


class GaussianPyramid(nn.Module):
    def __init__(self, n_layers):
        """
        n_layers: int
        """
        super(GaussianPyramid, self).__init__()
        kernel_size = 2 * n_layers - 1

        # kernels: [n_layers, 1, kernel_size, kernel_size]
        kernels = torch.zeros((n_layers, 1, kernel_size, kernel_size)).float()
        for k in range(n_layers):
            coords = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
            xx, yy = torch.meshgrid(coords, coords)
            r = xx ** 2 + yy ** 2
            kernels[k, 0, r < (k + 1) ** 2] = 1.
            kernels[k, 0] /= torch.sum(kernels[k, 0])

        self.gaussian_pyramid = torch.nn.Conv2d(1, n_layers, kernel_size=kernel_size, padding='same',
                                                padding_mode='reflect', bias=False)
        self.gaussian_pyramid.weight = torch.nn.Parameter(kernels)
        self.gaussian_pyramid.weight.requires_grad = False

    def forward(self, x):
        """
        x: [batch_size, 1, D, D]

        output: [batch_size, n_layers, D, D]
        """
        return self.gaussian_pyramid(x)
