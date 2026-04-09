# mvit_classification_only.py
# MViT(MeMViT) encoder + classification head only (no detection path).

import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .attention import MultiScaleBlock
from .stem_helper import PatchEmbed
from .utils import round_width, validate_checkpoint_wrapper_import

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


class TransformerBasicHead(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        frame_level=False,
    ):
        super().__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        if isinstance(num_classes, (list, tuple)):
            self.projection = nn.ModuleList(
                [nn.Linear(dim_in, c, bias=True) for c in num_classes]
            )
            self.multi_classification = True
        else:
            self.projection = nn.Linear(dim_in, num_classes, bias=True)
            self.multi_classification = False

        self.frame_level = frame_level
        if act_func == "softmax":
            self.act = nn.Softmax(dim=2 if frame_level else 1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{act_func} is not supported.")

    def forward(self, x):
        if self.frame_level:
            # x: [B, T, H, W, C] -> [B, T, C]
            x = x.mean(dim=[2, 3])

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        if self.multi_classification:
            out = []
            for proj in self.projection:
                z = proj(x)
                if not self.training:
                    z = self.act(z)
                out.append(z)
            return out

        x = self.projection(x)
        if not self.training:
            x = self.act(x)
        return x


class MemViT(nn.Module):
    """
    Multiscale Vision Transformers (classification-only).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.DETECTION.ENABLE:
            raise NotImplementedError(
                "This file is classification-only. Set DETECTION.ENABLE=False."
            )

        self.use_online_memory = cfg.MEMVIT.ENABLE
        pool_first = cfg.MVIT.POOL_FIRST

        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]

        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride

        num_classes = (
            cfg.MODEL.NUM_CLASSES_LIST
            if cfg.MODEL.NUM_CLASSES_LIST
            else cfg.MODEL.NUM_CLASSES
        )

        embed_dim = cfg.MVIT.EMBED_DIM
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE

        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        self.conv_q = cfg.MVIT.CONV_Q

        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only layernorm is supported.")

        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)

        self.input_dims = [temporal_size, spatial_size, spatial_size]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim)
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for _ in range(depth)]
        pool_kv = [[] for _ in range(depth)]
        stride_q = [[] for _ in range(depth)]
        stride_kv = [[] for _ in range(depth)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            idx = cfg.MVIT.POOL_Q_STRIDE[i][0]
            stride_q[idx] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL:
                pool_q[idx] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[idx] = [s + 1 if s > 1 else s for s in stride_q[idx]]

        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(depth):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            idx = cfg.MVIT.POOL_KV_STRIDE[i][0]
            stride_kv[idx] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL:
                pool_kv[idx] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[idx] = [s + 1 if s > 1 else s for s in stride_kv[idx]]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None
        self.blocks = nn.ModuleList()

        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        input_size = self.patch_dims
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])

            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i],
                kernel_kv=pool_kv[i],
                stride_q=stride_q[i],
                stride_kv=stride_kv[i],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_temporal=self.rel_pos_temporal,
                use_online_memory=self.use_online_memory,
                attn_max_len=cfg.MEMVIT.ATTN_MAX_LEN,
                keep_max_len=(
                    (cfg.MEMVIT.ATTN_MAX_LEN - 1) * int(cfg.MEMVIT.SAMPLER[-1]) + 1
                    if "gap" in cfg.MEMVIT.SAMPLER
                    else cfg.MEMVIT.ATTN_MAX_LEN
                ),
                causal=cfg.MVIT.CAUSAL,
                online_compress=cfg.MEMVIT.COMPRESS.ENABLE,
                compress_kernel=cfg.MEMVIT.COMPRESS.POOL_KERNEL,
                compress_stride=cfg.MEMVIT.COMPRESS.POOL_STRIDE,
                drop_attn_rate=cfg.MVIT.DROP_ATTN_RATE,
                cfg=cfg,
                conv_q=self.conv_q,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

            if len(stride_q[i]) > 0:
                input_size = [s // st for s, st in zip(input_size, stride_q[i])]
            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)
        self.head = TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            frame_level=cfg.MVIT.FRAME_LEVEL,
        )

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names += ["pos_embed_spatial", "pos_embed_temporal", "pos_embed_class"]
                else:
                    names.append(["pos_embed"])
            if self.rel_pos_spatial:
                names += ["rel_pos_h", "rel_pos_w"]
            if self.rel_pos_temporal:
                names += ["rel_pos_t"]
            if self.cls_embed_on:
                names.append("cls_token")
        return names

    def forward(self, x, video_names=None):
        x = x[0]
        H = x.shape[3] // self.patch_stride[1]

        x = self.patch_embed(x)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        W = x.shape[1] // H // T
        B, _, _ = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(1, self.patch_dims[0], 1)
                pos_embed = pos_embed + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], dim=1)
                x = x + pos_embed
            else:
                x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)
        if self.norm_stem:
            x = self.norm_stem(x)

        mem_selections = self.sample_memory() if self.use_online_memory else None
        thw = [T, H, W]
        for blk_idx, blk in enumerate(self.blocks):
            cur_selection = [] if blk_idx in self.cfg.MEMVIT.EXCLUDE_LAYERS else mem_selections
            x, thw = blk(x, thw, cur_selection, video_names)

        x = self.norm(x)

        if self.cfg.MVIT.FRAME_LEVEL:
            x = x[:, (1 if self.cls_embed_on else 0) :].reshape(
                [x.shape[0]] + thw + [x.shape[-1]]
            )
        else:
            x = x[:, 0] if self.cls_embed_on else x.mean(1)

        return self.head(x)

    def clear_memory(self):
        for block in self.blocks:
            block.attn.cached_x = []
            block.attn.cached_k = []
            block.attn.cached_v = []
            block.attn.cached_video_names = []

    def sample_memory(self):
        cur_len = len(self.blocks[0].attn.cached_k)
        if cur_len == 0:
            return []

        if "gap" in self.cfg.MEMVIT.SAMPLER:
            gap_size = int(self.cfg.MEMVIT.SAMPLER[-1])
            if cur_len < gap_size:
                return []
            num_used = cur_len // gap_size
            return list(range(cur_len)[-gap_size * num_used :: gap_size])

        if self.cfg.MEMVIT.SAMPLER == "all" or not self.training:
            return range(cur_len)

        raise NotImplementedError
