import sys, math
import torch.nn as nn

from pointcept.models.modules import PointModule
from pointcept.models.utils.structure import Point
from addict import Dict
import torch_scatter
import torch
import torch.nn.functional as F


from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.point_transformer_v3 import SerializedPooling, Embedding, SerializedUnpooling, Block


class SafeBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d that avoids training-time crashes when effective batch size is 1.
    Falls back to running-stat normalization in that case.
    """

    def forward(self, input):
        # Effective sample count for BN1d: N (2D) or N*L (3D).
        eff_n = input.shape[0]
        if input.dim() == 3:
            eff_n = eff_n * input.shape[2]
        if self.training and eff_n <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


class PointSequential_intermediate_output(PointSequential):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        intermediate_output = {}
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
            intermediate_output[k] = {'feat':input.feat,'code':input.serialized_code[0],'pooling_inverse':input['pooling_inverse']} #We take the first order (z-order)
        return input, intermediate_output
    
class PointTransformerV3Model(nn.Module):
    def __init__(self, in_channels, enable_flash, 
                enc_dim, output_dim,
                turn_off_bn, stride,
                embedding_type='MLP',
                enc_depths=(2, 2, 2, 6, 2),
                enc_num_head=(2, 4, 8, 16, 32),
                dec_depths=(2, 2, 2, 2),
                dec_num_head=(4, 4, 8, 16),
                dec_channels=None,
                enc_channels=None,
                pdnorm_bn=False,
                pdnorm_ln=False,
                pretrained_ckpt=None, 
                **kwargs,
                ):
        super(PointTransformerV3Model, self).__init__()
        if dec_channels is None:
            if output_dim==64:
                self.dec_channels = (64, 64, 128, 256)
            elif output_dim==128:
                self.dec_channels = (128, 128, 256, 256)
            elif output_dim==96:
                self.dec_channels = (96, 96, 128, 256)
            else:
                raise ValueError("Unsupported output_dim")
        else:
            self.dec_channels = dec_channels

        if enc_channels is None:
            if enc_dim==32:
                enc_channels = (32, 64, 128, 256, 512)
            elif enc_dim==64:
                enc_channels = (64, 96, 128, 256, 512)
            elif enc_dim==128:
                enc_channels = (128, 128, 256, 256, 512)
            else:
                raise ValueError("Unsupported enc_dim")
        else:
            enc_channels = enc_channels
        if enable_flash:
            enc_patch_size = (1024, 1024, 1024, 1024, 1024)[:len(enc_channels)]
            dec_patch_size=(1024, 1024, 1024, 1024)[:len(self.dec_channels)]
        else:
            enc_patch_size = (128, 128, 128, 128, 128)[:len(enc_channels)]
            dec_patch_size=(128, 128, 128, 128)[:len(self.dec_channels)]
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            embedding_type=embedding_type,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=self.dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=enable_flash, #disable flash attention first
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False,
            pdnorm_bn=pdnorm_bn,
            turn_off_bn = turn_off_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=True,
            pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"), #This does not matter as pdnorm_bn/ln=False
        )
        self.output_dim = self.dec_channels[0]

        if pretrained_ckpt is not None:
            sd = torch.load(pretrained_ckpt,map_location='cpu')['state_dict']
            sd = {k.replace('module.backbone.',''):v for k,v in sd.items() if 'backbone.' in k} # we only want the backbone
            #If shape does not match, we do not load the model
            load_sd = {}
            for k, v in self.backbone.state_dict().items():
                if k not in sd:
                    print(f"Key {k} not found in pretrained model")
                    continue
                if v.shape != sd[k].shape:
                    print(f"Shape mismatch for {k}, {v.shape} != {sd[k].shape}, train the weight from scratch")
                    continue
                load_sd[k] = sd[k]
            msg = self.backbone.load_state_dict(load_sd, strict=False)
            print(f"Loaded pretrained model from {pretrained_ckpt}", msg)

    def forward(self, x):
        y = self.backbone(x)
        return y
        
class PointTransformerV3(PointModule):
    def __init__(
        self,
        embedding_type='PT_embedding',
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        turn_off_bn = False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1
            
        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    SafeBatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            if turn_off_bn:
                bn_layer = nn.Identity
            else:
                bn_layer = partial(SafeBatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        if embedding_type == "PT_embedding":
            self.embedding = Embedding(
                in_channels=in_channels,
                embed_channels=enc_channels[0],
                norm_layer=bn_layer,
                act_layer=act_layer,
            )
        elif embedding_type == "MLP":
            self.embedding = PointSequential(
                nn.Linear(in_channels, enc_channels[0]),
                bn_layer(enc_channels[0]),
                act_layer(),
            )
        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential_intermediate_output()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point, multiscale_point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point