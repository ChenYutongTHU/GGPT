    
import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import  DataLoader
import os
import random
import cv2
from ggpt.model.pointtransformer_v3 import PointTransformerV3Model
from tqdm import tqdm
import torch.distributed as dist
import hydra
from hydra.utils import instantiate
import torchvision.transforms.v2 as v2
from pointcept.models.sparse_unet.spconv_unet_v1m1_base import SpUNetBase
from pointcept.models.spvcnn.ts_spvcnn import SPVCNN


class BasePredictor(torch.nn.Module):
    def __init__(self, 
        type_embed_dim=16, sinusoidal_dim=32,
        grid_resolution=384,
        predict_residual=True,
        zero_init=True,
        backbone_type='ptv3', 
        ptv3_config={},  head_width=256,
        spconv_config={}, spvcnn_config={}, #spvcnn: torchsparse not instal
        use_ff_only=False,
        use_ff_emb_only=False,
        wo_d2s_emb=False,
        ):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.grid_resolution = grid_resolution
        self.use_ff_only = use_ff_only
        self.use_ff_emb_only = use_ff_emb_only
        self.wo_d2s_emb = wo_d2s_emb
        if self.use_ff_only:
            type_embed_dim = None
        if type_embed_dim is not None:
            self.ff_type_embed = torch.nn.Parameter(torch.randn(type_embed_dim)) #learnable type embedding for feature fusion points
            self.dlt_type_embed = torch.nn.Parameter(torch.randn(type_embed_dim)) #learnable type embedding for dlt points
        else:
            #A small trick
            # set ff_type_embed and dlt_type_embed to zero vector (constant), so that the type embedding is effectively removed
            self.ff_type_embed = torch.nn.Parameter(torch.zeros(1)) 
            self.ff_type_embed.requires_grad = False
            self.dlt_type_embed = torch.nn.Parameter(torch.zeros(1)) 
            self.dlt_type_embed.requires_grad = False
            type_embed_dim = 1
        in_channel = type_embed_dim + 3+sinusoidal_dim*3 + 1 + 3+sinusoidal_dim*3 + 3
        # input: learnable-type-embedding + sinusoidal(xyz) + tgt-dlt-sinusoidal(xyz) + confidence
        self.in_channel = in_channel
        self.backbone_type = backbone_type
        if backbone_type == 'ptv3':
            self.backbone = PointTransformerV3Model(in_channel,  **ptv3_config)
            backbone_output_dim = ptv3_config.output_dim
        elif backbone_type == 'spconv_unet':
            self.backbone = SpUNetBase(in_channel, **spconv_config)
            backbone_output_dim = self.backbone.channels[-1]
        elif backbone_type == 'spvcnn':
            self.backbone = SPVCNN(in_channel, **spvcnn_config)
            backbone_output_dim = self.backbone.out_channels
        self.predict_residual = predict_residual

        self.head = torch.nn.Sequential(
            nn.Linear(backbone_output_dim+in_channel,  head_width),
            nn.ReLU(),
            nn.Linear(head_width, head_width),
            nn.ReLU(),
            nn.Linear(head_width, 4), #output: delta_xyz(3) + confidence 1
        )
        #TODO: an idea) output sim3d rather than delta_xyz?
        self.zero_init = zero_init
        if self.zero_init:
            torch.nn.init.constant_(self.head[-1].weight, 0)
            torch.nn.init.constant_(self.head[-1].bias, 0)
        

        

    def sinusoidal_embedding(self, pos, omega_0=100):
        #pos: N,3 in range [-1,1]
        N, _ = pos.shape
        if N == 0:
            if self.sinusoidal_dim == 0:
                return pos
            w = 3 + self.sinusoidal_dim * 3
            return pos.new_zeros((0, w))
        if self.sinusoidal_dim == 0:
            return pos  
        assert pos.shape[1] == 3
        assert self.sinusoidal_dim % 2 == 0
        device = pos.device
        omega = torch.arange(self.sinusoidal_dim//2, dtype=torch.float32, device=device)
        omega /= self.sinusoidal_dim / 2.0
        omega = 1.0 / omega_0**omega  # (D/2,)
        out = torch.einsum("m,d->md", pos.view(-1), omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (N*3, D)
        emb = emb.view(N,-1)  # (N, 3*D)
        emb = torch.cat([pos, emb], dim=1)  # (N, 3+3*D)
        return emb

    def embed_input(self, batchdata_list):
        B = len(batchdata_list)
        batched_feat = []
        offset = []
        coord = []
        grid_coords = []
        for data in batchdata_list:
            ff_pts_xyz = data['ff_pts']  # K,3
            device, dtype = ff_pts_xyz.device, ff_pts_xyz.dtype
            dlt_pts_xyz_all = data['geo_pts']  # N_all,3
            geo_msks = data['geo_msks']
            geo_empty = (
                not isinstance(dlt_pts_xyz_all, torch.Tensor)
                or dlt_pts_xyz_all.numel() == 0
                or dlt_pts_xyz_all.shape[0] == 0
            )
            dlt_emb_w = (
                3 if self.sinusoidal_dim == 0 else 3 + self.sinusoidal_dim * 3
            )

            if geo_empty:
                dlt_pts_xyz_emb_all = torch.empty(
                    0, dlt_emb_w, device=device, dtype=dtype
                )
                dlt_pts_xyz = torch.empty(0, 3, device=device, dtype=dtype)
            else:
                dlt_pts_xyz_emb_all = self.sinusoidal_embedding(dlt_pts_xyz_all)
                dlt_pts_xyz_emb_all[~geo_msks] = 0
                dlt_pts_xyz = dlt_pts_xyz_all[geo_msks]
            N_dlt_pts = dlt_pts_xyz.shape[0]
            if self.use_ff_only:
                N_dlt_pts = 0
            if N_dlt_pts > 0:
                dlt_pts_xyz_emb = dlt_pts_xyz_emb_all[geo_msks]
                dlt_pts_type_emb = self.dlt_type_embed.unsqueeze(0).expand(
                    dlt_pts_xyz.shape[0], -1
                )
                dlt_pts_feat = torch.zeros(
                    (N_dlt_pts, self.in_channel), device=dlt_pts_xyz.device
                )
                dlt_pts_feat_used = torch.cat(
                    [dlt_pts_type_emb, dlt_pts_xyz_emb], dim=1
                )
                dlt_pts_feat[:, : dlt_pts_feat_used.shape[1]] = dlt_pts_feat_used

            ff_pts_xyz_emb = self.sinusoidal_embedding(ff_pts_xyz)
            ff_pts_type_emb = self.ff_type_embed.unsqueeze(0).expand(
                ff_pts_xyz.shape[0], -1
            )
            ff_pts_conf = data["ff_pts_conf"].unsqueeze(-1)
            if geo_empty:
                ff_pts_tgtdlt_emb = torch.zeros(
                    ff_pts_xyz.shape[0], dlt_emb_w, device=device, dtype=dtype
                )
                ff_pts_totgt_delta = torch.zeros_like(ff_pts_xyz)
            else:
                ff_pts_tgtdlt_emb = dlt_pts_xyz_emb_all
                ff_pts_totgt_delta = (ff_pts_xyz - dlt_pts_xyz_all).reshape(-1, 3)
                if self.use_ff_emb_only or self.wo_d2s_emb:
                    ff_pts_tgtdlt_emb = torch.zeros_like(ff_pts_tgtdlt_emb)
                    ff_pts_totgt_delta = torch.zeros_like(ff_pts_totgt_delta)
                else:
                    ff_pts_totgt_delta[~geo_msks] = 0
            ff_pts_feat = torch.cat(
                [
                    ff_pts_type_emb,
                    ff_pts_xyz_emb,
                    ff_pts_conf,
                    ff_pts_tgtdlt_emb,
                    ff_pts_totgt_delta,
                ],
                dim=1,
            )
        
            if N_dlt_pts > 0:
                feat = torch.cat([ff_pts_feat, dlt_pts_feat], dim=0) #(K+N), in_channel
                coor = torch.cat([ff_pts_xyz, dlt_pts_xyz], dim=0) #(K+N),3
            else:
                feat = ff_pts_feat
                coor = ff_pts_xyz


            batched_feat.append(feat) #(K+N), in_channel
            offset.append(feat.shape[0])
            coord.append(coor) #(K+N),3
            factor = data.get('factor',1)
            grid_coords.append(((coor/factor).clamp(-1,1)+1) /2.0 * self.grid_resolution)

        model_input = {}

        model_input['feat'] = torch.cat(batched_feat, dim=0)
        # DDP: dlt_type_embed is otherwise unused when N_dlt_pts==0 on this rank.
        model_input["feat"] = model_input["feat"] + self.dlt_type_embed.sum() * 0
        model_input['offset'] = torch.tensor(offset).cumsum(dim=0).to(feat.device) #(B,)
        model_input['coord'] = torch.cat(coord, dim=0) #(sum(K+N)),3
        #model_input['grid_coord'] = ((model_input['coord'].clamp(-1,1) + 1) / 2.0 * self.grid_resolution).floor().int().clamp(0,self.grid_resolution-1) # (sum(K+N)),3 in [0, grid_resolution]
        model_input['grid_coord'] = torch.cat(grid_coords, dim=0).floor().int().clamp(0,self.grid_resolution-1) # (sum(K+N)),3 in [0, grid_resolution]
        model_input['grid_size'] = torch.ones([3])*1.0/self.grid_resolution
        return model_input

    def pack_output(self, xyz_out, conf_out, batchdata_list):
        output_dict = {
            'ff_pts_in': [],
            'ff_pts_out': [],
            'ff_pts_conf_out': [],
            'dlt_pts_out': [], #The same shape as ff_pts_out
            'dlt_pts_conf_out': [],
            'dlt_msks': [], #for supervision
            'offset': []
        }
        if 'gt_pts' in batchdata_list[0]:
            output_dict['gt_pts'] = []
            output_dict['gt_msks'] = []
        start_idx = 0
        for data in batchdata_list:
            num_ff = data["ff_pts"].shape[0]
            if self.use_ff_only:
                num_dlt = 0
            else:
                geo, msk = data["geo_pts"], data["geo_msks"]
                if (
                    isinstance(geo, torch.Tensor)
                    and isinstance(msk, torch.Tensor)
                    and geo.shape[0] == msk.shape[0]
                ):
                    num_dlt = int(geo[msk].shape[0])
                else:
                    num_dlt = 0
            end_idx = start_idx + num_ff + num_dlt
            ff_pts_out = xyz_out[start_idx:start_idx+num_ff]
            ff_pts_conf_out = conf_out[start_idx:start_idx+num_ff]
            dlt_pts_out = torch.zeros_like(data["geo_pts"])
            dlt_pts_conf_out = torch.zeros_like(data["geo_pts"][..., 0])
            if not self.use_ff_only and num_dlt > 0:
                dlt_pts_out[data["geo_msks"]] = xyz_out[
                    start_idx + num_ff : end_idx
                ]
                dlt_pts_conf_out[data["geo_msks"]] = conf_out[
                    start_idx + num_ff : end_idx
                ]
            output_dict['ff_pts_in'].append(data['ff_pts'])
            output_dict['ff_pts_out'].append(ff_pts_out)
            output_dict['ff_pts_conf_out'].append(ff_pts_conf_out)
            output_dict['dlt_pts_out'].append(dlt_pts_out)
            output_dict['dlt_pts_conf_out'].append(dlt_pts_conf_out)
            output_dict['dlt_msks'].append(data['geo_msks'])
            if 'gt_pts' in data:
                output_dict['gt_pts'].append(data['gt_pts'])
            if 'gt_msks' in data:
                output_dict['gt_msks'].append(data['gt_msks'])
            start_idx += num_ff + num_dlt
            output_dict['offset'].append(start_idx)
        for k in output_dict.keys():
            if k not in output_dict or len(output_dict[k])==0:
                continue
            if k in ['offset']:
                output_dict[k] = torch.tensor(output_dict[k], device=xyz_out.device)
            else:
                output_dict[k] = torch.cat(output_dict[k], dim=0) #N,3/(1)
        return output_dict

    def forward(self, batchdata_list):
        model_input = self.embed_input(batchdata_list)
        pt_out = self.backbone(model_input)
        pt_out_feat = pt_out['feat']  if self.backbone_type == 'ptv3' else pt_out
        head_in = torch.cat([pt_out_feat, model_input['feat']], dim=1) #(sum(K+N)), in_channel+backbone_output_dim
        head_out = self.head(head_in) #(sum(K+N)),4
        xyz_out, conf_out = head_out[:,:3], head_out[:,3].exp() + 1
        if self.predict_residual:
            xyz_out = model_input['coord'] + xyz_out #TODO: Shall we also supervise DLT points?
        else:
            xyz_out = xyz_out.to(model_input['coord'].dtype)
        output_dict = self.pack_output(xyz_out, conf_out, batchdata_list)
        return output_dict