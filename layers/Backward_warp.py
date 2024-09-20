import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.layers.backprojection import Backprojection
from src.layers.projection import Projection
from src.layers.transformation3d import Transformation3D

class BackwardWarping(nn.Module):

    def __init__(self,
                 out_hw: Tuple[int,int],
                 device: torch.device,
                 K:torch.Tensor) -> None:
        super(BackwardWarping,self).__init__()
        height, width = out_hw
        self.backproj = Backprojection(height,width).to(device)
        self.projection = Projection(height,width).to(device)
        self.transform3d = Transformation3D().to(device)

        H,W = height,width
        self.rgb = torch.zeros(H,W,3).view(-1,3).to(device)
        self.depth = torch.zeros(H, W, 1).view(-1, 1).to(device)
        self.K = K.to(device)
        self.inv_K = torch.inverse(K).to(device)
        self.K = self.K.unsqueeze(0)
        self.inv_K = self.inv_K.unsqueeze(0) # 1,4,4

    def forward(self,
                img_tgt: torch.Tensor,
                depth_tgt: torch.Tensor,
                depth_ref: torch.Tensor,
                T: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = img.shape

        # reproject
        pts3d = self.backproj(depth_ref,self.inv_K)
        pts3d_nv = self.transform3d(pts3d,T)
        nv_grid = self.projection(pts3d_nv,self.K,normalized=True)
        transformed_distance = pts3d_nv[:, 2:3].view(b,1,h,w)

        nv_img = F.grid_sample(img_tgt, nv_grid, mode = 'bilinear', padding_mode = 'zeros')
        nv_depth_trans = F.grid_sample(depth_tgt, nv_grid, mode='bilinear', padding_mode='zeros')

        # rm invalid depth
        valid_depth_mask = (transformed_distance < 1e6) & (nv_depth_trans > 0)

        # rm invalid coords
        vaild_coord_mask = (nv_grid[...,0]> -1) & (nv_grid[...,0] < 1) & (nv_grid[...,1]> -1) & (nv_grid[...,1] < 1)
        vaild_coord_mask.unsqueeze(1)

        valid_mask = valid_depth_mask & vaild_coord_mask
        nv_mask = ~valid_mask

        return nv_img.float(), nv_depth_trans.float(), nv_mask.float()


    # forwapr_warp = ForwardWarping(out_hw = out_hw, device = device, K= K)
    # ref2tgt = np.linalg.inv(c2w_tgt) @ c2w_ref
    # nv_img, nv_depth, nv_mask = forwarp_warp(img,depth,ref2tgt)