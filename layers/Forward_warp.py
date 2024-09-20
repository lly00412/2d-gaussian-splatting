import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from src.layers.backprojection import Backprojection
from src.layers.projection import Projection
from src.layers.transformation3d import Transformation3D

def sort_tensor(tensor: torch.Tensor, sort_ref: torch.Tensor) -> torch.Tensor:
    # [N,C], [N] --> [N,C]
    sorted_tensor, indices = torch.sort(sort_ref)
    sorted_tensor = tensor[indices]
    return  sorted_tensor

def find_first_occurance(indexing_tensor: torch.Tensor) -> torch.Tensor:
    unique, idx, counts = torch.unique(indexing_tensor, sorted=True, return_inverse=True,return_counts=True)
    _, ind_sorted = torch.sort(idx,stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((cum_sum[0:1]*0.,cum_sum[:-1])).long()
    first_indicies = ind_sorted[cum_sum]
    return first_indicies

class ForwardWarping(nn.Module):

    def __init__(self,
                 out_hw: Tuple[int,int],
                 device: torch.device,
                 K:torch.Tensor) -> None:
        super(ForwardWarping,self).__init__()
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

    def create_rgbdm(self,
                     inputs: torch.Tensor,
                     out_hw:Tuple[int,int],
                     ) -> torch.Tensor:
        H,W = out_hw
        inputs = sort_tensor(inputs,inputs[:,5])

        # rm invaild depth
        valid_mask = inputs[:, 5] < 1e6
        valid_mask = valid_mask & (inputs[:,5] > 0)

        inputs = inputs[valid_mask]

        # find unique
        inputs[:,:2] = torch.round(inputs[:,:2])

        inputs[:, 0 ] = torch.clip(inputs[:,0],0, W-1)
        inputs[:, 1] = torch.clip(inputs[:,1],0, H-1)

        uv_id = inputs[:, 1]*W + inputs[:, 0]
        uv_1d_unique_idx = find_first_occurance(uv_id)

        inputs = inputs[uv_1d_unique_idx]

        rgb = self.rgb* 0.
        depth = self.depth*0. + 1e8

        uv_1d_unique_idx2 = (inputs[:, 1]* W + inputs[:, 0]).long()
        rgb[uv_1d_unique_idx2] = inputs[:, 2:5]
        depth[uv_1d_unique_idx2] = inputs[:, 5:6]

        rgb = rgb.reshape(H,W,3)
        depth = depth.reshape(H,W,1)
        mask = (depth ==0)*1.

        rgbdm = torch.cat([rgb,depth,mask],dim=-1)

        # inpaint nearby 4 cornners
        rgbdm00 = rgbdm[:, W//4: W//4*3].clone()
        rgbdm01 = rgbdm[:, W // 4: W // 4 * 3].clone()
        rgbdm10 = rgbdm[:, W // 4: W // 4 * 3].clone()
        rgbdm11 = rgbdm[:, W // 4: W // 4 * 3].clone()

        rgbdm01[1:, :] = rgbdm00[:-1, :]
        rgbdm10[:, 1:] = rgbdm00[:, :-1]
        rgbdm11[1:,1:] = rgbdm00[:-1,:-1]

        channel = rgbdm.shape[-1]
        depths = torch.stack([rgbdm00[:,:,3], rgbdm01[:,:,3],rgbdm10[:,:,3],rgbdm11[:,:,3]])
        reshaped_rgbdm = torch.stack(
            [rgbdm00.reshape(-1,channel), rgbdm01.reshape(-1, channel), rgbdm10.reshape(-1, channel),
             rgbdm11.reshape(-1, channel)] # 4, H*W, C
        )
        reshaped_rgbdm = reshaped_rgbdm.permute(1,2,0)

        z_buffer_idx = torch.min(depths, dim=0).indices.reshape(-1)

        sorted_rgbdm = []

        for i in range(channel):
            selected_input = torch.gather(reshaped_rgbdm[:, i],1, z_buffer_idx.view(-1,1))
            sorted_rgbdm.append(selected_input)
        rgbdm_half = torch.stack(sorted_rgbdm,dim=1) # H*W,c
        rgbdm[:, W//4: W//4*3] = rgbdm_half

        return rgbdm.float()

    def forward_warping(self,
                        img: torch.Tensor, # N,3,H,W
                        depth: torch.Tensor, # N,1,H,W
                        coord: torch.Tensor, # N,H,W，2
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # (N,3,H,W), (N,1,H,W), (N,1,H,W)
        N, _, H,W = depth.shape
        uv = coord.permute(0,3,1,2)
        forwarp_input = torch.cat([uv,img,depth],dim=-1)
        input_dim = 6
        forwarp_input = forwarp_input.reshape(N,input_dim,-1).permute(0,2,1) # N,M,D

        rgbdm_all =[]
        for i in range(N):
            rgbdm = self.create_rgbdm(forwarp_input[i],(H,W))
            rgbdm_all.append(rgbdm)

        rgbdm_all = torch.stack(rgbdm_all)
        rgbdm_all = rgbdm_all.permute(0,3,1,2)
        new_rgb, new_depth, new_mask = rgbdm_all[:, :3], rgbdm_all[:, 3:4], rgbdm_all[:, 4:5]

        return new_rgb,new_depth,new_mask

    def forward(self,
                img: torch.Tensor,
                depth: torch.Tensor,
                T: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = img.shape

        pts3d = self.backproj(depth,self.inv_K)

        pts3d_nv = self.transform3d(pts3d,T)

        nv_grid = self.projection(pts3d_nv,self.K,normalized=False)

        transformed_distance = pts3d_nv[:, 2:3].view(b,1,h,w)

        nv_img,nv_depth,nv_mask = self.forward_warping(img, transformed_distance,nv_grid)

        invaild_depth_mask = (nv_depth>1e6)*1.
        nv_mask = ((nv_mask + invaild_depth_mask) > 0)* 1.

        return nv_img.float(), nv_depth.float(), nv_mask.float()


    # forwapr_warp = ForwardWarping(out_hw = out_hw, device = device, K= K)
    # ref2tgt = np.linalg.inv(c2w_tgt) @ c2w_ref
    # nv_img, nv_depth, nv_mask = forwarp_warp(img,depth,ref2tgt)