import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GetVirtualCam:
    def __init__(self, ref_c2w):
        super(GetVirtualCam, self).__init__()
        self.ref_c2w = ref_c2w # 3x4
        self.device = ref_c2w.device
        self.get_camera_direction_and_center()

    def get_camera_direction_and_center(self):
        self.cam_o = self.ref_c2w[:3,3].clone()
        self.left = self.ref_c2w[:3, 0].clone()
        self.up = self.ref_c2w[:3, 1].clone()
        self.forward = self.ref_c2w[:3, 2].clone()


    def get_scene_center(self):
        if not self.dense:
            return self.get_scene_center_sparse()
        else:
            return self.get_scene_center_dense()

    def get_scene_center_dense(self):
        depth_map = self.ref_depth_map.clone().to(self.device)
        height, width = self.ref_depth_map.shape

        ref_c2w = torch.eye(4)
        ref_c2w[:3] = self.ref_c2w.clone().cpu()
        ref_c2w = ref_c2w.to(device=self.device, dtype=torch.float32)
        ref_w2c = torch.inverse(ref_c2w)

        K = torch.eye(4)
        K[:3, :3] = self.K.clone().cpu()
        K = K.to(ref_w2c)

        bwd_proj = torch.matmul(ref_c2w, torch.inverse(K)).to(torch.float32)
        bwd_rot = bwd_proj[:3, :3]
        bwd_trans = bwd_proj[:3, 3:4]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                               torch.arange(0, width, dtype=torch.float32)],
                              indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.reshape(height * width), x.reshape(height * width)
        homog = torch.stack((x, y, torch.ones_like(x))).to(bwd_rot)

        # get world coords
        world_coords = torch.matmul(bwd_rot, homog)
        world_coords = world_coords * depth_map.reshape(1, -1)
        world_coords = world_coords + bwd_trans.reshape(3, 1)
        world_coords = torch.movedim(world_coords, 0, 1) # (h w) 3

        world_coords_real = world_coords[self.opacity>0]
        scene_center = world_coords_real.mean(0)

        return scene_center.cpu()

    def get_scene_center_sparse(self):
        depth_map = self.ref_depth_map.clone().to(self.device)
        if len(depth_map) > len(self.pixl_ids):
            depth_map = depth_map[self.pixl_ids]
        height, width = self.img_h, self.img_w

        ref_c2w = torch.eye(4)
        ref_c2w[:3] = self.ref_c2w.clone().cpu()
        ref_c2w = ref_c2w.to(device=self.device, dtype=torch.float32)
        ref_w2c = torch.inverse(ref_c2w)

        K = torch.eye(4)
        K[:3, :3] = self.K.clone().cpu()
        K = K.to(ref_w2c)

        bwd_proj = torch.matmul(ref_c2w, torch.inverse(K)).to(torch.float32)
        bwd_rot = bwd_proj[:3, :3]
        bwd_trans = bwd_proj[:3, 3:4]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                               torch.arange(0, width, dtype=torch.float32)],
                              indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.reshape(height * width), x.reshape(height * width)
        homog = torch.stack((x, y, torch.ones_like(x))).to(bwd_rot)
        homog = homog[...,self.pixl_ids]

        # get world coords
        world_coords = torch.matmul(bwd_rot, homog)
        world_coords = world_coords * depth_map.reshape(1, -1)
        world_coords = world_coords + bwd_trans.reshape(3, 1)
        world_coords = torch.movedim(world_coords, 0, 1) # (n_rays) 3

        world_coords_real = world_coords[self.opacity > 0]
        scene_center = world_coords_real.mean(0)

        return scene_center.cpu()

    def get_near_c2w(self, c2w, theta=5, axis='x'):
        cam_center = c2w[:3, 3:4].clone().to(self.scene_center)
        cam_center = cam_center.squeeze()
        trans_c2s = self.get_translation_matrix(cam_center,self.scene_center)
        rot = self.get_rotation_matrix(theta, axis)

        c2w_homo = torch.eye(4)
        c2w_homo[:3] = c2w.clone().cpu()
        c2w_homo = c2w_homo.to(torch.float32)
        w2c = torch.inverse(c2w_homo)

        w2c = torch.mm(trans_c2s,w2c)
        w2c = torch.mm(rot,w2c)
        w2c = torch.mm(torch.inverse(trans_c2s),w2c)

        new_c2w = torch.inverse(w2c)
        return new_c2w[:3]

    def get_rotation_by_direction(self,theta=5,direction='u'):
        if direction == 'u':
            theta = 0-theta
            rot = self.get_rotation_matrix(theta,axis='x')
        elif direction == 'd':
            rot = self.get_rotation_matrix(theta,axis='x')
        elif direction == 'l':
            theta = 0- theta
            rot = self.get_rotation_matrix(theta, axis='y')
        elif direction == 'r':
            rot = self.get_rotation_matrix(theta,axis='y')
        elif direction == 'f':
            theta = 0
            rot = self.get_rotation_matrix(theta, axis='y')
        elif direction == 'b':
            theta = 180
            rot = self.get_rotation_matrix(theta,axis='y')
        return rot.to(self.device)

    def get_rotation_matrix(self, theta=5, axis='x'): # rot theta degree across x axis
        phi = (theta * (np.pi / 180.))
        rot = torch.eye(4)
        if axis=='x':
            rot[:3,:3] = torch.Tensor([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
            ])
        elif axis == 'y':
            rot[:3,:3] = torch.Tensor([
                [np.cos(phi), 0, -np.sin(phi)],
                [0, 1, 0],
                [np.sin(phi), 0, np.cos(phi)]
            ])
        elif axis=='z':
            rot[:3,:3] = torch.Tensor([
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ])
        return rot

    def get_translation_matrix(self,origin,destination): # both should be (x,y,z)
        trans = torch.eye(4).to(destination)
        trans[:3,3] = destination-origin
        return trans

    def get_translation_by_direction(self,step=0.05, direction='u'):
        trans = torch.eye(4).to(self.cam_o)
        if direction == 'u':
            trans[:3,3] += self.up*step
        elif direction == 'd':
            trans[:3,3] -= self.up*step
        elif direction == 'l':
            trans[:3,3] += self.left*step
        elif direction == 'r':
            trans[:3,3] -= self.left*step
        elif direction == 'f':
            trans[:3,3] += self.forward*step
        elif direction == 'b':
            trans[:3,3] -= self.forward*step
        return trans.to(self.device)

    def get_transform_to_new_center(self,new_o):
        trans = torch.eye(4).to(self.cam_o)
        trans[:3, 3] = new_o.to(self.cam_o) - self.cam_o
        return trans

    def get_target_c2w(self, theta=90,step = 0.05, direction='u'):
        trans = self.get_translation_by_direction(step,'f')
        rot = self.get_rotation_by_direction(theta,direction)

        c2w_homo = torch.eye(4).to(self.ref_c2w)
        c2w_homo[:3] = self.ref_c2w[:3].clone()
        w2c = torch.inverse(c2w_homo)
        w2c = trans @  rot @ w2c
        new_c2w = torch.inverse(w2c)
        return new_c2w

    def get_c2w_at_new_center(self,new_o, theta=90, direction = 'u'):
        trans = self.get_transform_to_new_center(new_o)
        rot = self.get_rotation_by_direction(theta,direction)

        c2w_homo = torch.eye(4).to(self.ref_c2w)
        c2w_homo[:3] = self.ref_c2w[:3].clone()
        w2c = torch.inverse(c2w_homo)
        w2c = rot @ trans @ w2c
        new_c2w = torch.inverse(w2c)
        return new_c2w

    def get_new_c2w_look_at(self,new_o, look_at):
        new_o = new_o.to(self.device)
        look_at = look_at.to(self.device)

        forward = look_at - new_o
        forward /= torch.linalg.norm(forward)
        forward = forward.to(new_o)

        world_up = torch.tensor([0,1,0]).to(new_o) # new to be careful for the openGL system!!!

        right = torch.cross(world_up, forward)
        right /= torch.linalg.norm(right)

        up = torch.cross(forward,right)

        new_c2w = torch.eye(4).to(new_o)
        new_c2w[:3,:3] = torch.vstack([right,up,forward]).T
        new_c2w[:3,3] = new_o
        return new_c2w


    def mark_visible(self, K, c2w, img_wh, depth, voxel_idx,o3d_voxel_grid):

        pt3d_a = []
        for ind in voxel_idx:
            points = o3d_voxel_grid.get_voxel_center_coordinate(ind[:, None]) # (3, 1) can be done by open3d
            pt3d_a.append(points)
        pt3d_a = np.stack(pt3d_a)
        pt3d_a = torch.from_numpy(pt3d_a)
        pt3d_a = pt3d_a.unsqueeze(0).permute(0,2,1)
        b,_,n = pt3d_a.shape
        homo = torch.ones(b,1,n)
        pt3d_homo = torch.cat((pt3d_a,homo),1).float()

        Rt = torch.inverse(c2w)
        points_camera = torch.bmm(Rt, pt3d_homo)
        pt2d = torch.bmm(K, points_camera).squeeze(0).permute(1,0)
        uv = pt2d[:, :2] / pt2d[:, 2:3]
        d = pt2d[:, 2:3]

        xx = uv[:, 0] / (img_wh[0] -1)
        yy = uv[:, 1] / (img_wh[1] -1)
        xx = (xx - 0.5)*2
        yy = (yy-0.5)*2
        # get depth in uv
        sample_gird = torch.stack([xx,yy],1).view(1,-1,1,2)
        depth = depth.unsqueeze(0).unsqueeze(0).to(sample_gird) # 1,1,H,W
        depth_at_uv = F.grid_sample(depth,sample_gird,mode='bilinear',padding_mode='zeros')
        depth_at_uv = depth_at_uv.view(-1)

        NEAREST = 0.05
        FARST = 3.0

        visible = (uv[:, 0] >= 0) & (uv[:, 0] < img_wh[0]) & \
                  (uv[:, 1] >= 0) & (uv[:, 1] < img_wh[1]) & \
                  (d >= NEAREST) & (d < depth_at_uv)

        free = visible
        free_voxel = voxel_idx[free.detach().cpu().numpy()]
        return free_voxel