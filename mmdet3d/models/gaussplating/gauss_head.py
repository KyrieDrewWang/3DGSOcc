import torch.nn as nn
import torch
from .splatting_renderer import render_feature_map
import numpy as np
from scipy.spatial import KDTree
from torch.autograd import Variable
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476,  243808, 2457947, 497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031,141625221, 2307405309])


def distCUDA2(points):
    points_np = points.float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)
    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)


from mmdet.models import HEADS
@HEADS.register_module()
class GausSplatingHead(nn.Module):
    def __init__(self,
                 point_cloud_range,
                 voxel_size,
                 use_depth_sup=False,
                 voxel_feature_dim=17,
                 num_classes=18,
                 sem_mask_size=(512, 1408),
                 balance_cls_weight=True,
                 gaussian_sem_weight=1.0,
                 white_background = False,
                 x_lim_num=200, 
                 y_lim_num=200, 
                 z_lim_num=16,
                 ) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.use_depth_sup = use_depth_sup
        self.xyz_min = torch.Tensor(point_cloud_range[:3])
        self.xyz_max = torch.Tensor(point_cloud_range[3:])
        self.xyz_range = (self.xyz_max - self.xyz_min).float()
        self.voxel_feature_dim = voxel_feature_dim
        self.num_classes = num_classes
        self.white_background = white_background
        self.x_lim_num=x_lim_num
        self.y_lim_num=y_lim_num
        self.z_lim_num=z_lim_num
        # parameters for splatting rasterizing
        pc_xyz = self.get_presudo_xyz()
        self.pc_xyz = pc_xyz.requires_grad_(False)
        dist = torch.clamp_min(distCUDA2(self.get_presudo_xyz()), 0.0000001)
        scales = torch.log(torch.sqrt(dist))[...,None].repeat(1, 3)
        self.scales = scales.requires_grad_(False)
        rots = torch.zeros((pc_xyz.shape[0], 4))
        rots[:, 0] = 1        
        self.rots = rots.requires_grad_(False)
        self.scale_act = torch.exp
        self.rot_act = torch.nn.functional.normalize

        # self.splatting_semantic_mlp = nn.Sequential(
        #     nn.Linear(self.voxel_feature_dim, self.voxel_feature_dim*2),
        #     nn.Softplus(),
        #     nn.Linear(self.voxel_feature_dim*2, num_classes-1),
        # )
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
        else:
            self.class_weights = torch.ones(17)/17    
        self.bce_contrastive_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean")
        self.gaussian_sem_weight=gaussian_sem_weight
    
    def get_presudo_xyz(self):
        x_lim_num, y_lim_num, z_lim_num = self.x_lim_num, self.y_lim_num, self.z_lim_num
        
        vox_grid_x1 = torch.arange(self.xyz_min[0], self.xyz_max[0]+self.voxel_size-0.1, self.voxel_size)[:x_lim_num]
        vox_grid_x2 = torch.arange(self.xyz_min[0], self.xyz_max[0]+self.voxel_size-0.1, self.voxel_size)[1:]
        vox_grid_x  = (vox_grid_x1 + vox_grid_x2) / 2
    
        vox_grid_y1 = torch.arange(self.xyz_min[1], self.xyz_max[1]+self.voxel_size-0.1, self.voxel_size)[:y_lim_num]
        vox_grid_y2 = torch.arange(self.xyz_min[1], self.xyz_max[1]+self.voxel_size-0.1, self.voxel_size)[1:]
        vox_grid_y  = (vox_grid_y1 + vox_grid_y2) / 2
  
        vox_grid_z1 = torch.arange(self.xyz_min[-1], self.xyz_max[-1]+self.voxel_size-0.1, self.voxel_size)[:z_lim_num]
        vox_grid_z2 = torch.arange(self.xyz_min[-1], self.xyz_max[-1]+self.voxel_size-0.1, self.voxel_size)[1:]
        vox_grid_z = (vox_grid_z1 + vox_grid_z2) / 2
            
        X, Y, Z = torch.meshgrid(vox_grid_x,
                                 vox_grid_y,
                                 vox_grid_z)
        
        voxel_points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        return voxel_points
    
    def forward(self, voxel_feats, cameras, density, **kwargs):
        loss_sem_batch = 0
        for batch_id in range(voxel_feats.shape[0]):
            view_points = [c[batch_id] for c in cameras[:-1]]
            vox_feature_i = voxel_feats[batch_id]
            density_i = density[batch_id]  
            gt_sem_batch_id = cameras[-1][batch_id]
            density_i = density_i.reshape(-1,1)
            vox_feature_i = vox_feature_i.reshape(-1, self.voxel_feature_dim)
            loss_sem_c_id = 0
            for c_id in range(view_points[0].shape[0]):
                view_point = [p[c_id] for p in view_points]
                rendered_feature_map = render_feature_map(
                    feature_dim = self.voxel_feature_dim,
                    viewpoint_camera=view_point,
                    voxel_xyz=self.pc_xyz.to(vox_feature_i), # n*3
                    opacity=density_i, # n*1
                    scaling=self.scale_act((self.scales.to(vox_feature_i))), # n*3
                    rotations=self.rot_act(self.rots.to(vox_feature_i)), # n*4
                    voxel_features=vox_feature_i,  # n*32
                    white_background = self.white_background,
                )
                # rendered_semantic_map = self.splatting_semantic_mlp(rendered_feature_map.permute(1,2,0))
                rendered_semantic_map = rendered_feature_map.permute(1,2,0)
                # print(rendered_semantic_map.shape)
                rendered_semantic_map = rendered_semantic_map.reshape(-1, self.num_classes-1)
                gt_sem = gt_sem_batch_id[c_id]   
                gt_sem = gt_sem.reshape(-1).long()
                mask = gt_sem != 0
                render_feature_map_masked = torch.masked_select(rendered_semantic_map, mask.unsqueeze(1)).reshape(-1, self.voxel_feature_dim)
                gt_sem_masked = torch.masked_select(gt_sem, mask).long()
                loss_sem_id = self.bce_contrastive_loss(render_feature_map_masked, gt_sem_masked)
                loss_sem_c_id = loss_sem_c_id + loss_sem_id
            loss_sem_c_id = loss_sem_c_id / view_points[0].shape[0]
            loss_sem_batch = loss_sem_batch + loss_sem_c_id
        loss_sem = loss_sem_batch / voxel_feats.shape[0] * self.gaussian_sem_weight           
        return {"render_sem_loss": loss_sem}
    
    