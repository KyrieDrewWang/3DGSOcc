import torch.nn as nn
import torch
from .splatting_renderer import render_feature_map

from mmdet.models import HEADS
@HEADS.register_module()
class GausSplatingHead(nn.Module):
    def __init__(self,
                 point_cloud_range,
                 voxel_size,
                 use_depth_sup=False,
                 voxel_feature_dim=32
                 ) -> None:
        super().__init__()
        self.pc_range=point_cloud_range
        self.voxel_size = voxel_size
        self.use_depth_sup = use_depth_sup
        self.xyz_min = torch.Tensor(self.pc_range[:3])
        self.xyz_max = torch.Tensor(self.pc_range[3:])
        self.xyz_range = (self.xyz_max - self.xyz_min).float()
        self.voxel_feature_dim = voxel_feature_dim
        self.softplus_activation = nn.Softplus()
        
        self.pc_xyz = nn.Parameter(self.get_presudo_xyz(), requires_grad=False)
        dist = torch.tensor(self.voxel_size).repeat(self.pc_xyz.shape[0])
        self.scales = nn.Parameter(torch.log(dist)[:, None], requires_grad=False)
        # scales.requires_grad_(True)
        self.rots = torch.zeros((self.pc_xyz.shape[0], 4))
        self.rots[:, 0] = 1
        self.rots = nn.Parameter(self.rots, requires_grad=False)
    
    def compute_loss(self,):
        pass
      
    # def get_presudo_xyz(self, vox_feature):
    #     x_lim, y_lim, z_lim, dim = vox_feature.shape
    def get_presudo_xyz(self):
        x_lim, z_lim = 200, 16
        vox_grid_xy1 = torch.range(self.xyz_min[0], self.xyz_max[0], self.voxel_size)[:x_lim]
        vox_grid_xy2 = torch.range(self.xyz_min[0], self.xyz_max[0], self.voxel_size)[1:]
        vox_grid_xy  = (vox_grid_xy1 + vox_grid_xy2) / 2
        
        vox_grid_z1 = torch.range(self.xyz_min[-1], self.xyz_max[-1], self.voxel_size)[:z_lim]
        vox_grid_z2 = torch.range(self.xyz_min[-1], self.xyz_max[-1], self.voxel_size)[1:]
        vox_grid_z = (vox_grid_z1 + vox_grid_z2) / 2
            
        X, Y, Z = torch.meshgrid(vox_grid_xy,
                                 vox_grid_xy,
                                 vox_grid_z)
        
        voxel_points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        return voxel_points
    
    def forward(self, vox_features, cameras, density, bad=None, **kwargs):
        
        rendered_feature_map_batch = []
        for batch_id in range(vox_features.shape[0]):
            view_points = [c[batch_id] for c in cameras[:-1]]
            vox_feature_i = vox_features[batch_id]
            density_i = density[batch_id]  
            rendered_feature_map_batch_i = []
            for c_id in range(view_points[0].shape[0]):
                view_point = [p[c_id] for p in view_points]
                rendered_feature_map = render_feature_map(
                    viewpoint_camera=view_point,
                    voxel_xyz=self.pc_xyz,
                    opacity=self.softplus_activation(density_i).reshape(-1,1),
                    scaling=self.scales,
                    rotations=self.rots,
                    voxel_features=vox_feature_i.reshape(-1,self.voxel_feature_dim)
                )
                rendered_feature_map_batch_i.append(rendered_feature_map["render"])
            rendered_feature_map_batch_i = torch.stack(rendered_feature_map_batch_i)
            rendered_feature_map_batch.append(rendered_feature_map_batch_i) 
        rendered_feature_map_batch = torch.stack(rendered_feature_map_batch) 
               
        return rendered_feature_map_batch
    
    