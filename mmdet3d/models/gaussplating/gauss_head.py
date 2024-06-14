import torch.nn as nn
import torch
from .splatting_renderer import render_feature_map
import numpy as np
from scipy.spatial import KDTree
from torch.autograd import Variable
from torchvision.transforms import functional as F
from .utils import silog_loss

nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476,  243808, 2457947, 497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031,141625221, 2307405309])

dynamic_class = [0, 1, 3, 4, 5, 7, 9, 10]

nusc_class_nums = torch.Tensor([
    2854504, 7291443, 141614, 4239939, 32248552, 
    1583610, 364372, 2346381, 582961, 4829021, 
    14073691, 191019309, 6249651, 55095657, 
    58484771, 193834360, 131378779
])

def distCUDA2(points):
    points_np = points.float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)
    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

from mmdet.models import HEADS
@HEADS.register_module()
class GausSplatingHead(nn.Module):
    def __init__(self,
                 point_cloud_range,
                 voxel_size,
                 use_depth_sup=False,
                 use_aux_weight=False,
                 voxel_feature_dim=17,
                 num_classes=18,
                 render_img_shape=None,
                 balance_cls_weight=True,
                 gaussian_sem_weight=1.0,
                 gaussian_dep_weight=1.0,
                 weight_adj = 0.3,
                 weight_dyn = 0.0,
                 white_background = False,
                 x_lim_num=200, 
                 y_lim_num=200, 
                 z_lim_num=16,
                 use_sam=False,
                 use_sam_mask=False,
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
        # coordinates of vox points
        pc_xyz = self.get_presudo_xyz()
        self.pc_xyz = pc_xyz.cuda().requires_grad_(False)
        # scales
        dist = torch.clamp_min(distCUDA2(self.get_presudo_xyz()), 0.0000001)
        scales = torch.log(torch.sqrt(dist))[...,None].repeat(1, 3) 
        self.scales = scales.cuda().requires_grad_(False)
        # rots
        rots = torch.zeros((pc_xyz.shape[0], 4))
        rots[:, 0] = 1
        self.rots = rots.cuda().requires_grad_(False)
        # activation function
        self.scale_act = torch.exp
        self.rot_act = torch.nn.functional.normalize
        # size of rendered image
        self.render_image_height=render_img_shape[0]
        self.render_image_width=render_img_shape[1]

        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
        else:
            self.class_weights = torch.ones(17)/17    
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
    
    def forward(self, voxel_feats, cameras, opacity, **kwargs):
        loss_render_sem_batch = 0
        for batch_id in range(voxel_feats.shape[0]):
            view_points = [c[batch_id] for c in cameras[:-4]]
            vox_feature_i = voxel_feats[batch_id]
            opacity_i = opacity[batch_id]
            gt_sem_batch_id = cameras[-2][batch_id]
            sem_label_mask_batch_id  = cameras[-1][batch_id]
            opacity_i = opacity_i.reshape(-1,1)
            vox_feature_i = vox_feature_i.reshape(-1, self.voxel_feature_dim)
            loss_render_sem = 0
            for c_id in range(view_points[0].shape[0]):
                view_point = [p[c_id] for p in view_points]
                rendered_semantic_map = render_feature_map(
                    feature_dim = self.voxel_feature_dim,
                    viewpoint_camera=view_point,
                    voxel_xyz=self.pc_xyz, # n*3
                    opacity=opacity_i, # n*1
                    scaling=self.scale_act(self.scales), # n*3
                    rotations=self.rot_act(self.rots), # n*4
                    voxel_features=vox_feature_i,  # n*C_v
                    white_background = self.white_background,
                    render_image_height=self.render_image_height,
                    render_image_width=self.render_image_width
                )
                rendered_semantic_map = rendered_semantic_map.permute(1,2,0)  # torch.Size([17, 450, 800]) --> torch.Size([450, 800, 17])
                rendered_semantic_map = rendered_semantic_map.reshape(-1, self.num_classes-1)
                # interpolate label mask(where there is label) 
                sem_label_mask = sem_label_mask_batch_id[c_id]
                sem_label_mask = torch.nn.functional.interpolate(sem_label_mask.unsqueeze(0).unsqueeze(0), size=(self.render_image_height, self.render_image_width), mode='nearest').squeeze(0).squeeze(0)
                sem_label_mask = sem_label_mask.reshape(-1).bool()
                # interpolate semantic label
                gt_sem = gt_sem_batch_id[c_id]   
                gt_sem = torch.nn.functional.interpolate(gt_sem.unsqueeze(0).unsqueeze(0), size=(self.render_image_height, self.render_image_width), mode='nearest').squeeze(0).squeeze(0)
                gt_sem = gt_sem.reshape(-1).long()
                rendered_semantic_map_masked = torch.masked_select(rendered_semantic_map, sem_label_mask.unsqueeze(1))
                rendered_semantic_map_masked = rendered_semantic_map_masked.view(-1, self.num_classes-1)
                # mask the semantic label
                gt_sem_masked = torch.masked_select(gt_sem, sem_label_mask)
                # semantic loss 
                criterion = nn.CrossEntropyLoss(weight=self.class_weights.type_as(rendered_semantic_map_masked), reduction="mean")
                loss_sem_id = criterion(rendered_semantic_map_masked, gt_sem_masked)
                loss_render_sem = loss_render_sem + loss_sem_id
            loss_render_sem_batch = loss_render_sem_batch + loss_render_sem / view_points[0].shape[0]
        loss_sem = loss_render_sem_batch / voxel_feats.shape[0] * self.gaussian_sem_weight  
        return {"render_sem_loss": loss_sem}
    
    