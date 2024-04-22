import torch.nn as nn
import torch
from .splatting_renderer import render_feature_map
import numpy as np
from scipy.spatial import KDTree
from torch.autograd import Variable
from torchvision.transforms import functional as F
import cv2
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476,  243808, 2457947, 497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031,141625221, 2307405309])
import sys
sys.path.append('tools/sam_encoder/')
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

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
                 voxel_feature_dim=17,
                 num_classes=18,
                 render_img_shape=None,
                 balance_cls_weight=True,
                 gaussian_sem_weight=1.0,
                 white_background = False,
                 x_lim_num=200, 
                 y_lim_num=200, 
                 z_lim_num=16,
                 use_sam=False
                 ) -> None:
        super().__init__()
        self.use_sam = use_sam
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
        # self.pc_xyz = nn.Parameter(pc_xyz)
        dist = torch.clamp_min(distCUDA2(self.get_presudo_xyz()), 0.0000001)
        scales = torch.log(torch.sqrt(dist))[...,None].repeat(1, 3)
        self.scales = scales.requires_grad_(False)
        # self.scales = nn.Parameter(scales)
        rots = torch.zeros((pc_xyz.shape[0], 4))
        rots[:, 0] = 1        
        self.rots = rots.requires_grad_(False)
        # self.rots = nn.Parameter(rots)
        self.scale_act = torch.exp
        self.rot_act = torch.nn.functional.normalize
        self.render_image_height=render_img_shape[0]
        self.render_image_width=render_img_shape[1]
        self.sam_proj = torch.nn.Sequential(
            torch.nn.Linear(256, 64, bias=True),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64, bias=True),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, voxel_feature_dim, bias=True)
        )
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

        sam = sam_model_registry["vit_h"]("ckpts/sam_vit_h_4b8939.pth").to('cuda')
        self.SAM_encoder = SamPredictor(sam)
        self.SAM_decoder = SamAutomaticMaskGenerator(
        sam, 
        pred_iou_thresh = 0.88, 
        stability_score_thresh = 0.95, 
        min_mask_region_area = 0
    )
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
    
    def forward(self, voxel_feats, cameras, opacity, imgs, **kwargs):
        loss_sem_batch = 0
        for batch_id in range(voxel_feats.shape[0]):
            view_points = [c[batch_id] for c in cameras[:-4]]
            vox_feature_i = voxel_feats[batch_id]
            opacity_i = opacity[batch_id]  
            sam_embd_batch_id = cameras[-4][batch_id]
            # gt_sem_batch_id = cameras[-2][batch_id]
            # sem_label_mask_batch_id  = cameras[-1][batch_id]
            view_imgs = cameras[-3][batch_id]
            opacity_i = opacity_i.reshape(-1,1)
            vox_feature_i = vox_feature_i.reshape(-1, self.voxel_feature_dim)
            loss_sem_c_id = 0
            
            for c_id in range(view_points[0].shape[0]):
                view_point = [p[c_id] for p in view_points]
                rendered_feature_map = render_feature_map(
                    feature_dim = self.voxel_feature_dim,
                    viewpoint_camera=view_point,
                    voxel_xyz=self.pc_xyz.to(vox_feature_i), # n*3
                    opacity=opacity_i, # n*1
                    scaling=self.scale_act((self.scales.to(vox_feature_i))), # n*3
                    rotations=self.rot_act(self.rots.to(vox_feature_i)), # n*4
                    voxel_features=vox_feature_i,  # n*32
                    white_background = self.white_background,
                    render_image_height=self.render_image_height,
                    render_image_width=self.render_image_width
                )
                # rendered_semantic_map = self.splatting_semantic_mlp(rendered_feature_map.permute(1,2,0))
                # print(rendered_semantic_map.shape)
                # rendered_semantic_map = rendered_semantic_map.reshape(-1, self.num_classes-1)
                # sem_label_mask = sem_label_mask_batch_id[c_id]
                # sem_label_mask = F.resize(sem_label_mask.unsqueeze(0), size=(self.render_image_height, self.render_image_width)).squeeze(0)
                # sem_label_mask = sem_label_mask.reshape(-1).bool()
                # gt_sem = gt_sem_batch_id[c_id]   
                # gt_sem = F.resize(gt_sem.unsqueeze(0), size=(self.render_image_height, self.render_image_width)).squeeze(0)
                # gt_sem = gt_sem.reshape(-1).long()
                # # mask by the projected labels
                # rendered_semantic_map_masked = torch.masked_select(rendered_semantic_map, sem_label_mask.unsqueeze(1))
                # rendered_semantic_map_masked = rendered_semantic_map_masked.view(-1, self.num_classes-1)
                # gt_sem_masked = torch.masked_select(gt_sem, sem_label_mask)
                # loss_sem_id = self.bce_contrastive_loss(rendered_semantic_map_masked, gt_sem_masked)
                if self.use_sam:
                    rendered_semantic_map = rendered_feature_map.permute(1,2,0)
                    sam_embd_full = torch.nn.functional.interpolate(sam_embd_batch_id[c_id].unsqueeze(0), size=rendered_feature_map.shape[1:]).squeeze()
                    sam_embd = sam_embd_full.permute(1,2,0)
                    sam_embd_down = self.sam_proj(sam_embd)
                    loss_sem_id = l1_loss(rendered_semantic_map, sam_embd_down)
                    loss_sem_c_id = loss_sem_c_id + loss_sem_id
                v_img = view_imgs[c_id]
                with torch.no_grad():
                    img4feature = cv2.resize(v_img.cpu().numpy(), dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
                    self.SAM_encoder.set_image(img4feature)
                    sam_features = self.SAM_encoder.features
                    sam_masks = self.SAM_decoder.generate(v_img.cpu().numpy())
                    mask_list = []
                    for m in sam_masks:
                        m_score = torch.from_numpy(m['segmentation']).float()[None, None, :, :].to('cuda')
                        m_score = torch.nn.functional.interpolate(m_score, size=(200,200) , mode='bilinear', align_corners=False).squeeze()
                        m_score[m_score >= 0.5] = 1
                        m_score[m_score != 1] = 0
                        mask_list.append(m_score)
                    sam_masks = torch.stack(mask_list, dim=0)
                H,W = sam_features.shape[-2:]
                low_dim_sam_features = self.sam_proj(sam_features.reshape(-1, H*W).permute([1,0])).permute([1,0]).reshape(self.voxel_feature_dim, H, W)

                low_sam_masks = torch.nn.functional.interpolate(sam_masks.unsqueeze(0), size=sam_features.shape[-2:] , mode='nearest').squeeze()
                nonzero_masks = low_sam_masks.sum(dim=(1,2)) > 0
                low_sam_masks = low_sam_masks[nonzero_masks,:,:]
                full_resolution_sam_masks = sam_masks[nonzero_masks,:,:]

                prototypes = (low_sam_masks.unsqueeze(1) * low_dim_sam_features).sum(dim = (2,3))
                prototypes /= low_sam_masks.sum(dim=(1,2)).unsqueeze(-1)

                pp = torch.einsum('NC, CHW -> NHW', prototypes, rendered_feature_map)
                prob = torch.sigmoid(pp)

                full_resolution_sam_masks = torch.nn.functional.interpolate(full_resolution_sam_masks.unsqueeze(0), size=prob.shape[-2:] , mode='bilinear').squeeze()
                full_resolution_sam_masks[full_resolution_sam_masks <= 0.5] = 0

                bce_contrastive_loss = full_resolution_sam_masks * torch.log(prob + 1e-8) + ((1 - full_resolution_sam_masks) * torch.log(1 - prob + 1e-8))
                loss_sem_id = -bce_contrastive_loss.mean()
                loss_sem_c_id = loss_sem_c_id + loss_sem_id

            loss_sem_c_id = loss_sem_c_id / view_points[0].shape[0]
            loss_sem_batch = loss_sem_batch + loss_sem_c_id
        loss_sem = loss_sem_batch / voxel_feats.shape[0] * self.gaussian_sem_weight           
        return {"render_sem_loss": loss_sem}
    
    