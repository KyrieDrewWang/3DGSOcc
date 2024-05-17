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
        self.use_aux_weight = use_aux_weight
        self.weight_dyn = weight_dyn
        self.dynamic_class = torch.tensor(dynamic_class)
        self.dynamic_weight = torch.exp(0.005 * (nusc_class_nums.max() / nusc_class_nums - 1))
        self.weight_adj=weight_adj
        self.use_sam_mask = use_sam_mask
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
        if use_sam or use_sam_mask:
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
        self.bce_contrastive_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")
        self.gaussian_sem_weight=gaussian_sem_weight
        self.gaussian_dep_weight=gaussian_dep_weight
        self.depth_loss = silog_loss()

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
        loss_render_sem_batch = 0
        loss_render_depth_batch = 0
        for batch_id in range(voxel_feats.shape[0]):
            view_points = [c[batch_id] for c in cameras[:-4]]
            vox_feature_i = voxel_feats[batch_id]
            opacity_i = opacity[batch_id]
            sam_embd_batch_id = cameras[-4][batch_id]
            gt_sem_batch_id = cameras[-2][batch_id]
            sem_label_mask_batch_id  = cameras[-1][batch_id]
            depth_label_batch_id = cameras[-5][batch_id]
            sam_mask_batch_id = cameras[-3][batch_id]
            opacity_i = opacity_i.reshape(-1,1)
            vox_feature_i = vox_feature_i.reshape(-1, self.voxel_feature_dim)
            loss_render_sem = 0
            loss_render_depth = 0
            for c_id in range(view_points[0].shape[0]):
                view_point = [p[c_id] for p in view_points]
                rendered_semantic_map, rendered_depth_feature_map = render_feature_map(
                    feature_dim = self.voxel_feature_dim,
                    viewpoint_camera=view_point,
                    voxel_xyz=self.pc_xyz.to(vox_feature_i), # n*3
                    opacity=opacity_i, # n*1
                    scaling=self.scale_act((self.scales.to(vox_feature_i))), # n*3
                    rotations=self.rot_act(self.rots.to(vox_feature_i)), # n*4
                    voxel_features=vox_feature_i,  # n*C_v
                    white_background = self.white_background,
                    render_image_height=self.render_image_height,
                    render_image_width=self.render_image_width
                )
                if not self.use_sam and not self.use_sam_mask:
                    rendered_semantic_map = rendered_semantic_map.permute(1,2,0)  # torch.Size([17, 450, 800]) --> torch.Size([450, 800, 17])
                    rendered_semantic_map = rendered_semantic_map.reshape(-1, self.num_classes-1)
                    rendered_depth_feature_map = rendered_depth_feature_map.squeeze(0)
                    rendered_depth_feature_map = rendered_depth_feature_map.reshape(-1)
                    # interpolate label mask(where there is label) 
                    sem_label_mask = sem_label_mask_batch_id[c_id]
                    sem_label_mask = torch.nn.functional.interpolate(sem_label_mask.unsqueeze(0).unsqueeze(0), size=(self.render_image_height, self.render_image_width), mode='nearest').squeeze(0).squeeze(0)
                    sem_label_mask = sem_label_mask.reshape(-1).bool()
                    # interpolate semantic label
                    gt_sem = gt_sem_batch_id[c_id]   
                    gt_sem = torch.nn.functional.interpolate(gt_sem.unsqueeze(0).unsqueeze(0), size=(self.render_image_height, self.render_image_width), mode='nearest').squeeze(0).squeeze(0)
                    gt_sem = gt_sem.reshape(-1).long()
                    # interpolate depth label
                    depth_label = depth_label_batch_id[c_id]
                    depth_label = torch.nn.functional.interpolate(depth_label.unsqueeze(0).unsqueeze(0), size=(self.render_image_height, self.render_image_width), mode='nearest').squeeze(0).squeeze(0)
                    depth_label = depth_label.reshape(-1)
                    # check whether to use aux weight
                    if self.use_aux_weight and c_id in [0, 1, 2, 3, 4, 5]:
                        weight_t = torch.full((gt_sem.shape), self.weight_adj).to(gt_sem.device)
                        dynamic_mask = (self.dynamic_class.to(gt_sem)==gt_sem.unsqueeze(1)).any(-1)
                        weight_t[dynamic_mask] = self.weight_dyn
                        weight_b = self.dynamic_weight.to(gt_sem.device)[gt_sem.long()]
                        aux_weight = weight_t * weight_b
                    else:
                        aux_weight = torch.ones_like(gt_sem)
                    # mask the aux weight by the label mask
                    aux_weight = torch.masked_select(aux_weight, sem_label_mask)
                    # mask the projected semantic feature maps
                    rendered_semantic_map_masked = torch.masked_select(rendered_semantic_map, sem_label_mask.unsqueeze(1))
                    rendered_semantic_map_masked = rendered_semantic_map_masked.view(-1, self.num_classes-1)
                    # mask the semantic label
                    gt_sem_masked = torch.masked_select(gt_sem, sem_label_mask)
                    # mask the depth label
                    depth_label = torch.masked_select(depth_label, sem_label_mask) 
                    # mask the projected depth feature maps
                    rendered_depth_feature_map_masked = torch.masked_select(rendered_depth_feature_map, sem_label_mask)
                    # semantic loss 
                    loss_sem_id = self.bce_contrastive_loss(rendered_semantic_map_masked, gt_sem_masked)
                    loss_sem_id = loss_sem_id * aux_weight
                    loss_sem_id = loss_sem_id.sum()
                    num_label = len(sem_label_mask[sem_label_mask])
                    loss_sem_id = loss_sem_id / num_label
                    # depth loss
                    loss_depth_id = self.depth_loss(rendered_depth_feature_map_masked+1e-7, depth_label)
                '''
                if self.use_sam:
                    rendered_semantic_map = rendered_feature_map.permute(1,2,0)
                    sam_features = sam_embd_batch_id[c_id]
                    H,W = sam_features.shape[-2:]
                    sam_embd_full = torch.nn.functional.interpolate(sam_features, size=rendered_feature_map.shape[1:], mode='bilinear').squeeze()
                    sam_embd = sam_embd_full.permute(1,2,0)
                    sam_embd_down = self.sam_proj(sam_embd)
                    loss_sem_id = l1_loss(rendered_semantic_map, sam_embd_down)
                    loss_render_id = loss_sem_id

                if self.use_sam_mask:
                    sam_features = sam_embd_batch_id[c_id]
                    sam_masks = sam_mask_batch_id[c_id]
                    H,W = sam_features.shape[-2:]
                    low_dim_sam_features = self.sam_proj(sam_features.reshape(-1, H*W).permute([1,0])).permute([1,0]).reshape(self.voxel_feature_dim, H, W)

                    low_sam_masks = torch.nn.functional.interpolate(sam_masks.unsqueeze(0), size=sam_features.shape[-2:], mode='nearest').squeeze()
                    if len(low_sam_masks.shape) < 3:
                        continue
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

                    # regulation loss:
                    rands = torch.rand(vox_feature_i.shape[0], device=prob.device)
                    reg_loss = torch.relu(torch.einsum('NC,KC->NK', vox_feature_i[rands > 0.9, :], prototypes)).mean()
                    loss_sem_id = loss_sem_id + 0.1*reg_loss

                    # coorespondence loss
                    NHW = low_sam_masks  # N 64 64
                    N,H,W = NHW.shape
                    NL = NHW.view(N,-1)
                    intersection = torch.einsum('NL,NC->LC', NL, NL) # 64^2 64^2
                    union = NL.sum(dim = 0, keepdim = True) + NL.sum(dim = 0, keepdim = True).T - intersection
                    similarity = intersection / (union + 1e-5)
                    HWHW = similarity.view(H,W,H,W)
                    HWHW[HWHW == 0] = -1
                    norm_rendered_feature = torch.nn.functional.normalize(torch.nn.functional.interpolate(rendered_feature_map.unsqueeze(0), (H,W), mode = 'bilinear').squeeze(), dim=0, p=2)
                    correspondence = torch.relu(torch.einsum('CHW,CJK->HWJK', norm_rendered_feature, norm_rendered_feature))
                    corr_loss = -HWHW * correspondence
                    loss_sem_id = loss_sem_id + corr_loss.mean()
                    loss_render_id = loss_sem_id
                '''
                loss_render_sem = loss_render_sem + loss_sem_id
                loss_render_depth = loss_render_depth + loss_depth_id
            loss_render_sem_batch = loss_render_sem_batch + loss_render_sem
            loss_render_depth_batch = loss_render_depth_batch + loss_render_depth

        loss_sem = loss_render_sem_batch / voxel_feats.shape[0] * self.gaussian_sem_weight  
        loss_dep = loss_render_depth_batch / voxel_feats.shape[0] * self.gaussian_dep_weight
        return {"render_sem_loss": loss_sem,
                "render_depth_loss": loss_dep}
    
    