# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
import mmcv
import mmcv.parallel
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
import math
from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore
from .ray import generate_rays
import sys
sys.path.append('tools/sam_encoder/')
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)


nusc_class_nums = torch.Tensor([
    2854504, 7291443, 141614, 4239939, 32248552, 
    1583610, 364372, 2346381, 582961, 4829021, 
    14073691, 191019309, 6249651, 55095657, 
    58484771, 193834360, 131378779
])
dynamic_class = [0, 1, 3, 4, 5, 7, 9, 10]


def load_depth(img_file_path, gt_path):
    file_name = os.path.split(img_file_path)[-1]
    cam_depth = np.fromfile(os.path.join(gt_path, f'{file_name}.bin'),
        dtype=np.float32,
        count=-1).reshape(-1, 3)
    
    coords = cam_depth[:, :2].astype(np.int16)
    depth_label = cam_depth[:,2]
    return coords, depth_label

def load_seg_label(img_file_path, gt_path, img_size=[900,1600], mode='lidarseg'):
    if mode=='lidarseg':  # proj lidarseg to img
        coor, seg_label = load_depth(img_file_path, gt_path)
        seg_map = np.zeros(img_size)
        seg_map[coor[:, 1],coor[:, 0]] = seg_label
    else:
        file_name = os.path.join(gt_path, f'{os.path.split(img_file_path)[-1]}.npy')
        seg_map = np.load(file_name)
    return seg_map

def get_sensor_transforms(cam_info, cam_name):
    w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
    # sweep sensor to sweep ego
    sensor2ego_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    sensor2ego_tran = torch.Tensor(
        cam_info['cams'][cam_name]['sensor2ego_translation'])
    sensor2ego = sensor2ego_rot.new_zeros((4, 4))
    sensor2ego[3, 3] = 1
    sensor2ego[:3, :3] = sensor2ego_rot
    sensor2ego[:3, -1] = sensor2ego_tran
    # sweep ego to global
    w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
    ego2global_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    ego2global_tran = torch.Tensor(
        cam_info['cams'][cam_name]['ego2global_translation'])
    ego2global = ego2global_rot.new_zeros((4, 4))
    ego2global[3, 3] = 1
    ego2global[:3, :3] = ego2global_rot
    ego2global[:3, -1] = ego2global_tran

    return sensor2ego, ego2global


@DATASETS.register_module()
class NuScenesDataset3DGS(NuScenesDataset):
    def __init__(self, 
                use_rays=False,
                use_camera=False,
                semantic_gt_path=None,
                depth_gt_path=None,
                aux_frames=[-1,1],
                max_ray_nums=0,
                wrs_use_batch=False,
                znear=0.01, 
                zfar=40,
                render_img_shape=(225, 400),
                use_sam=False,
                gen_sam=False,
                **kwargs):
        super().__init__(**kwargs)
        self.gen_sam = gen_sam
        self.use_rays = use_rays
        self.use_camera = use_camera
        self.semantic_gt_path = semantic_gt_path
        self.depth_gt_path = depth_gt_path
        self.aux_frames = aux_frames
        self.max_ray_nums = max_ray_nums
        self.znear=znear
        self.zfar = zfar
        self.use_sam = use_sam
        if wrs_use_batch:   # compute with batch data
            self.WRS_balance_weight = None
        else:               # compute with total dataset
            self.WRS_balance_weight = torch.exp(0.005 * (nusc_class_nums.max() / nusc_class_nums - 1))

        self.dynamic_class = torch.tensor(dynamic_class)
        # print("render_img_shape", render_img_shape)
        self.render_image_height = render_img_shape[0]
        self.render_image_width = render_img_shape[1]  
        if gen_sam:
            sam = sam_model_registry["vit_h"]("ckpts/sam_vit_h_4b8939.pth").to('cuda')
            self.SAM_encoder = SamPredictor(sam)
            self.SAM_decoder = SamAutomaticMaskGenerator(
            sam, 
            pred_iou_thresh = 0.88, 
            stability_score_thresh = 0.95, 
            min_mask_region_area = 0)

    def get_rays(self, index):
        info = self.data_infos[index]

        sensor2egos = []
        ego2globals = []
        intrins = []
        coors = []
        label_depths = []
        label_segs = []
        time_ids = {}
        idx = 0

        for time_id in [0] + self.aux_frames:
            time_ids[time_id] = []
            select_id = max(index + time_id, 0)
            if select_id>=len(self.data_infos) or self.data_infos[select_id]['scene_token'] != info['scene_token']:
                select_id = index  # out of sequence
            info = self.data_infos[select_id]

            for cam_name in info['cams'].keys():
                intrin = torch.Tensor(info['cams'][cam_name]['cam_intrinsic'])
                sensor2ego, ego2global = get_sensor_transforms(info, cam_name)
                img_file_path = info['cams'][cam_name]['data_path']

                # load seg/depth GT of rays
                seg_map = load_seg_label(img_file_path, self.semantic_gt_path)
                coor, label_depth = load_depth(img_file_path, self.depth_gt_path)
                label_seg = seg_map[coor[:,1], coor[:,0]]

                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                intrins.append(intrin)
                coors.append(torch.Tensor(coor))
                label_depths.append(torch.Tensor(label_depth))
                label_segs.append(torch.Tensor(label_seg))
                time_ids[time_id].append(idx)
                idx += 1
        
        T, N = len(self.aux_frames)+1, len(info['cams'].keys())
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        sensor2egos = sensor2egos.view(T, N, 4, 4)
        ego2globals = ego2globals.view(T, N, 4, 4)

        # calculate the transformation from adjacent_sensor to key_ego
        keyego2global = ego2globals[0, :,  ...].unsqueeze(0)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        sensor2keyegos = sensor2keyegos.view(T*N, 4, 4)

        # generate rays for all frames
        rays = generate_rays(
            coors, label_depths, label_segs, sensor2keyegos, intrins,
            max_ray_nums=self.max_ray_nums, 
            time_ids=time_ids, 
            dynamic_class=self.dynamic_class, 
            balance_weight=self.WRS_balance_weight)
        return rays


    def focal2fov(self, focal, pixels):
        return 2*math.atan(pixels/(2*focal))

    def getWorld2View2(self, R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
        Rt = torch.zeros((4, 4))
        # Rt[:3, :3] = R.transpose()
        Rt[:3, :3] = torch.transpose(R, 0, 1)
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = torch.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = torch.linalg.inv(C2W)
        return Rt.float()

    def getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def cameras(self, time_ids, sensor2keyegos, intrins, label_segs, label_depths, label_masks, SAM_embs, SAM_masks):
        FoVx_lst = []
        FoVy_lst = []
        world_view_transform_lst = []
        full_proj_transform_lst = []
        camera_center_lst = []

        
        for time_id in time_ids:    # multi frames
            for i in time_ids[time_id]:    # multi cameras of single frame
                c2w = sensor2keyegos[i]
                intrin = intrins[i]
                
                focal_length_x, focal_length_y = intrin[0][0], intrin[1][1]
                FoVy = self.focal2fov(focal_length_y, self.render_image_height)
                FoVx = self.focal2fov(focal_length_x, self.render_image_width)      

                world_view_transform = torch.inverse(c2w).transpose(0, 1)
                projection_matrix = self.getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FoVx, fovY=FoVy).transpose(0,1)
                full_proj_transform = torch.bmm(world_view_transform.unsqueeze(0), projection_matrix.unsqueeze(0)).squeeze(0)
                camera_center = c2w[:3, 3]
                FoVx_lst.append(FoVx)       
                FoVy_lst.append(FoVy)       
                world_view_transform_lst.append(world_view_transform)      
                full_proj_transform_lst.append(full_proj_transform)      
                camera_center_lst.append(camera_center)    
               
        FoVx = torch.Tensor(FoVx_lst)
        FoVy = torch.Tensor(FoVy_lst)
        world_view_transform = torch.stack(world_view_transform_lst)
        full_proj_transform = torch.stack(full_proj_transform_lst)
        camera_center = torch.stack(camera_center_lst)
        label_segs = torch.stack(label_segs)
        label_masks = torch.stack(label_masks)
        # label_depths = torch.stack(label_depths)
        SAM_embs = mmcv.parallel.DataContainer(SAM_embs)
        SAM_masks = mmcv.parallel.DataContainer(SAM_masks)
        return (FoVx, FoVy, world_view_transform, full_proj_transform, camera_center, SAM_embs, SAM_masks, label_segs, label_masks)
        
        
    def get_viewpoints(self, index):
        info = self.data_infos[index]
        time_ids = {}
        sensor2egos = []
        ego2globals = []
        intrins = []
        coors = []
        label_depths = []
        label_segs = []
        label_masks = []
        SAM_embs = []
        SAM_masks = []
        idx = 0
        
        for time_id in [0] + self.aux_frames:
            time_ids[time_id] = []
            select_id = max(index + time_id, 0)
            if select_id>=len(self.data_infos) or self.data_infos[select_id]['scene_token'] != info['scene_token']:
                select_id = index  # out of sequence
            info = self.data_infos[select_id]
            for cam_name in info['cams'].keys():
                intrin = torch.Tensor(info['cams'][cam_name]['cam_intrinsic'])
                sensor2ego, ego2global = get_sensor_transforms(info, cam_name)
                img_file_path = info['cams'][cam_name]['data_path']

                # load seg/depth GT of rays
                seg_map = load_seg_label(img_file_path, self.semantic_gt_path)
                if self.use_sam:
                    SAM_f_path = img_file_path.replace("samples", "SAM_embeddings").replace(".jpg", "_fmap_CxHxW.pt")
                    SAM_emb = torch.load(SAM_f_path)
                    # SAM_emb = SAM_emb.permute(1,2,0)
                else:
                    SAM_emb=torch.zeros((1))

                if self.gen_sam:
                    v_img = cv2.imread(img_file_path)
                    img4feature = cv2.resize(v_img, dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
                    with torch.no_grad():
                        self.SAM_encoder.set_image(img4feature)
                        SAM_emb = self.SAM_encoder.features
                        sam_masks = self.SAM_decoder.generate(v_img)
                        mask_list = []
                        for m in sam_masks:
                            m_score = torch.from_numpy(m['segmentation']).float()[None, None, :, :].to('cuda')
                            m_score = torch.nn.functional.interpolate(m_score, size=(200,200) , mode='bilinear', align_corners=False).squeeze()
                            m_score[m_score >= 0.5] = 1
                            m_score[m_score != 1] = 0
                            mask_list.append(m_score)
                        SAM_mask = torch.stack(mask_list, dim=0)
                        # if len(SAM_mask.shape) < 3:
                        #     SAM_mask = SAM_mask.unsqueeze(0)
                else:
                    SAM_emb=torch.zeros((1))
                    SAM_mask=torch.zeros((1))
                coor, label_depth = load_depth(img_file_path, self.depth_gt_path)
                mask = np.zeros_like(seg_map)
                mask[coor[:,1], coor[:,0]] = 1
                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                intrins.append(intrin)
                coors.append(torch.Tensor(coor))
                label_depths.append(torch.Tensor(label_depth))
                label_segs.append(torch.Tensor(seg_map))
                label_masks.append(torch.Tensor(mask))
                SAM_embs.append(SAM_emb)
                SAM_masks.append(SAM_mask)
                time_ids[time_id].append(idx)
                idx += 1
        T, N = len(self.aux_frames)+1, len(info['cams'].keys())  # number of frame and cameras
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        sensor2egos = sensor2egos.view(T, N, 4, 4)
        ego2globals = ego2globals.view(T, N, 4, 4)

        # calculate the transformation from adjacent_sensor to key_ego
        keyego2global = ego2globals[0, :,  ...].unsqueeze(0)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()  # as for sensor2egos[0, :, ...]
        sensor2keyegos = sensor2keyegos.float()
        sensor2keyegos = sensor2keyegos.view(T*N, 4, 4)
        cameras = self.cameras(time_ids, sensor2keyegos, intrins, label_segs, label_depths, label_masks, SAM_embs, SAM_masks)
        return cameras

    def get_data_info(self, index):
        input_dict = super(NuScenesDataset3DGS, self).get_data_info(index)
        input_dict['with_gt'] = self.data_infos[index]['with_gt'] if 'with_gt' in self.data_infos[index] else True
        if 'occ_path' in self.data_infos[index]:
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        # generate rays for rendering supervision
        if self.use_rays:
            rays_info = self.get_rays(index)
            input_dict['rays'] = rays_info
        else:
            input_dict['rays'] = torch.zeros((1))
        if self.use_camera:
            input_dict["camera_info"] = self.get_viewpoints(index)
        else:
            input_dict['camera_info'] = torch.zeros((1))
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)


        return self.occ_eval_metrics.count_miou()
