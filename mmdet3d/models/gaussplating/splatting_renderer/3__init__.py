#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from torch import nn

def render_feature_map(
    viewpoint_camera:list,
    voxel_xyz:torch.tensor,
    opacity:torch.tensor,
    # depth:torch.tensor,
    scaling:torch.tensor,
    rotations:torch.tensor,
    voxel_features:torch.tensor,
    active_sh_degree=0,
    feature_dim=17,
    white_background = False,
    scaling_modifier = 1.0,
    debug = False,
    render_image_height = 900,
    render_image_width = 1600,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    background_color = torch.ones([feature_dim], dtype=torch.float32) if white_background else torch.zeros([feature_dim], dtype=torch.float32)
    background_color = background_color.to(voxel_features)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(voxel_xyz, dtype=voxel_xyz.dtype, requires_grad=True) + 0
    screenspace_points = screenspace_points.to(voxel_features)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera[0] * 0.5)
    tanfovy = math.tan(viewpoint_camera[1] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=render_image_height,
        image_width=render_image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera[2],
        projmatrix=viewpoint_camera[3],
        sh_degree=active_sh_degree, # 0 -- > 1 -- > 2 -- > 3 球谐函数的次数, 最开始是0, 每隔1000次迭代, 将球谐函数的次数增加1，此处采用默认为0
        campos=viewpoint_camera[4],
        prefiltered=False,
        debug=debug
    )
    depth_feature = torch.mm(torch.cat([voxel_xyz, torch.ones((voxel_xyz.shape[0]), 1).to(voxel_xyz)], -1), viewpoint_camera[2])[:, 2].unsqueeze(1).unsqueeze(1)  # p_view.z
    depth_feature = depth_feature / torch.abs(depth_feature).max() # normalize p_view.z
    depth_feature.requires_grad_(False)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = voxel_xyz   # pc position
    means2D = screenspace_points # the shape is the same as means3D
    opacity = opacity  # 不透明度 the shape is the same as means3D   
    scales = scaling  # 尺度
    rotations = rotations  # 旋转参数
    # depth_feature = depth.unsqueeze(1)
    rendered_image, depth_feature, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,  # None
        colors_precomp = voxel_features,  # feature map
        semantic_feature = depth_feature, 
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)
    # print("depth_feature", depth_feature)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii,
    #         'depth_feature': depth_feature}
    return rendered_image, depth_feature
