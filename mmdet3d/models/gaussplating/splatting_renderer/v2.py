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
    scaling:torch.tensor,
    rotations:torch.tensor,
    voxel_features:torch.tensor,
    active_sh_degree=0,
    feature_dim=32,
    white_background = False,
    scaling_modifier = 1.0,
    debug = True,
    override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    background_color = torch.ones(feature_dim, dtype=torch.float32) if white_background else torch.zeros(feature_dim, dtype=torch.float32)
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
        image_height=int(viewpoint_camera[2]),
        image_width=int(viewpoint_camera[3]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera[4],
        projmatrix=viewpoint_camera[5],
        sh_degree=active_sh_degree, # 0 -- > 1 -- > 2 -- > 3 球谐函数的次数, 最开始是0, 每隔1000次迭代, 将球谐函数的次数增加1，此处采用默认为0
        campos=viewpoint_camera[6],
        prefiltered=False,
        debug=debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = voxel_xyz   # pc position
    means2D = screenspace_points # the shape is the same as means3D
    opacity = opacity  # 不透明度 the shape is the same as means3D 

    scales = scaling  # 尺度
    rotations = rotations  # 旋转参数

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = torch.zeros((voxel_features.shape[0], 16, 3)).to(voxel_features)
    colors_precomp = None
    semantic_feature = voxel_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, feature_map, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        semantic_feature = semantic_feature.unsqueeze(1), 
        opacities = opacity,  # 不透明度
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'feature_map': feature_map}

