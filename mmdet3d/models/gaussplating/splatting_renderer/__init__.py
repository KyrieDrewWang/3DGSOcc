import torch
import math
from diff_gaussian_rasterization_contrastive_f import GaussianRasterizationSettings as GaussianRasterizationSettingsContrastiveF
from diff_gaussian_rasterization_contrastive_f import GaussianRasterizer as GaussianRasterizerContrastiveF

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
    debug = False,
    
):
    """
    Render the input feature map. 
    
    Background tensor (bg_color) must be on GPU!
    """
    device = voxel_features.device
    
    background_color = torch.ones(feature_dim, dtype=torch.float32, device=device) if white_background else torch.zeros(feature_dim, dtype=torch.float32, device=device)
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(voxel_xyz, dtype=voxel_xyz.dtype, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera[0] * 0.5)
    tanfovy = math.tan(viewpoint_camera[1] * 0.5)

    raster_settings = GaussianRasterizationSettingsContrastiveF(
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

    rasterizer = GaussianRasterizerContrastiveF(raster_settings=raster_settings)

    means3D = voxel_xyz   # pc position
    means2D = screenspace_points # the shape is the same as means3D
    opacity = opacity  # 不透明度 the shape is the same as means3D   
    scales = scaling  # 尺度
    rotations = rotations  # 旋转参数

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,  # None
        colors_precomp = voxel_features,  # feature map
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    return {
        "render":rendered_image,
        "radii": radii,  # 每个2D gaussian在图像上的半径
        }
    