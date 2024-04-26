_base_ = ['./bevstereo-occ.py']
render_img_shape=(480, 800)
use_sam=False
use_sam_mask=False
model = dict(
    type='SplattingOcc',
    final_softplus=True,
    use_3d_loss=False,
    use_lss_depth_loss=False,
    gauss_head=dict(
        type='GausSplatingHead',
        point_cloud_range= [-40, -40, -1, 40, 40, 5.4],
        voxel_size=0.4,
        voxel_feature_dim=17,
        num_classes=18,
        gaussian_sem_weight=1.0,
        white_background=False,
        x_lim_num=200, 
        y_lim_num=200, 
        z_lim_num=16,
        render_img_shape=render_img_shape,
        use_sam=use_sam,
        use_sam_mask=use_sam_mask
    ),
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)

depth_gt_path = 'DATA/depth_gt'
semantic_gt_path = 'DATA/seg_gt_lidarseg'

data = dict(
    samples_per_gpu=4,  # with 8 GPU, Batch Size=16 
    workers_per_gpu=4,
    train=dict(
        use_rays=False,
        use_camera=True,
        depth_gt_path=depth_gt_path,
        semantic_gt_path=semantic_gt_path,
        aux_frames=[],
        # aux_frames=[-3,-2,-1,1,2,3],
        max_ray_nums=38400,
        znear=0.01, 
        zfar=100,
        render_img_shape=render_img_shape,
        use_sam=use_sam,
        use_sam_mask=use_sam_mask,
    )
)


runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict(
    interval=50,
)
