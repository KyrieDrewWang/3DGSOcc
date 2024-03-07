import torch
from torch import nn
import numpy as np
import math

class Camera(nn.Module):
    def __init__(self,
                 R,
                 T,
                 FoVx,
                 FoVy,
                 image_h,
                 image_w,
                 trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0,
                 ) -> None:
        super().__init__()
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.img_h = image_h
        self.img_w = image_w
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(self.getWorld2View2(R, T, trans, scale)).transpose(0, 1) 
        self.projection_matrix = self.getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        
    def getWorld2View2(self, R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)
        
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