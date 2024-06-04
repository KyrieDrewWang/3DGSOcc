import torch.nn as nn
import torch

class silog_loss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus
    def forward(self, depth_est, depth_gt, weight):
        d_ = torch.log(depth_est) - torch.log(depth_gt)
        d = d_ * weight
        # print("(d ** 2).mean():", (d ** 2).mean())
        # print("d.mean() ** 2:", d.mean() ** 2)
        # print("(d ** 2).mean() - self.variance_focus * (d.mean() ** 2)", (d ** 2).mean() - self.variance_focus * (d.mean() ** 2))
        loss = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)+1e-7)
        return loss

# class silog_loss(nn.Module):
#     def __init__(self, variance_focus=0.85):
#         super(silog_loss, self).__init__()
#         self.variance_focus = variance_focus
#     def forward(self, depth_est, depth_gt):
#         d = torch.log(depth_est) - torch.log(depth_gt)
#         return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2))
