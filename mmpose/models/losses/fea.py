import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class FeaLoss(nn.Module):

    """PyTorch version of feature-based distillation
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        alpha_fea (float, optional): Weight of dis_loss. Defaults to 0.00007
    """
    def __init__(self,
                 name,
                 use_this,
                 student_channels,
                 teacher_channels,
                 alpha_fea=0.00007,
                 ):
        super(FeaLoss, self).__init__()
        self.alpha_fea = alpha_fea

        if teacher_channels != student_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """

        if self.align is not None:
            outs = self.align(preds_S)
        else:
            outs = preds_S

        loss = self.get_dis_loss(outs, preds_T)

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        dis_loss = loss_mse(preds_S, preds_T)/N*self.alpha_fea

        return dis_loss

@MODELS.register_module()
class CORAL_loss(nn.Module):
    def __init__(self,
                 name,
                 use_this,
                 alpha_fea=0.00007,
                 ):
        super(CORAL_loss, self).__init__()
        self.alpha_fea=alpha_fea
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self,source, target):
        source = self.gap1(source)
        source = source.view(source.size(0), -1)
        target = self.gap2(target)
        target = target.view(target.size(0), -1)
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4 * d * d)*self.alpha_fea

        return loss