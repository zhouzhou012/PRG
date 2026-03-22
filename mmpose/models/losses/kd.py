import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class KDLoss(nn.Module):

    """ PyTorch version of KD for Pose """

    def __init__(self,
                 name,
                 use_this,
                 loss_weight=1.0,
                 ):
        super(KDLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.loss_weight = loss_weight

    def forward(self, pred, pred_t, beta, target_weight):
        ls_x, ls_y = pred
        lt_x, lt_y = pred_t

        lt_x = lt_x.detach()
        lt_y = lt_y.detach()

        num_joints = ls_x.size(1)
        loss = 0

        loss += (
            self.loss(ls_x, lt_x, beta, target_weight))
        loss += (
            self.loss(ls_y, lt_y, beta, target_weight))

        return loss / num_joints

    def loss(self, logit_s, logit_t, beta, weight):
        
        N = logit_s.shape[0]
        Bins = logit_s.shape[-1]

        if len(logit_s.shape) == 3:
            K = logit_s.shape[1]
            logit_s = logit_s.reshape(N * K, -1)
            logit_t = logit_t.reshape(N * K, -1)

        # N*W(H)
        s_i = self.log_softmax(logit_s * beta)
        t_i = F.softmax(logit_t * beta, dim=1)

        # kd
        loss_all = torch.sum(self.kl_loss(s_i, t_i), dim=1)
        loss_all = loss_all.reshape(N, K).sum(dim=1).mean()
        loss_all = self.loss_weight * loss_all

        return loss_all


@MODELS.register_module()
class JointsKLDLoss(nn.Module):
    def __init__(self,
                 name,
                 use_this,
                 loss_weight=1.0,
                 ):
        super(JointsKLDLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.loss_weight = loss_weight
        # self.softmax = F.Softmax(dim=1)
        # self.logsoftmax = nn.(dim=1)


    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        width = output.size(2)
        height = output.size(3)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        # [B, 4096]
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()


            heatmap_pred = F.log_softmax(heatmap_pred.mul(target_weight[:,idx].unsqueeze(1)), dim=1)
            heatmap_gt = F.softmax(heatmap_gt.mul(target_weight[:,idx].unsqueeze(1)), dim=1)

            loss += self.criterion(
             heatmap_pred, heatmap_gt
           )

        loss = torch.sum(torch.sum(loss, dim=1), dim=0)

        return loss / batch_size / (width * height)*self.loss_weight





