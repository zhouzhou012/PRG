import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from mmengine.model import BaseModule



class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, h, w = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap

@MODELS.register_module()  #相当于热图softmax得到概率分布，再用kl loss计算损失
class Channel_FeaLoss(nn.Module):

    def __init__(self,
                 name,
                 use_this,
                 student_channels=17,
                 teacher_channels=17,
                 norm_type='channel',
                 divergence='kl',
                 temperature=1.0,
                 loss_weight=0.00007,
                 ):

        super(Channel_FeaLoss, self).__init__()

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence
        self.loss_weight=loss_weight
        if teacher_channels != student_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None


    def forward(self, preds_S, preds_T):

        if self.align is not None:
            preds_S = self.align(preds_S)

        n, c, h, w = preds_S.shape
        # import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S/self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()

        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s, norm_t)

        # item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        # import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
        else:
            loss /= n * h * w

        return loss * (self.temperature ** 2)*self.loss_weight