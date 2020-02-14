import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss



@LOSSES.register_module
class AlwaysZeroLoss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(AlwaysZeroLoss, self).__init__()


    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        return  pred*0
