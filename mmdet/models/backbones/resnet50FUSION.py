import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer

import sys
sys.path.insert(1, './tdt_4265_code/')
from eval_utils import get_ensemble_model

import torch

root_work_dir = './work_dirs/work_dirs_external/'
work_dir = root_work_dir+ 'ensemble/20200411_1733'
time_stamp_rgb = '20200329_1239'
time_stamp_infrared = '20200330_0959'

@BACKBONES.register_module
class ResNetFUSION50(nn.Module):
    def __init__(self):
        super(ResNetFUSION50, self).__init__()
        
        self.model = get_ensemble_model(root_work_dir, time_stamp_rgb, time_stamp_infrared, 50, False, 1024, 64, 3)
        self.model = self.model.eval()
    
    def init_weights(self, pretrained=None):
        print('init_weights does nothing now')
    
    def forward(self, x):
        outs = []
        
        x_rgb = x['rgb']
        x_rgb = self.model.rgb1(x_rgb)
        outs.append(x_rgb)
        x_rgb = self.model.rgb2(x_rgb)
        outs.append(x_rgb)
        x_rgb = self.model.rgb3(x_rgb)
        outs.append(x_rgb)
        
        x_infrared = x['infrared']
        x_infrared = self.model.infrared1(x_infrared)
        x_infrared = self.model.infrared2(x_infrared)
        x_infrared = self.model.infrared3(x_infrared)
        x_infrared = self.model.upsample(x_infrared)


        x = torch.cat((x_rgb, x_infrared), dim=1)
        x = self.model.dimension_reducer(x)
        x = self.model.rgb4(x)
        outs.append(x)
        
        return tuple(outs)

    def train(self, mode=True):
        super(ResNetFUSION50, self).train(mode)
