import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
import torchvision
from torch import nn
import torch
import os

class ResNet(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes,
                train_layer2 = True):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.model = torchvision.models.resnet18( pretrained = True )
        
        #Change model to fit number of input channels
        if image_channels != 3:
            self.model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)

        
        
        self.model.fc = nn.Linear( 512 , num_classes ) 
        
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False            
        
        for param in self.model.fc.parameters(): # Unfreeze the last fully - connected
            param.requires_grad = True 
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True
        for param in self.model.layer3.parameters(): 
            param.requires_grad = True 
        
        if train_layer2:
            for param in self.model.layer2.parameters(): 
                param.requires_grad = True 
        
        
        
    def forward ( self , x):
        #x = nn.functional.interpolate(x , scale_factor =8)

        x = self.model(x)
        return x
   
    
class ResNetEnsembleInfraredRGB(nn.Module):

    def __init__(self, num_classes, ResNetRGB, ResNetIR, train_layer2 = True):

        super().__init__()

        #self.ResNetRGB = ResNetRGB
        #self.ResNetIR = ResNetIR

        
        for param in ResNetRGB.parameters(): # Freeze all parameters
            param.requires_grad = False  
        
        for param in ResNetRGB.model.layer4.parameters():
            param.requires_grad = True
        
        for param in ResNetRGB.model.layer3.parameters():
            param.requires_grad = True

        if train_layer2:
            for param in ResNetRGB.model.layer2.parameters(): 
                param.requires_grad = True
        
        for param in ResNetIR.parameters(): # Freeze all parameters
            param.requires_grad = False  
        
        
        for param in ResNetIR.model.layer3.parameters():
            param.requires_grad = True
            
        if train_layer2:
            for param in ResNetIR.model.layer2.parameters(): 
                param.requires_grad = True

        
        self.rgb_base = nn.Sequential(
            ResNetRGB.model.conv1,
            ResNetRGB.model.bn1,
            ResNetRGB.model.relu,
            ResNetRGB.model.maxpool,
            ResNetRGB.model.layer1,
            ResNetRGB.model.layer2,
            ResNetRGB.model.layer3
        )
        
        self.infrared_base = nn.Sequential(
            ResNetIR.model.conv1,
            ResNetIR.model.bn1,
            ResNetIR.model.relu,
            ResNetIR.model.maxpool,
            ResNetIR.model.layer1,
            ResNetIR.model.layer2,
            ResNetIR.model.layer3
        )
        
            
        self.upsample = nn.Sequential( #Double size 3 times
                                        nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
                                    )

        
        #After concat of layers from infrared and rgb. do 1x1 conv layer to reduce dimensions.
        self.dimension_reducer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0,bias=False)
        
        self.model_Head = nn.Sequential(                                    
                                    ResNetRGB.model.layer4,
                                    ResNetRGB.model.avgpool,
                                    nn.Flatten(),
                                    ResNetRGB.model.fc 
                                    )      


        #self.fc = nn.Linear( 18 , 9 )
    
        #print(len(list(self.ResNetRGB.children())) )
    
    def forward ( self , x_rgb, x_infrared): 
        #x_rgb = self.ResNetRGB(x_rgb)
        #x_infrared = self.ResNetIR(x_infrared)
        #x = torch.cat((x_rgb, x_infrared),dim=1)
        #x = self.fc(x)
        
        #return x

         #rgb_base
        x_rgb = self.rgb_base(x_rgb)

        #infrared_base
        x_infrared = self.infrared_base(x_infrared)
        
        # give outputs the same dimension... scale up infrared or downscale rgb
        x_infrared= self.upsample(x_infrared)        
        
        x = torch.cat((x_rgb, x_infrared), dim=1)
        x = self.dimension_reducer(x)
        x = self.model_Head(x)


        return x

@BACKBONES.register_module
class ResNetFUSION(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """


    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 model_path_fusion = './work_dirs/work_dirs_external/ensemble/20200219_1221/model_best.pth.tar',
                 model_path_rgb = './work_dirs/work_dirs_external/rgb/20200218_1940/model_best.pth.tar',
                 model_path_infrared = './work_dirs/work_dirs_external/infrared/20200218_1929/model_best.pth.tar'
                ):
        
        super(ResNetFUSION, self).__init__()
        
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        
        model_rgb = ResNet(image_channels=3, num_classes=9)
        model_rgb.load_state_dict(torch.load(model_path_rgb)['state_dict'])

        model_infrared = ResNet(image_channels=3, num_classes=9)
        model_infrared.load_state_dict(torch.load(model_path_infrared)['state_dict'])
        
        model = ResNetEnsembleInfraredRGB(num_classes=9, ResNetRGB=model_rgb, ResNetIR=model_infrared)
        pretrained_dict = torch.load(model_path_fusion)['state_dict']
        model.load_state_dict(pretrained_dict, strict=True)  
        
        self.model = model

    def init_weights(self, pretrained=None):
        print('init_weights does nothing right now')
            
    def forward(self, x):
        print('FORWARD')
        print(x['rgb'].shape)
        print(x['infrared'].shape)
        outs = []
        x_rgb = self.model.rgb_base[:-2](x['rgb'])
        outs.append(x_rgb)
        x_rgb = self.model.rgb_base[-2](x[x_rgb])
        outs.append(x_rgb)
        x_rgb = self.model.rgb_base[-1](x[x_rgb])
        outs.append(x_rgb)
        
        
        x_infrared = self.model.infrared_base(x['infrared'])
        x_infrared = self.model.upsample(x_infrared)
        
        x = torch.cat((x_rgb, x_infrared), dim=1)
        x = self.model.dimension_reducer(x)
        outs.append(x)
        
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(ResNetFUSION, self).train(mode)

