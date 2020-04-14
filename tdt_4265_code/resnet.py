import torchvision
from torch import nn
import torch

layer_depths_dict = {
    18: (64,128,256,512),
    34: (64,128,256,512),
    50: (256, 512, 1024, 2048),
    101: (256, 512, 1024, 2048),
    152: (256, 512, 1024, 2048)
}


class ResNet(nn.Module):

    def __init__(self,
                 image_channels=3,
                 num_classes=9,
                train_layer2 = True,
                depth = 18,
                X=False, # if X=True, use resNext not resNet
                test_mode= False):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        
        assert depth in [18, 34, 50, 101, 152], 'ResNet only available for depth 18, 34, 50, 101 and 152'
        
        if X:
            assert depth in [50, 101], 'ResNext only available for depth 50 and 101'
        
        if depth == 18:
            self.model = torchvision.models.resnet18( pretrained = True )
        if depth == 34:
            self.model = torchvision.models.resnet34( pretrained = True )
        if depth == 50:
            if X:
                self.model = torchvision.models.resnext50_32x4d( pretrained = True )
            else:
                self.model = torchvision.models.resnet50( pretrained = True )

        if depth == 101:
            if X:
                self.model = torchvision.models.resnext101_32x8d( pretrained = True )
            else:
                self.model = torchvision.models.resnet101( pretrained = True )
        if depth == 152:
            self.model = torchvision.models.resnet152( pretrained = True )
        
        layer_depths = layer_depths_dict[depth]
        
        #Change model to fit number of input channels
        if image_channels != 3:
            self.model.conv1 = nn.Conv2d(image_channels, layer_depths[0], kernel_size=7, stride=2, padding=3,bias=False)

        
        self.model.fc = nn.Linear( layer_depths[-1] , num_classes ) 
        
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False            
        
        if not test_mode:
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

    def __init__(self,
                 num_classes,
                 ResNetRGB,
                 ResNetIR,
                 train_layer2 = True,
                 rgb_size=1280,
                 infrared_size=160,
                 fuse_after_layer = 3,
                 depth = 18,
                test_mode=False):

        
        super().__init__()
        self.fuse_after_layer = fuse_after_layer
        self.rgb_size = rgb_size
        self.infrared_size =infrared_size
        
        self.layer_depths = layer_depths_dict[depth]
        
        if fuse_after_layer < 5:
            self.fuse_depth = self.layer_depths[fuse_after_layer-1]
        else:
            self.fuse_depth = 256
        
        
        for param in ResNetRGB.parameters(): # Freeze all parameters
            param.requires_grad = False  
        
        if not test_mode:
            for param in ResNetRGB.model.layer4.parameters():
                param.requires_grad = True

            for param in ResNetRGB.model.layer3.parameters():
                param.requires_grad = True

            if train_layer2:
                for param in ResNetRGB.model.layer2.parameters(): 
                    param.requires_grad = True
        
        for param in ResNetIR.parameters(): # Freeze all parameters
            param.requires_grad = False  
        
        if not test_mode:
            if fuse_after_layer >= 3:
                for param in ResNetIR.model.layer3.parameters():
                    param.requires_grad = True


            if fuse_after_layer >= 4:
                for param in ResNetIR.model.layer4.parameters():
                    param.requires_grad = True

            if train_layer2:
                for param in ResNetIR.model.layer2.parameters(): 
                    param.requires_grad = True

        
        self.rgb1 = nn.Sequential(
            ResNetRGB.model.conv1,
            ResNetRGB.model.bn1,
            ResNetRGB.model.relu,
            ResNetRGB.model.maxpool,
            ResNetRGB.model.layer1,
        )
        
            
        self.rgb2 = ResNetRGB.model.layer2        
        self.rgb3 = ResNetRGB.model.layer3
        self.rgb4 = ResNetRGB.model.layer4
        
        self.infrared1 = nn.Sequential(
            ResNetIR.model.conv1,
            ResNetIR.model.bn1,
            ResNetIR.model.relu,
            ResNetIR.model.maxpool,
            ResNetIR.model.layer1,
        )
        
        self.infrared2 = ResNetIR.model.layer2
        self.infrared3 = ResNetIR.model.layer3
        self.infrared4 = ResNetIR.model.layer4
        
        if int(rgb_size/infrared_size) == 2**4:
            self.upsample = nn.Sequential( #Double size 4 times
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1),
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1),
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1),
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1),
                                        )
        
        if int(rgb_size/infrared_size) == 2**3:
            self.upsample = nn.Sequential( #Double size 3 times
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1),
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1),
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1),
                                        )
        elif int(rgb_size/infrared_size) == 2**2:
            self.upsample = nn.Sequential( #Double size twice
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1),
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1)
                                        )
        elif int(rgb_size/infrared_size) == 2**1:
            self.upsample = nn.Sequential( #Double once
                                  nn.ConvTranspose2d(self.fuse_depth, self.fuse_depth, 3, stride=2, padding=1, output_padding=1),
                                        )
        
        #After concat of layers from infrared and rgb. do 1x1 conv layer to reduce dimensions.
        self.dimension_reducer = nn.Conv2d(self.fuse_depth*2, self.fuse_depth, kernel_size=1, stride=1, padding=0,bias=False)
        
        self.model_Head_rgb = nn.Sequential(                                    
                                    ResNetRGB.model.avgpool,
                                    nn.Flatten(),
                                    ResNetRGB.model.fc 
                                    )
        self.model_Head_infrared = nn.Sequential(                                    
                                    ResNetIR.model.avgpool,
                                    nn.Flatten(),
                                    ResNetIR.model.fc 
                                    )

        self.extra_fc = nn.Linear(in_features=18, out_features=9, bias=True)


    
    def forward ( self , x_rgb, x_infrared): 

        #rgb_base
        x_rgb = self.rgb1(x_rgb)
        x_rgb = self.rgb2(x_rgb)

        #infrared_base
        x_infrared = self.infrared1(x_infrared)
        x_infrared = self.infrared2(x_infrared)
        
        if self.fuse_after_layer == 2:
            if int(self.rgb_size/self.infrared_size) > 1:
                x_infrared= self.upsample(x_infrared)
            x = torch.cat((x_rgb, x_infrared), dim=1)
            x = self.dimension_reducer(x)
            x = self.rgb3(x)
            x = self.rgb4(x)
            x = self.model_Head_rgb(x)
            
            return x
        
        x_rgb = self.rgb3(x_rgb)
        x_infrared = self.infrared3(x_infrared)
        
        if  self.fuse_after_layer == 3:
            if int(self.rgb_size/self.infrared_size) > 1:
                x_infrared= self.upsample(x_infrared)
            x = torch.cat((x_rgb, x_infrared), dim=1)
            x = self.dimension_reducer(x)
            x = self.rgb4(x)
            x = self.model_Head_rgb(x)
            
            return x
        
        x_rgb = self.rgb4(x_rgb)
        x_infrared = self.infrared4(x_infrared)
        
        if  self.fuse_after_layer == 4:
            if int(self.rgb_size/self.infrared_size) > 1:
                x_infrared= self.upsample(x_infrared)
            x = torch.cat((x_rgb, x_infrared), dim=1)
            x = self.dimension_reducer(x)
            x = self.model_Head_rgb(x)
            
            return x
        
        x_rgb = self.model_Head_rgb(x_rgb)
        x_infrared = self.model_Head_infrared(x_infrared)
        
        x = torch.cat((x_rgb, x_infrared), dim=1)
        x = self.extra_fc(x)
        
        return x
    

    
class ResNetEnsembleInfraredRGBOld(nn.Module):

    def __init__(self, num_classes, ResNetRGB, ResNetIR, train_layer2 = False):

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
