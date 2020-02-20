import torchvision
from torch import nn
import torch

class ResNet(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes,
                train_layer2 = False):
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
