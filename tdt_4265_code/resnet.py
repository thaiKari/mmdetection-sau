import torchvision
from torch import nn

class ResNet(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.model = torchvision.models.resnet18( pretrained = True )
        self.model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.model.fc = nn.Linear( 512 , num_classes ) # No need to apply softmax ,
        # as this is done in nn. C r o s s E n t r o p y L o s s
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully - connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers
        #for param in self.model.layer3.parameters(): # Unfreeze the last 5 convolutional
        #    param.requires_grad = True # layers
        #for param in self.model.layer2.parameters(): # Unfreeze the last 5 convolutional
        #    param.requires_grad = True # layers
        
        
    def forward ( self , x):
        #x = nn.functional.interpolate(x , scale_factor =8)
        x = self.model(x)
        return x