from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose,
    Rotate,
    ReplayCompose,
    Normalize,
    OneOf,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    RandomSizedCrop
)

def get_aug(aug, min_area=0., min_visibility=0.):
    return ReplayCompose(aug, bbox_params=BboxParams(format='coco', min_area=min_area, 
                                               min_visibility=min_visibility, label_fields=['category_id']))

#Augmentations applied to both rgb and ir images
def common_augmentations(crop_shape = (1200, 1200) ):
     return get_aug([VerticalFlip(p=0.5),
                        HorizontalFlip(p=0.5),
                        Rotate(p=0.5,
                               limit=360),
                        RandomSizedCrop((1000,1400), *crop_shape), #Crop a random part of the input (cropsize: (800 to 1600)) and rescale it to (1200, 1200). 
                       ])

    
def rgb_augmentations_bare_bones(resize_shape=(1280,1280)):
    return  get_aug([Resize(*resize_shape ),
                     Normalize()
                     ])

    


def rgb_augmentations(resize_shape=(1280,1280)):
    return  get_aug([Resize(*resize_shape ),
                     OneOf(
                        [
                            # apply one of transforms to 50% of images
                            RandomContrast(), # apply random contrast
                            RandomGamma(), # apply random gamma
                            RandomBrightness(), # apply random brightness
                        ],
                        p = 0.5),
                     Normalize()
                     ])


def infrared_augmentations(resize_shape=(160,160)):
    return  get_aug([Resize(*resize_shape),
                     Normalize(mean= (0.2, 0.2, 0.2)),                     
                     ])

