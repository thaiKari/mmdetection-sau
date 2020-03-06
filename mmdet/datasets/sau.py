from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class SauDataset(CocoDataset):

    CLASSES = ('sheep')
    
    def _parse_ann_info(self, img_info, ann_info):        
        return super()._parse_ann_info(img_info, ann_info)
        
