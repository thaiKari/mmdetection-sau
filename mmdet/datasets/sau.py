from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class SauDataset(CocoDataset):

    CLASSES = ('sheep')
    
    def _parse_ann_info(self, img_info, ann_info):
        if 'bbox' in ann_info[0].keys():
            return super()._parse_ann_info(img_info, ann_info)
        
        else: #grid only
            gt_labels = []
            gt_grid_mask = []

            for i, ann in enumerate(ann_info):
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_grid_mask.append(ann['grid_mask'])


            if not gt_labels:
                gt_labels = np.array([], dtype=np.int64)



            ann = dict(
                labels=gt_labels,
                grid_mask=gt_grid_mask
                )
            
            return ann