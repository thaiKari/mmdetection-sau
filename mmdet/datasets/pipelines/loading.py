import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False, type='rgb'):
        self.to_float32 = to_float32

    def __call__(self, results):
        print('img_prefix', results['img_prefix'])
        
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
            
        
        
        if '.npy' in filename[-4:]:
            img = mmcv.imread(np.load(filename))
            
        else:
            img = mmcv.imread(filename)
        
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadFusionImageFromFile(object):

    def __init__(self, to_float32=False, type='rgb'):
        self.to_float32 = to_float32

    def __call__(self, results):
        print('img_prefix', results['img_prefix'])
        

        filename_infrared = osp.join(results['img_prefix'], 'infrared',
                            results['img_info']['filename'])
        filename_rgb = osp.join(results['img_prefix'], 'rgb',
                            results['img_info']['filename'])

            
        img_infrared = mmcv.imread(filename_infrared)
        img_rgb = mmcv.imread(filename_rgb)
        
        if self.to_float32:
            img_rgb = img_rgb.astype(np.float32)
            img_infrared = img_infrared.astype(np.float32)

        results['filename'] = filename_rgb
        results['img'] = {'infrared': img_infrared,
                          'rgb': img_rgb
                            }
        results['img_shape'] = img_rgb.shape
        results['ori_shape'] = img_rgb.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        if 'bboxes' in ann_info.keys():
            results['gt_bboxes'] = ann_info['bboxes']
        else:
            results['gt_bboxes'] = ann_info
            
        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

        
    def _load_labels(self, results):
        if 'labels' in results['ann_info'].keys():
            results['gt_labels'] = results['ann_info']['labels']
            
        else:
            results['gt_labels'] = [0]*len(results['ann_info'])
        
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    

    
    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)    

        #print('==========')
        #print('results', results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
