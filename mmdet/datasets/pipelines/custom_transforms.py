import inspect

import albumentations
import mmcv
import numpy as np
from albumentations import Compose
from imagecorruptions import corrupt
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES

@PIPELINES.register_module
class Flip(object):
    """
    Filp img and bboxes.
    Author: Zhu Feijia. zzzfinal@foxmail.com
    """
    
    def __init__(self, direction='r'):
        """
        :param direction:
            'h':horizontal
            'v':vertical
            'hv': h then v (180° rotated )
            'r': random choose from h,v,hv
            'n': no flip
        """
        self._directions = ['h', 'v', 'hv', 'n', 'r']
        assert direction in self._directions
        self.direction = direction
        print()

    def __call__(self, results): #img, bboxes, labels):
        """
        Filp img and bboxes.
        :param img: :An image.
        :param bboxes(ndarray): shape (..., 4*k): (xmin,ymin,xmax,ymax)
        :param labels: labels of bboxes
        :return: flipped img, flipped bboxes, labels
        """
        if self.direction == 'n':
            results['flip']=False
            return results
        if self.direction == 'h':
            results['img'] = np.flip(results['img'], axis=1)
        elif self.direction == 'v':
            results['img'] = np.flip(results['img'], axis=0)
        elif self.direction == 'hv':
            results['img'] = np.flip(results['img'], axis=1)
            results['img'] = np.flip(results['img'], axis=0)
        
        # flip bboxes
        for key in results.get('bbox_fields', []):
            results[key] = self.bbox_flip(results[key],
                                          results['img_shape'])
        # flip masks
        for key in results.get('mask_fields', []):
            results[key] = [mask[:, ::-1] for mask in results[key]]

        if self.direction == 'r':
            self.direction = random.choice(self._directions[:-1])
            results = self(results)
            self.direction = 'r'
        results['flip']=True
        return results

    def bbox_flip(self, bboxes, img_shape, direction=None):
        """
        Flip bboxes.
        :param bboxes: ndarray, shape (..., 4*k): (xmin,ymin,xmax,ymax)
        :param img_shape: img.shape
        :param direction:
                'h':horizontal
                'v':vertical
                'hv': h then v (180° rotated )
        :return: flipped bboxes
        """

        d = self.direction if direction is None else direction
        assert bboxes.shape[-1] % 4 == 0
        h, w = img_shape[0], img_shape[1]
        flipped = bboxes.copy()

        if d == 'h':
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        elif d == 'v':
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        elif d == 'hv':
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        return flipped


