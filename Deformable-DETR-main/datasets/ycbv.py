# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Face dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/Face_utils.py
"""

from pathlib import Path

import torch
from PIL import Image
import torch.utils.data
import random
import torchvision
from pycocotools import mask as Face_mask
import glob
import numpy as np
import datasets.transforms as T

backgrounds_path = glob.glob('./YCB_Video_Dataset/nvidia/nvidia4/*.png')
class FaceDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(FaceDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertFacePolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(FaceDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        path = self.coco.loadImgs(idx)[0]["file_name"]
        if 'syn' in path:
            postfix = path.split('-')[0].split('/')[-1]
            rndIdx = random.randint(0, len(backgrounds_path) - 1)
            bg = Image.open(backgrounds_path[rndIdx]).convert('RGB')
            mask = Image.open('./YCB_Video_Dataset/data_syn/' + postfix + '-label.png')
            mask = np.array(mask)
            mask[ mask>0] = 255
            mask = Image.fromarray(mask)
            img = Image.composite(img, bg, mask)
            bg.close()
            mask.close()
            # img.show()
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_Face_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = Face_mask.frPyObjects(polygons, height, width)
        mask = Face_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertFacePolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        RTs = [obj["RTs"] for obj in anno]
        RTs = torch.tensor(RTs, dtype=torch.float32)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_Face_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to Face api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
      
        target['RTs'] = RTs

        rel_position = [obj["relative_pose"]['position'] for obj in anno]
        rel_position = torch.tensor(rel_position, dtype=torch.float32)
        target["relative_position"] = rel_position

        rel_rotation = [obj["relative_pose"]['rotation'] for obj in anno]
        rel_rotation = torch.tensor(rel_rotation, dtype=torch.float32)
        target["relative_rotation"] = rel_rotation

        return image, target


def make_face_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    #root = Path(args.data_path)
    #assert root.exists(), f'provided Face path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": ( "./ycbv_BOP", './ycbv_BOP/annotations/train.json'),
        "test": ( "./ycbv_BOP", './ycbv_BOP/annotations/test.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = FaceDetection(img_folder, ann_file, transforms=make_face_transforms(image_set), return_masks=args.masks)
    return dataset