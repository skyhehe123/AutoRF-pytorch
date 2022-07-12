import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import structures
from detectron2.projects import point_rend
coco_metadata = MetadataCatalog.get("coco_2017_val")

import numpy as np
import cv2


import kitti_util

cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("scripts/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
predictor = DefaultPredictor(cfg)

class Prepare(torch.utils.data.Dataset):

    def __init__(self, ):
        super().__init__()

        self.ids = range(
            len(os.listdir(
            '/data0/billyhe/KITTI/training/label_2'))
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        objs = kitti_util.read_label('/data0/billyhe/KITTI/training/label_2/%06d.txt' % id)
        img = cv2.imread('/data0/billyhe/KITTI/training/image_2/%06d.png' % id)
    
        insts = predictor(img)["instances"]
        insts = insts[insts.pred_classes == 2] # 2 for ca
        ious = structures.pairwise_iou(
            structures.Boxes(torch.Tensor([obj.box2d for obj in objs])).to(insts.pred_boxes.device),
            insts.pred_boxes
        )

        if ious.numel() == 0:
            return 1
        
        for i, obj in enumerate(objs):

            if obj.type == 'DontCare':
                continue
            if obj.t[2] > 50:
                continue
            if obj.ymax - obj.ymin < 64:
                continue
            iou, j = torch.max(ious[i]), torch.argmax(ious[i])
            if iou<.8:
                continue
            rgb_gt = img[int(obj.ymin):int(obj.ymax), int(obj.xmin):int(obj.xmax), :]
            msk_gt = insts.pred_masks[j][int(obj.ymin):int(obj.ymax), int(obj.xmin):int(obj.xmax)]

            cv2.imwrite('/data0/billyhe/KITTI/training/nerf/%06d_%02d_patch.png' % (id, i), rgb_gt)
            cv2.imwrite('/data0/billyhe/KITTI/training/nerf/%06d_%02d_mask.png' % (id, i), np.stack([msk_gt.cpu()*255]*3, -1))  
            anno = [obj.xmin, obj.xmax, obj.ymin, obj.ymax] + list(obj.t) + list(obj.dim) + [obj.ry]
            anno = [str(x) for x in anno]
            with open('/data0/billyhe/KITTI/training/nerf/%06d_%02d_label.txt' % (id, i), 'w') as f:
                f.writelines(' '.join(anno))
            
            
        return 1


if __name__ == "__main__":  


    loader = torch.utils.data.DataLoader(
            Prepare(),
            batch_size=1,
            shuffle=False,
            num_workers=0
    )

    for _ in loader:
        pass