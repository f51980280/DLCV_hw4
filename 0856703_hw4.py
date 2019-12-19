import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os
import cv2
import random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from itertools import groupby
from pycocotools import mask as maskutil
from detectron2.data.datasets import register_coco_instances

setup_logger()

def custom_dataset(set_name, set_dir_json, set_dir):
    register_coco_instances(set_name, {}, set_dir_json, set_dir)
    dataset_metadata = MetadataCatalog.get(set_name)
    dataset_dicts = DatasetCatalog.get(set_name)

    return dataset_metadata, dataset_dicts

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

def write_for_json(predictor, test_dataset_dicts):
    import random
    import json

    data = []

    for d in (test_dataset_dicts):    
        #print(d)
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
      
        te = np.array((outputs['instances'].pred_masks).tolist())
        for i in range(len(outputs['instances'].pred_classes.tolist())):
            mask_ans = binary_mask_to_rle(te[i])
            data.append ( {
        		'image_id' : d['image_id'],
        		'score' :float(outputs['instances'].scores[i]),
        		'category_id' : int((outputs['instances'].pred_classes[i]))+1,
        		'segmentation' : mask_ans
   		        }
    		)
    with open('0856703_5.json','w') as f:
        json.dump(data,f)

def train(set_name, model_frame, weights_source, iterations):
    cfg = get_cfg()
    cfg.merge_from_file(model_frame)
    cfg.DATASETS.TRAIN = (set_name,)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = weights_source
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = iterations   
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # 

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return cfg

if __name__ == '__main__':
    
    train_dataset_metadata, train_dataset_dicts = custom_dataset("train_dataset", "pascal_train.json", "train_images")
    test_dataset_metadata, test_dataset_dicts = custom_dataset("test_dataset", "test.json", "test_images")
        
    model_frame = "/root/wei/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    model_weights = "detectron2://ImageNetPretrained/MSRA/R-50.pkl" 

    cfg = train("train_dataset", model_frame, model_weights, 50000)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   
    cfg.DATASETS.TEST = ("test_dataset", )
    predictor = DefaultPredictor(cfg)

    write_for_json(predictor, test_dataset_dicts)