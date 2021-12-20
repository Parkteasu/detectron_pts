#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install opencv-python


# In[2]:


import os


# In[3]:


import glob


# In[4]:


print(os.getcwd())


# In[5]:


os.chdir("/Users/tspark/detectron2_2/detectron2_repo")
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# In[6]:


os.chdir("/Users/tspark/detectron2_2")


# In[7]:


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("spark",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (person)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
cfg.MODEL.WEIGHTS = "C:/Users/tspark/detectron2_2/output/model_final.pth" # 여기부분은 본인의 model이저장된 경로로 수정해줍니다.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("spark", )
predictor = DefaultPredictor(cfg)


# In[10]:


path = "C:/Users/tspark/detectron2_2/data/test_images"
i=0
for d in glob.glob(path+'/*.jpg'):
    i=i+1
    j=str(i)
    im = cv2.imread(d)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.9)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('result',v.get_image()[:, :, ::-1])
#     cv2.imwrite('/data/test_images/result.jpg',v.get_image()[:, :, ::-1])
    cv2.imwrite('result'+j+'.jpg',v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:




