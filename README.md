# DLCV_hw4
##  Semantic Segmentation

refrence from https://github.com/facebookresearch/detectron2
and git clone it in repository  

### follow https://github.com/facebookresearch/detectron2 for build detectron2  
env : ```!pip install -U torch torchvision cython
!pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```

check pascal.json and train_image dir weather correct  
then ```python 0856703_hw4.py```  

Model : MaskR-CNN  
Backbone : FPN from https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml  
Pretrain weight : "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  

