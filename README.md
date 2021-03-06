# DLCV_hw4
##  Semantic Segmentation

refrence from https://github.com/facebookresearch/detectron2
and git clone it in repository  
```unzip 0856703_hw4``` can get my folder in this homework

### follow https://github.com/facebookresearch/detectron2 for build detectron2   
#### Installation   
Colab Notebook has step-by-step instructions that install detectron2. The Dockerfile also installs detectron2 with a few simple commands.  ```https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5```

#### Requirements  
Python >= 3.6  
PyTorch 1.3  
torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.  
OpenCV, needed by demo and visualization  
fvcore: ```pip install -U 'git+https://github.com/facebookresearch/fvcore'  
pycocotools: pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'  ```
GCC >= 4.9  

#### Build Detectron2  
After having the above dependencies, run:    
```git clone https://github.com/facebookresearch/detectron2.git  
cd detectron2  
python setup.py build develop 
``` 
#### Implement for training
then check 0856703_hw4.py file pascal.json and train_image dir weather correct  
then ```python 0856703_hw4.py```  start training

Model : MaskR-CNN  
Backbone : FPN from https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml  
Pretrain weight : "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  

