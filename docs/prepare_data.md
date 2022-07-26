# Data Preparation
We support 4 popular settings in DAOD as listed below:

|                         | Source             | Target                   | Test                   |
| ----------------------- | ------------------ | ------------------------ | ---------------------- |
| normal to foggy (C2F)   | cityscapes (train) | cityscapes-foggy (train) | cityscapes-foggy (val) |
| small to large (C2B)    | cityscapes (train) | BDD100K (train)          | BDD100K (val)          |
| across cameras (K2C)    | KITTI (train-car)  | cityscapes (train-car)   | cityscapes (val-car)   |
| synthetic to real (S2C) | Sim10K (train-car) | cityscapes (train-car)   | cityscapes (val-car)   |

All datasets are aranged in the format of PASCAL VOC as follows:
```shell
# cityscapes          
- VOC2007_city 
    - ImageSets  
    - JPEGImages  
    - Annotations  
```
Here are datasets used in this paper.
```shell
SPLITS = [
        ("VOC2007_citytrain", 'data/VOC2007_citytrain', "train", 8),
        ("VOC2007_foggytrain", 'data/VOC2007_foggytrain', "train", 8),
        ("VOC2007_foggyval", 'data/VOC2007_foggyval', "val", 8),
        ("VOC2007_citytrain1", 'data/VOC2007_citytrain1', "train", 1),
        ("VOC2007_cityval1", 'data/VOC2007_cityval1', "val", 1),
        ("VOC2007_bddtrain", 'data/VOC2007_bddtrain', "train", 8),
        ("VOC2007_bddval", 'data/VOC2007_bddval', "val", 8),
        ("VOC2007_kitti1", 'data/kitti', "train", 1),
        ("VOC2007_sim1", 'data/sim', "train", 1),
    ]
```
## CitysScape and FoggyCityscape
1. register and download from [CitysScape](https://www.cityscapes-dataset.com/) to ```data/```
2. transform segmentation annotations to detection formats
   - For multi classes, ```python tools/trans_seg_to_det_multi.py```
   - For single class, ```python tools/trans_seg_to_det_multi.py```
3. check the annotations and make txt, ```python tools/make_VOC_txt.py```

## KITTI, Sim10k, BDD100k
Download from [KITTI](http://www.cvlibs.net/datasets/kitti/), [Sim10k](https://fcav.engin.umich.edu/sim-dataset/), [BDD100k](https://www.bdd100k.com/) to ```data/```.