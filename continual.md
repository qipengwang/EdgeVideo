# Continual Learning

## Prepare

### Download
- download the [CityScapes dataset (leftImg8bit_trainvaltest.zip)](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
- unzip the images to the `data/images` directory and move all of the citys to `data/images`
```
unzip -d data/images/ leftImg8bit_trainvaltest.zip
cd data/images
mv leftImg8bit/*/* .
rm -r README license.txt leftImg8bit/
```

### Environment
- torch >= 1.9.0
- torchvision >= 0.10.0
- `pip install -r requirements.txt`

### Generate SSD label

Use the command below to generate the label.


```
python genlabel.py
```

The csv label files are saved to `data/label_{TYPE}` directory. The `TYPE` is in `[voc, coco80, coco91]`. 
- the voc type has 21 classes; 
- the coco80 has 80 classes (the #output of the Yolo model, also used in many other projects); 
- the coco91 has 91 classes (the classes definied in the coco paper)

The label files are stored in the `preprocess/` directory


## Train

### use the pretrained model provided  by README

the model is trained on the VOC dataset

follow the instructions in [README](./README.md) to download the pretrained models into `models/` directory

```
python continual_ssd.py --dataset_type city_scapes --datasets data --validation_dataset data --net mb2-ssd-lite --resume models/mb2-ssd-lite-mp-0_686.pth --use_cuda
```

refer to [README](./README.md) for more information about the args

### use pretrained model provided by pytorch

the model is trained on the COCO dataset with 91 labels

```
python continual_ssd_tvmodel.py --dataset_type city_scapes --datasets data --validation_dataset data --net mb2-ssd-lite --use_cuda
```
