import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import cv2

class BasicDataset(Dataset):
    '''
    This is the basic dataset of continual without window
    the dataset structure is like this:
    rootdir
        |---imagedir
        |   |---subdir1
        |       |---img1
        |       |---img2
        |       |---img3
        |   |---subdir2
        |       .
        |       .
        |   |---subdir3
        |---imagedir
        |   |---subdir1
        |       |---img1.csv
        |       |---img2.csv
        |       |---img3.csv
        |   |---subdir2
        |       .
        |       .
        |   |---subdir3
    '''
    
    def __init__(self, rootdir: str, subdirs: list=[], transform=None, target_transform=None, mode: str='ALL', 
                    labelfile: str='preprocess/cityscapes-labels.txt', imagedir: str='images', labeldir: str='labels'):
        super().__init__()
        mode = mode.upper()
        self.rootdir = rootdir
        self.subdirs = subdirs
        self.image_rootdir = os.path.join(rootdir, imagedir)
        self.label_rootdir = os.path.join(rootdir, labeldir)
        if not self.subdirs or 'ALL' in self.subdirs:
            self.subdirs = os.listdir(self.image_rootdir)
        self.transform = transform
        self.target_transform = target_transform
        assert mode in ['ALL', 'TRAIN', 'TEST'], f'mode {mode} is not supported that should in ["TRAIN", "TEST", "ALL"]'
        self.mode = mode
        self.test_percentage = 0.2
        self.ids = self.get_image_ids()
        self.class_names = BasicDataset.get_classname(labelfile)
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
    

    def get_classname(cls, labelfile: str=None):
        if not labelfile or not os.path.exists(labelfile):
            return []
        with open(labelfile) as f:
            classname = [line.strip() for line in f]
        return classname

    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.image_rootdir, self.ids[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels = self.get_label(index)
        # print(boxes.shape, labels.shape, sep='\n')
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return self.ids[index], image, boxes, labels
    
    def __len__(self):
        return len(self.ids)

    def get_classnames(cls, labelfile: str=None):
        if not labelfile:
            return ('BACKGROUND',
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            )
        with open(labelfile) as f:
            classnames = [line.strip() for line in f]
        return tuple(classnames)
    
    def get_image_ids(self):
        ids = []
        # print('get_image_ids')
        for city in self.subdirs:
            for root, _, files in os.walk(os.path.join(self.image_rootdir, city) if city else self.image_rootdir):
                for file in files:
                    relative_path = os.path.relpath(os.path.join(root, file), self.image_rootdir)
                    df = pd.read_csv(os.path.join(self.label_rootdir, relative_path + '.csv'))
                    if len(df):
                        ids.append(os.path.join(relative_path))
        if self.mode == 'TRAIN':
            ids = [ids[i] for i in range(len(ids)) if i % int(len(ids) * self.test_percentage) != 0]
        elif self.mode == 'TEST':
            ids = [ids[i] for i in range(len(ids)) if i % int(len(ids) * self.test_percentage) == 0]
        return ids
    
    def get_image(self, index):
        image = cv2.imread(os.path.join(self.image_rootdir, self.ids[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            boxes, labels = self.get_label(index)
            image, _, _ = self.transform(image, boxes, labels)
        return image

    def get_label(self, index):
        # xmin,ymin,xmax,ymax,confidence,class,name
        df = pd.read_csv(os.path.join(self.label_rootdir, self.ids[index] + '.csv'))
        boxes, labels = [], []
        for _, row in df.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(row['class'])
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    


        
