import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import cv2
import math
from .cityscapes_dataset import CityscapesDataset
from more_itertools import chunked

class ContinualCityscapesDataset(CityscapesDataset):
    def __init__(self, rootdir: str, city: str="", transform=None, target_transform=None, 
                mode: str='TRAIN', labelfile: str='models/cityscapes-labels.txt', imagedir: str='images', labeldir: str='labels', num_window: int=10):
        super(ContinualCityscapesDataset, self).__init__(rootdir, city=city, imagedir=imagedir, labeldir=labeldir, labelfile=labelfile,
                                                         transform=transform, target_transform=target_transform, mode='ALL')
        mode = mode.upper()
        assert mode in ['TRAIN', 'TEST'], f'mode should in ["TRAIN", "TEST"]'
        if mode == 'TRAIN':
            self.cur_window = 0
        else:
            self.cur_window = 0
        self.window_size = math.ceil(len(self.ids) / num_window)
        self.chunked_ids = list(chunked(self.ids, self.window_size))
        self.num_window = len(self.chunked_ids)
        print(mode, len(self.ids), len(self.chunked_ids))

    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.image_rootdir, self.chunked_ids[self.cur_window][index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels = self.get_label(index)
        # print(boxes.shape, labels.shape, sep='\n')
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return self.chunked_ids[self.cur_window][index], image, boxes, labels
    
    def __len__(self):
        return len(self.chunked_ids[self.cur_window])
    
    def get_image(self, index):
        image = cv2.imread(os.path.join(self.image_rootdir, self.chunked_ids[self.cur_window][index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            boxes, labels = self.get_label(index)
            image, _, _ = self.transform(image, boxes, labels)
        return image

    def get_label(self, index):
        # xmin,ymin,xmax,ymax,confidence,class,name
        df = pd.read_csv(os.path.join(self.label_rootdir, self.chunked_ids[self.cur_window][index] + '.csv'))
        boxes, labels = [], []
        for _, row in df.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(row['class'])
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def next_window(self):
        self.cur_window += 1
        return self.cur_window < self.num_window



        
