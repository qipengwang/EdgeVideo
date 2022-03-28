import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import cv2
import math
from .basic_dataset import BasicDataset
from more_itertools import chunked
from vision.sampler.basic_sampler import BasicSampler

class ContinualDataset(BasicDataset):
    def __init__(self, rootdir: str, subdirs: list=[], transform=None, target_transform=None, 
                mode: str='TRAIN', labelfile: str='models/cityscapes-labels.txt', 
                imagedir: str='images', labeldir: str='labels', num_window: int=10,
                sampler: BasicSampler=None):
        super(ContinualDataset, self).__init__(rootdir, subdirs=subdirs, imagedir=imagedir, labeldir=labeldir, labelfile=labelfile,
                                                         transform=transform, target_transform=target_transform, mode='ALL')
        mode = mode.upper()
        assert mode in ['TRAIN', 'TEST'], f'mode should in ["TRAIN", "TEST"]'
        self.cur_window = 0
        self.window_size = math.ceil(len(self.ids) / num_window)
        self.chunked_ids = list(chunked(self.ids, self.window_size))
        self.num_window = len(self.chunked_ids)
        self.sampler = sampler
        self.current_samples = self.chunked_ids[self.cur_window]
        if self.sampler:
            self.current_samples = self.sampler.sample(self.current_samples)
            print(f'current_samples {len(self.chunked_ids[self.cur_window])}--{len(self.current_samples)}')
        

    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.image_rootdir, self.current_samples[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels = self.get_label(index)
        # print(boxes.shape, labels.shape, sep='\n')
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return self.current_samples[index], image, boxes, labels
    
    def __len__(self):
        return len(self.current_samples)
    
    def get_image(self, index):
        image = cv2.imread(os.path.join(self.image_rootdir, self.current_samples[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            boxes, labels = self.get_label(index)
            image, _, _ = self.transform(image, boxes, labels)
        return image

    def get_label(self, index):
        # xmin,ymin,xmax,ymax,confidence,class,name
        df = pd.read_csv(os.path.join(self.label_rootdir, self.current_samples[index] + '.csv'))
        boxes, labels = [], []
        for _, row in df.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(row['class'])
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def next_window(self):
        self.cur_window += 1
        if self.cur_window < self.num_window:
            self.current_samples = self.chunked_ids[self.cur_window]
            if self.sampler:
                self.current_samples = self.sampler.sample(self.current_samples)
                print(f'current_samples {len(self.chunked_ids[self.cur_window])}--{len(self.current_samples)}')
        return self.cur_window < self.num_window



        
