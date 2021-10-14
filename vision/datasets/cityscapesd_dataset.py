import torch
from torch.utils.data import Dataset
import os

class CityscapesDataset(Dataset):
    def __init__(self, rootdir, label_rootdir):
        super().__init__()
        self.rootdir = rootdir
        self.image_rootdir = os.path.join(rootdir, 'images')
        self.label_rootdir = os.path.join(rootdir, 'labels')
        self.ids = CityscapesDataset.get_image_ids(self.image_rootdir)
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        return len(self.ids)
    
    def get_image_ids(cls, image_rootdir):
        ids = []
        for root, dirs, files in os.walk(image_rootdir):
            for file in files:
                ids.append(os.path.relpath(os.path.join(root, file), image_rootdir))
        return ids

        
