import torch
import os
import shutil
from collections import defaultdict

coco2voc = defaultdict(int)
with open('preprocess/coco2voc.txt') as  f:
    for line in f:
        coco, voc = line.strip().split()
        coco2voc[int(coco)] = int(voc)

coco80_to_coco91 = defaultdict(int)
with open('preprocess/coco80-to-coco91.txt') as  f:
    for line in f:
        coco80, coco91 = line.strip().split()
        coco80_to_coco91[int(coco80)] = int(coco91)

image_root = 'data/images/'
coco80_root = 'data/labels_coco80/'
coco91_root = 'data/labels_coco91/'
voc_root = 'data/labels_voc/'

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
if torch.cuda.is_available():
    model.cuda()

for root, folders, files in os.walk(image_root):
    for file in files:
        filepath = os.path.join(root, file)
        results = model([filepath])
        df_raw = results.pandas().xyxy[0]

        df_coco80 = df_raw.copy()
        savepath = os.path.join(coco80_root, os.path.relpath(filepath, image_root) + '.csv')
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        df_coco80.to_csv(savepath, index=False)
        
        df_coco91 = df_raw.copy()
        df_coco91['class'] = df_coco91['class'].map(lambda x: coco80_to_coco91[x])
        savepath = os.path.join(coco91_root, os.path.relpath(filepath, image_root) + '.csv')
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        df_coco91.to_csv(savepath, index=False)

        df_voc = df_raw.copy()
        df_voc['class'] = df_voc['class'].map(lambda x: coco2voc[x])
        savepath = os.path.join(voc_root, os.path.relpath(filepath, image_root) + '.csv')
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        df_voc.to_csv(savepath, index=False)
        exit()


