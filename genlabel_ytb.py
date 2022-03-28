'''
cd EdgeVideo && source activate torch10
CUDA_VISIBLE_DEVICES=0 python -u genlabel_ytb.py 0 15
CUDA_VISIBLE_DEVICES=0 python -u genlabel_ytb.py 16 32

CUDA_VISIBLE_DEVICES=1 python -u genlabel_ytb.py 33 48
CUDA_VISIBLE_DEVICES=1 python -u genlabel_ytb.py 49 65

CUDA_VISIBLE_DEVICES=2 python -u genlabel_ytb.py 66 81
CUDA_VISIBLE_DEVICES=2 python -u genlabel_ytb.py 82 98

CUDA_VISIBLE_DEVICES=3 python -u genlabel_ytb.py 99 114
CUDA_VISIBLE_DEVICES=3 python -u genlabel_ytb.py 115 131
'''

import torch
import os
import sys
import shutil
import cv2
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


    
def gen_label(frame, videoname, filename):
    results = model([frame])
    df_raw = results.pandas().xyxy[0]

    df_coco80 = df_raw.copy()
    savepath = os.path.join(coco80_root, videoname, filename)
    df_coco80.to_csv(savepath, index=False)
    
    df_coco91 = df_raw.copy()
    df_coco91['class'] = df_coco91['class'].map(lambda x: coco80_to_coco91[x])
    savepath = os.path.join(coco91_root, videoname, filename)
    df_coco91.to_csv(savepath, index=False)

    df_voc = df_raw.copy()
    df_voc['class'] = df_voc['class'].map(lambda x: coco2voc[x])
    savepath = os.path.join(voc_root, videoname, filename)
    df_voc.to_csv(savepath, index=False)


if __name__ == '__main__':
    # ['Kyiv', 'LaGrange', 'Tokyo', 'Amsterdam']
    root = f'data/youtube/{sys.argv[1]}'
    os.makedirs(os.path.join(root, 'labels_fps'), exist_ok=True)
    os.makedirs(os.path.join(root, 'images_fps'), exist_ok=True)
    skip = set()
    if os.path.exists(os.path.join(root, 'labels_fps/finished.txt')):
        with open(os.path.join(root, 'labels_fps/finished.txt')) as f:
            for l in f:
                skip.add(l)

    video_root  = os.path.join(root, 'videos')
    coco80_root = os.path.join(root, 'labels_fps/labels_coco80/')
    coco91_root = os.path.join(root, 'labels_fps/labels_coco91/')
    voc_root    = os.path.join(root, 'labels_fps/labels_voc/')

    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    if torch.cuda.is_available():
        model.cuda()

    fout = open(os.path.join(root, 'labels_fps/finished.txt'), 'a+')
    files = [f for f in os.listdir(video_root) if f.endswith('.mp4')]
    for idx, file in enumerate(files):
        os.makedirs(os.path.join(coco80_root, os.path.splitext(file)[0]), exist_ok=True)
        os.makedirs(os.path.join(coco91_root, os.path.splitext(file)[0]), exist_ok=True)
        os.makedirs(os.path.join(voc_root, os.path.splitext(file)[0]), exist_ok=True)
        
        cap = cv2.VideoCapture(os.path.join(video_root, file))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS), 0))
        success, frame = cap.read()
        i, j = 0, 0
        while success :
            i += 1
            if i % fps == 0:
                j += 1
                gen_label(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), os.path.splitext(file)[0], f'{j}.png.csv')
                if j % 100 == 0:
                    print(f'[{idx+1:3d}/{len(files):3d}] [{file}]: {j} frames')
            success, frame = cap.read()
        fout.write(f'{file}\n')
    fout.close()


