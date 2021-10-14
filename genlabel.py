import torch
import os
import shutil

image_root = 'data/images/'
label_root = 'data/labels/'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.cuda()
imgs = []
index = 0
for root, folders, files in os.walk(image_root):
    for file in files:
        filepath = os.path.join(root, file)
        results = model([filepath])
        df = results.pandas().xyxy[0]
        label_dir = os.path.join(label_root, os.path.dirname(filepath).replace(image_root, ''))
        if not os.path.exists(label_dir):
            print(label_dir)
            os.makedirs(label_dir)
        df.to_csv(os.path.join(label_dir, file + '.csv'), index=False)
        index += 1
        if index % 100 == 0:
            print(f'finish {index} images')

