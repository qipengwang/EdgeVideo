import cv2
import os
import time
import multiprocessing as mp
from vision.utils.configurer import Configurer
import pandas as pd
from more_itertools import chunked


PREFIX = 'Sharx Demo Live Stream Cam - traffic circle _ rotary _ roundabout in Derry NH USA '
SUFFIX = '-SOKol9vTpwQ'
# parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
# parser.add_argument('--config', default='cfg/default.cfg', help='config file name')
# args = parser.parse_args()
# cfg = Configurer(args.config)

def rename_task(A, B, C):
    start_t = time.time()
    for f in os.listdir(os.path.join(A, B, C)):
        if '.png' not in f and f.endswith('.csv'):
            os.rename(os.path.join(A, B, C, f), os.path.join(A, B, C, f.replace('.csv', '.png.csv')))
    print(A, B, C, f'rename takes {time.time() - start_t} seconds')


def rename():
    start_t = time.time()
    label_dir = '../youtube/labels_fps'
    task_set = []
    for label_type in os.listdir(label_dir):
        for date in os.listdir(os.path.join(label_dir, label_type)):
            task_set.append([label_dir, label_type, date])
    with mp.Pool(int(mp.cpu_count() / 2)) as pool:
        pool.starmap(rename_task, [(label_dir, label_type, date) for label_dir, label_type, date in task_set])
    print(f'rename takes {time.time() - start_t} seconds')


def check_label_detect():
    with open('preprocess/coco-paper-labels.txt') as f:
        coco_labels = [l for l in f]
    det = pd.read_csv('test.csv')
    imgs = pd.unique(det['image'])
    for img_fn in imgs:
        img = cv2.imread(f'data/cityscapes/images/{img_fn}')
        labels = pd.read_csv(f'data/cityscapes/labels/labels_coco91/{img_fn}.csv')
        for _, label in labels.iterrows():
            cv2.rectangle(img, (int(label['xmin']), int(label['ymin'])), (int(label['xmax']), int(label['ymax'])), (255, 0, 0), 2)
            cv2.putText(img, f'{label["name"]}', (int(label['xmin']), int(label['ymin'])), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        for _, detection in det.iterrows():
            if detection['image'] == img_fn:
                cv2.rectangle(img, (int(detection['xmin']), int(detection['ymin'])), (int(detection['xmax']), int(detection['ymax'])), (0, 0, 255), 2)
                cv2.putText(img, f'{coco_labels[int(detection["prediction"])]}', (int(detection['xmin']), int(detection['ymin'])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{img_fn.split("/")[-1]}.png', img)
        print(img_fn)
    
##########################################       begin dump 1fps       ##########################################
# def dump_task(root, video_name):
#     print(f'start {video_name}')
#     cap = cv2.VideoCapture(os.path.join(root, 'videos', video_name))
#     fps = int(round(cap.get(cv2.CAP_PROP_FPS), 0))
#     save_dir = os.path.join(root, 'images_fps', os.path.splitext(video_name)[0])
#     os.makedirs(save_dir, exist_ok=True)
#     success, frame = cap.read()
#     i, j = 0, 0
#     while success :
#         i += 1
#         if i % fps == 0:
#             j += 1
#             cv2.imwrite(os.path.join(save_dir, f'{j}.png'), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             if j % 100 == 0:
#                 print(f'[{os.path.splitext(video_name)[0]}][{j}]')
#         success, frame = cap.read()
    
#     print(f'finish {video_name}')


# root = '../youtube/Derry/'
# videos = [
#     '2022_02_07-16_00.mp4',
#     '2022_02_07-22_00.mp4',
#     '2022_02_08-04_00.mp4',
#     '2022_02_08-10_00.mp4'
# ]
# with mp.Pool(4) as pool:
#     pool.starmap(dump_task, [(root, video) for video in videos])
##########################################       finish dump 1fps       ##########################################
# ['Kyiv', 'LaGrange', 'Tokyo']
for city in ['Amsterdam']:
    video_path = os.path.join('data/youtube/', city, 'videos')
    for file in os.listdir(video_path):
        os.rename(
            os.path.join(video_path, file),
            os.path.join(video_path, file
                                    .replace('【LIVE】Tokyo Shinjuku Live Cam新宿大ガード交差点【2022】 ', '').replace('-RQA5RcIZlAM', '')
                                    .replace('La Grange, Kentucky USA - Virtual Railfan LIVE ', '').replace('-y7QiNgui5Tg', '')
                                    .replace('DIRECTO KIEV _ Plaza de la Independencia ', '').replace('-ex5HS_63Hb4', '')
                                    .replace('WebCam.NL _ bizdam.nl _ live ultraHD Pan Tilt Zoom camera Beurs van Berlage, Amsterdam. (4K) ', '').replace('-KZP_sVrAE4U', '')
                                    .replace('-', '_').replace(' ', '-'))
        )
      