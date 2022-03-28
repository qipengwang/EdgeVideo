import os
import cv2
import json
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt

from filter.differencer import PixelDiff, AreaDiff, EdgeDiff


ROOT = 'data/youtube'
city = 'Derry'
videos = [
    '2022_02_07-16_00.mp4',
    '2022_02_07-22_00.mp4',
    '2022_02_08-04_00.mp4',
    '2022_02_08-10_00.mp4'
]


def diff_task(video):
    pixel_diff = PixelDiff()
    area_diff = AreaDiff()
    edge_diff = EdgeDiff()
    print(f'begin {video}')
    idx = 2
    save_path = os.path.join(ROOT, city, 'diff_fps', os.path.splitext(video)[0])
    os.makedirs(save_path, exist_ok=True)
    image_dir = os.path.join(ROOT, city, 'images_fps', os.path.splitext(video)[0])
    pre_frame = cv2.cvtColor(cv2.imread(os.path.join(image_dir, '1.png')), cv2.COLOR_BGR2RGB)
    cur_frame = None
    pd, ad, ed = [], [], []

    while os.path.exists(os.path.join(image_dir, f'{idx}.png')):
        cur_frame = cv2.cvtColor(cv2.imread(os.path.join(image_dir, f'{idx}.png')), cv2.COLOR_BGR2RGB)
        pd.append(pixel_diff.cal_frame_diff(cur_frame, pre_frame))
        ad.append(area_diff.cal_frame_diff(cur_frame, pre_frame))
        ed.append(edge_diff.cal_frame_diff(cur_frame, pre_frame))
        pre_frame = cur_frame
        idx += 1
        if idx % 100 == 0:
            print(f'[{video}] [{idx}]')
    
    with open(os.path.join(save_path, 'pixel_diff.json'), 'w') as f:
        json.dump(pd, f, indent=2)
    
    with open(os.path.join(save_path, 'area_diff.json'), 'w') as f:
        json.dump(ad, f, indent=2)
    
    with open(os.path.join(save_path, 'edge_diff.json'), 'w') as f:
        json.dump(ed, f, indent=2)
    
    print(f'finish {video}')


# with mp.Pool(4) as pool:
#     pool.starmap(diff_task, [(video, ) for video in videos])

pd, ad, ed = [], [], []
for video in videos:
    save_path = os.path.join(ROOT, city, 'diff_fps', os.path.splitext(video)[0])
    with open(os.path.join(save_path, 'pixel_diff.json')) as f:
        pd.extend(json.load(f))
    with open(os.path.join(save_path, 'area_diff.json')) as f:
        ad.extend(json.load(f))
    with open(os.path.join(save_path, 'edge_diff.json')) as f:
        ed.extend(json.load(f))

plt.figure()
sns.distplot(pd, rug=True)
plt.savefig('images/pd.png')
plt.xlim(0, 0.2)
plt.savefig('images/pd-xlim.png')

plt.figure()
sns.distplot(ed, rug=True)
plt.savefig('images/ed.png')
plt.xlim(0, 0.2)
plt.savefig('images/ed-xlim.png')

plt.figure()
sns.distplot(ad, rug=True)
plt.savefig('images/ad.png')
plt.xlim(0, 0.2)
plt.savefig('images/ad-xlim.png')
