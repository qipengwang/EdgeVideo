from vision.utils.eval_via_voc_metrics import voc_eval, voc_ap
import pandas as pd
from collections import defaultdict
import os
import json

def dump_continual_result(rootdir, city, savedir=None, baseline=False, num_window=10):
    if not savedir:
        savedir = rootdir
    result = defaultdict(list)

    def dump_df_result(df):
        for _, row in df.iterrows():
            result[row['image']].append({
                'class': row['prediction'],
                'score': row['score'],
                'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            })

    if baseline:
        df = pd.read_csv(os.path.join(rootdir, f'continual_{city}_baseline.csv'))  # image,prediction,score,xmin,ymin,xmax,ymax
        dump_df_result(df)
        with open(os.path.join(savedir, 'continual_baseline.json'), 'w') as f:
            json.dump(result, f)
    else:
        for i in range(1, num_window+1):
            df = pd.read_csv(os.path.join(rootdir, f'continual_{city}_{i}_{num_window}.csv'))  # image,prediction,score,xmin,ymin,xmax,ymax
            dump_df_result(df)
        with open(os.path.join(savedir, 'continual.json'), 'w') as f:
            json.dump(result, f, indent=2)


def dump_ground_truth(rootdir, city, savedir):
    if not savedir:
        savedir = os.path.join(rootdir, city)
    gt = defaultdict(list)
    for file in os.listdir(os.path.join(rootdir, city)):
        img_key = os.path.join(city, file.replace('.csv', ''))
        df = pd.read_csv(os.path.join(rootdir, city, file))
        for _, row in df.iterrows():
            if row['confidence'] > 0.5:
                gt[img_key].append({
                    'class': row['class'],
                    'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                })
    with open(os.path.join(savedir, 'ground_truth.json'), 'w') as f:
        json.dump(gt, f, indent=2)


def dump2json(city='berlin', savedir=None):
    dump_continual_result(rootdir='evaluate', city=city, savedir=savedir, baseline=False)
    dump_continual_result(rootdir='evaluate', city=city, savedir=savedir, baseline=True)
    dump_ground_truth(rootdir='data/labels_coco91', city=city, savedir=savedir)


if __name__ == "__main__":
    with open('data/labels_coco91/berlin/ground_truth.json') as f:
        gt = json.load(f)
    classes = set()
    for labels in gt.values():
        for v in labels:
            classes.add(v['class'])
    classes = sorted(classes)
    # print('=========== continual ===========')
    for class_id in classes:
        continual_result = voc_eval(detpath='evaluate/continual.json', annopath='data/labels_coco91/berlin/ground_truth.json', classid=class_id)[2]
        baseline_result  = voc_eval(detpath='evaluate/continual_baseline.json', annopath='data/labels_coco91/berlin/ground_truth.json', classid=class_id)[2]
        print(class_id, 'baseline = ', baseline_result, 'continual = ', continual_result)
    #     print(voc_eval(detpath='evaluate/continual.json', annopath='data/labels_coco91/berlin/ground_truth.json', classid=class_id)[2])
    # print('=========== baseline  ===========')
    # for class_id in classes:
    #     print(voc_eval(detpath='evaluate/continual_baseline.json', annopath='data/labels_coco91/berlin/ground_truth.json', classid=class_id)[2])
 
    
