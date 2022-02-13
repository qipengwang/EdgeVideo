from vision.utils.eval_via_voc_metrics import voc_eval, voc_ap
import pandas as pd
from collections import defaultdict
import os
import json
import argparse
from vision.utils.configurer import Configurer

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluating With Pytorch')
parser.add_argument('--config', default='cfg/default.cfg', help='config file name')
args = parser.parse_args()
cfg = Configurer(args.config)
cfg_fn_no_extension = os.path.splitext(os.path.basename(args.config))[0]

def dump_continual_result(rootdir, baseline=False, num_window=10):
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
        df = pd.read_csv(os.path.join(rootdir, f'{cfg_fn_no_extension}_baseline.csv'))  # image,prediction,score,xmin,ymin,xmax,ymax
        dump_df_result(df)
        with open(os.path.join(savedir, f'{cfg_fn_no_extension}_baseline.json'), 'w') as f:
            json.dump(result, f)
    else:
        for i in range(1, num_window+1):
            df = pd.read_csv(os.path.join(rootdir, f'{cfg_fn_no_extension}_{i}_{num_window}.csv'))  # image,prediction,score,xmin,ymin,xmax,ymax
            dump_df_result(df)
        with open(os.path.join(savedir, f'{cfg_fn_no_extension}_continual.json'), 'w') as f:
            json.dump(result, f, indent=2)


def dump_ground_truth(rootdir):
    savedir = rootdir
    gt = defaultdict(list)
    for city in os.listdir(rootdir):
        if not os.path.isdir(os.path.join(rootdir, city)): continue
        for file in os.listdir(os.path.join(rootdir, city)):
            if not file.endswith('.csv'): continue
            img_key = os.path.join(city, file.replace('.csv', ''))
            # print(os.path.join(rootdir, city, file))
            df = pd.read_csv(os.path.join(rootdir, city, file))
            for _, row in df.iterrows():
                if row['confidence'] > 0.5:
                    gt[img_key].append({
                        'class': row['class'],
                        'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                    })
    with open(os.path.join(savedir, 'ground_truth.json'), 'w') as f:
        json.dump(gt, f, indent=2)


def dump2json():
    dump_continual_result(rootdir=f'evaluate/{cfg_fn_no_extension}', baseline=False)
    dump_continual_result(rootdir=f'evaluate/{cfg_fn_no_extension}', baseline=True)
    dump_ground_truth(rootdir='data/labels_coco91')
    # dump_ground_truth(rootdir='data/labels_coco80')
    # dump_ground_truth(rootdir='data/labels_voc')
    # exit()


if __name__ == "__main__":
    # dump2json()
    with open('data/labels_coco91/ground_truth.json') as f:
        gt = json.load(f)
    classes = set()
    for labels in gt.values():
        for v in labels:
            classes.add(v['class'])
    classes = sorted(classes)
    print(classes)
    # print('=========== continual ===========')
    for class_id in classes:
        continual_result = voc_eval(detpath=f'evaluate/{cfg_fn_no_extension}/{cfg_fn_no_extension}_continual.json', annopath='data/labels_coco91/ground_truth.json', classid=class_id)[2]
        baseline_result  = voc_eval(detpath=f'evaluate//{cfg_fn_no_extension}/{cfg_fn_no_extension}_baseline.json', annopath='data/labels_coco91/ground_truth.json', classid=class_id)[2]
        print(class_id, 'baseline = ', baseline_result, 'continual = ', continual_result)
    #     print(voc_eval(detpath='evaluate/continual.json', annopath='data/labels_coco91/berlin/ground_truth.json', classid=class_id)[2])
    # print('=========== baseline  ===========')
    # for class_id in classes:
    #     print(voc_eval(detpath='evaluate/continual_baseline.json', annopath='data/labels_coco91/berlin/ground_truth.json', classid=class_id)[2])
 
    
