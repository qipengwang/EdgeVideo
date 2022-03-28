from vision.utils.eval_via_voc_metrics import voc_eval, voc_ap
import pandas as pd
from collections import defaultdict
import os
import json
import numpy as np
import argparse
from vision.utils.configurer import Configurer

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluating With Pytorch')
parser.add_argument('--config', default='cfg/default.cfg', help='config file name')
parser.add_argument('--thres', type=float, default=0.5)
args = parser.parse_args()
cfg = Configurer(args.config)
cfg_fn_no_extension = os.path.splitext(os.path.basename(args.config))[0]

def dump_continual_result(rootdir, baseline=False, num_window=cfg.num_window):
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
    for i, subdir in enumerate(cfg.subdirs):
        print(f'{i+1}/{len(cfg.subdirs)}')
        if not os.path.isdir(os.path.join(rootdir, subdir)): continue
        gt = defaultdict(list)
        for file in os.listdir(os.path.join(rootdir, subdir)):
            if not file.endswith('.csv'): continue
            img_key = os.path.join(subdir, file.replace('.csv', ''))
            df = pd.read_csv(os.path.join(rootdir, subdir, file))
            for _, row in df.iterrows():
                if row['confidence'] > 0.5:
                    gt[img_key].append({
                        'class': row['class'],
                        'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                    })
        with open(os.path.join(savedir, f'{subdir}.json'), 'w') as f:
            json.dump(gt, f, indent=2)


def dump2json():
    dump_continual_result(rootdir=f'evaluate/{cfg_fn_no_extension}', baseline=False)
    dump_continual_result(rootdir=f'evaluate/{cfg_fn_no_extension}', baseline=True)
    # dump_ground_truth(rootdir=os.path.join(cfg.validation_dataset, cfg.labeldir))
    # dump_ground_truth(rootdir='data/labels_coco80')
    # dump_ground_truth(rootdir='data/labels_voc')
    # exit()


if __name__ == "__main__":
    # dump2json()
    # savedir = os.path.join(cfg.validation_dataset, 'groundtruth_fps' if 'fps' in cfg.labeldir else 'groundtruth', *cfg.labeldir.split('/')[1:])

    # os.makedirs(savedir, exist_ok=True)
    # print(savedir)
    classes = set()
    for _, subdir in enumerate(cfg.subdirs):
        with open(os.path.join(savedir, f'{subdir}.json')) as f:
            gt = json.load(f)
        for labels in gt.values():
            for v in labels:
                classes.add(v['class'])
    if not classes:
        with open(os.path.join(cfg.datasets, cfg.labeldir, 'ground_truth.json')) as f:
            gt = json.load(f)
        for labels in gt.values():
            for v in labels:
                classes.add(v['class'])
    classes = sorted(classes)
    print(args.config, classes)
    CRs, BRs = [], []
    # print('=========== continual ===========')
    for class_id in classes:
        annopaths = [os.path.join(savedir, f'{subdir}.json') for subdir in cfg.subdirs]
        if not annopaths:
            annopaths = os.path.join(cfg.datasets, cfg.labeldir, 'ground_truth.json')
        continual_result = voc_eval(detpath=f'evaluate/{cfg_fn_no_extension}/{cfg_fn_no_extension}_continual.json', 
                                    annopaths=annopaths, ovthresh=args.thres,
                                    classid=class_id)[2]
        baseline_result  = voc_eval(detpath=f'evaluate/{cfg_fn_no_extension}/{cfg_fn_no_extension}_baseline.json', 
                                    annopaths=annopaths, ovthresh=args.thres,
                                    classid=class_id)[2]
        if continual_result or baseline_result:
            CRs.append(continual_result)
            BRs.append(baseline_result)
        print(class_id, 'baseline = ', baseline_result, 'continual = ', continual_result)
    print(len(BRs), 'baseline = ', np.mean(BRs), 'continual = ', np.mean(CRs))
    #     print(voc_eval(detpath='evaluate/continual.json', annopath='data/labels_coco91/berlin/ground_truth.json', classid=class_id)[2])
    # print('=========== baseline  ===========')
    # for class_id in classes:
    #     print(voc_eval(detpath='evaluate/continual_baseline.json', annopath='data/labels_coco91/berlin/ground_truth.json', classid=class_id)[2])
 
    
