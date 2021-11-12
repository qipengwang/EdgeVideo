import time
import argparse
import torch
import logging
import itertools
import os
import pandas as pd
import json


def str2bool(s):
    return s.lower() in ('true', '1')


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval
        

def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)
        
        
def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))

def arg_parser():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')

    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and open_images.')

    parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
    parser.add_argument('--validation_dataset', help='Dataset directory path')
    parser.add_argument('--balance_data', action='store_true',
                        help="Balance training data by down-sampling more frequent labels.")
    parser.add_argument('--city', default='berlin',help='The chosed city in cityscapes dataset')

    parser.add_argument('--net', default="mb3-large-ssd-lite",
                        choices=['mb1-ssd', 'mb1-ssd-lite', 'mb2-ssd-lite', 'mb3-large-ssd-lite', 'mb3-small-ssd-lite', 'vgg16-ssd'],
                        help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.")
    parser.add_argument('--freeze_base_net', action='store_true',
                        help="Freeze base net layers.")
    parser.add_argument('--freeze_net', action='store_true',
                        help="Freeze all the layers except the prediction head.")

    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                        help='Width Multiplifier for MobilenetV2')
    parser.add_argument('--ssd320', action='store_true', help='use SSD320 instead of SSD300')

    # Params for SGD
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--base_net_lr', default=None, type=float,
                        help='initial learning rate for base net.')
    parser.add_argument('--extra_layers_lr', default=None, type=float,
                        help='initial learning rate for the layers not in base net and prediction heads.')

    # Params for loading pretrained basenet or checkpoints.
    parser.add_argument('--base_net',
                        help='Pretrained base model')
    parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')

    # Scheduler
    parser.add_argument('--scheduler', default="multi-step", type=str, choices=['multi-step', 'cosine'],
                        help="Scheduler for SGD. It can one of multi-step and cosine")

    # Params for Multi-step Scheduler
    parser.add_argument('--milestones', default="80,100", type=str,
                        help="milestones for MultiStepLR")

    # Params for Cosine Annealing
    parser.add_argument('--t_max', default=120, type=float,
                        help='T_max value for Cosine Annealing Scheduler.')

    # Train params
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', default=200, type=int,
                        help='the number epochs')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--validation_epochs', default=5, type=int,
                        help='the number epochs')
    parser.add_argument('--debug_steps', default=100, type=int,
                        help='Set the debug log output frequency.')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA to train model')

    parser.add_argument('--checkpoint_folder', default='models/',
                        help='Directory for saving checkpoint models')
    return parser



def dump_cityscapes_imageids(rootdir: str, savepath, city: str=""):
    image_rootdir = os.path.join(rootdir, 'images')
    label_rootdir = os.path.join(rootdir, 'labels')
    ids = []
    for root, _, files in os.walk(os.path.join(image_rootdir, city) if city else image_rootdir):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), image_rootdir)
            df = pd.read_csv(os.path.join(label_rootdir, relative_path + '.csv'))
            if len(df):
                ids.append(os.path.join(file))
    with open(savepath, 'w')  as f:
        f.write('\n'.join(ids))
    return ids


def dump_cityscapes_labels(rootdir: str, savepath, city: str=""):
    # xmin,ymin,xmax,ymax,confidence,class,name
    label_rootdir = os.path.join(rootdir, 'labels')
    records = {}
    for root, _, files in os.walk(os.path.join(label_rootdir, city) if city else label_rootdir):
        for file in files:
            df = pd.read_csv(os.path.join(root, file))
            labels = []
            for row in df.iterrows():
                labels.append({'class': row['class'], 'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']]})
            records[file] = labels
    with open(savepath, 'w') as f:
        json.dump(records, f, indent=2)
    return records

