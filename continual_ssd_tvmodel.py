import argparse
import os
import sys
import itertools
import numpy as np
from numpy.core.fromnumeric import squeeze
import pandas as pd
import cv2

import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from vision.datasets.continual_cityscapes_dataset import ContinualCityscapesDataset
from vision.transforms.transforms import *
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels, arg_parser
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.datasets.cityscapes_dataset import CityscapesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config, mobilenetv1_ssd_config, squeezenet_ssd_config, mobilenetv3_ssd320_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.utils.configurer import Configurer
from vision.utils.logger import Logger


# prepare for training
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--config', default='cfg/default.cfg', help='config file name')
args = parser.parse_args()
cfg = Configurer(args.config)
Logger.set_log_name(args.config)
logger = Logger.get_logger()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and cfg.use_cuda else "cpu")


def predict(net, dataset, savepath, cfg):
    logger.info('predicting...')
    net.eval()
    
    df = pd.DataFrame(columns=['image', 'prediction', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    loader = DataLoader(dataset, batch_size=4, num_workers=cfg.num_workers, shuffle=False, collate_fn=lambda a: list(map(list, zip(*a))))

    for _, data in enumerate(loader):
        ids, images, _, _ = data
        images = torch.stack(images).to(DEVICE)
        predictions = net(images)
        for id, pred in zip(ids, predictions):
            indices = torch.nonzero(pred['scores'] >= 0.5).squeeze(1)
            labels = pred['labels'][indices].cpu().detach().numpy()
            boxes = pred['boxes'][indices].cpu().detach().numpy()
            scores = pred['scores'][indices].cpu().detach().numpy()
            for i in range(boxes.shape[0]):
                df_row = {
                    'image': id, 
                    'prediction': labels[i], 
                    'score': scores[i], 
                    'xmin': boxes[i][0], 
                    'ymin': boxes[i][1],
                    'xmax': boxes[i][2],
                    'ymax': boxes[i][3]
                }
                df = df.append(df_row, ignore_index=True)
    df.to_csv(savepath, index=None)


def train(net, dataset, optimizer, scheduler, debug_steps=10):
    logger.info('training...')
    net.train()
    index = 0
    loader = DataLoader(dataset, cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, collate_fn=lambda a: list(map(list, zip(*a))))
    for epoch in range(cfg.num_epochs):
        scheduler.step()
        running_loss = 0.0
        for _, data in enumerate(loader):
            index += 1
            _, images, boxes, labels = data
            images = torch.stack(images).to(DEVICE)
            targets = [{
                'boxes': torch.tensor(box, device=DEVICE),
                'labels': torch.tensor(label, device=DEVICE)
            } for box, label in zip(boxes, labels)]
            optimizer.zero_grad()
            res = net(images, targets)
            loss = res['bbox_regression'] + res['classification']
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if index and index % debug_steps == 0:
                logger.info(f"Epoch: {epoch}, Global Step: {index}, Average Loss: {running_loss / debug_steps:.4f}")
                running_loss = 0.0



if __name__ == '__main__':
    if cfg.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("Use Cuda.")

    target_transform = None
    test_transform = Compose([
        ConvertFromInts(),
        lambda img, boxes=None, labels=None: (img / 255, boxes, labels),
        ToTensor(),
    ])
    train_transform = Compose([
        ConvertFromInts(),
        PhotometricDistort(),
        lambda img, boxes=None, labels=None: (img / 255, boxes, labels),
        ToTensor(),
    ])

    assert len(cfg.datasets) == 1, "continual learning not supports multiply dataset"
    
    dataset_path = cfg.datasets[0]
    if cfg.dataset_type == 'city_scapes':
        train_dataset = ContinualCityscapesDataset(dataset_path, city=cfg.city, labeldir='labels_coco91', labelfile='preprocess/coco-paper-labels.txt',
                                            transform=train_transform, target_transform=target_transform, mode='TRAIN')
        val_dataset = ContinualCityscapesDataset(cfg.validation_dataset, city=cfg.city, labeldir='labels_coco91', labelfile='preprocess/coco-paper-labels.txt',
                                                transform=test_transform, target_transform=target_transform, mode='TEST')
        num_classes = len(train_dataset.class_names)
    else:
        raise ValueError(f"Dataset type {cfg.dataset_type} is not supported.")

    net = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    last_epoch = -1
    base_net_lr = cfg.base_net_lr if cfg.base_net_lr is not None else cfg.lr
    extra_layers_lr = cfg.extra_layers_lr if cfg.extra_layers_lr is not None else cfg.lr
    if cfg.freeze_base_net:
        logger.info("Freeze base net.")
        freeze_net_layers(net.backbone.features)
        params = [
            {'params': net.backbone.extra.parameters(), 'lr': extra_layers_lr},
            {'params': net.head.parameters()}
        ]
    elif cfg.freeze_net:
        freeze_net_layers(net.backbone)
        params = net.head.parameters()
        logger.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.backbone.features.parameters(), 'lr': base_net_lr},
            {'params': net.backbone.extra.parameters(), 'lr': extra_layers_lr},
            {'params': net.head.parameters()}
        ]
    
    if cfg.scheduler == 'multi-step':
        logger.info("Uses MultiStepLR scheduler.")
    elif cfg.scheduler == 'cosine':
        logger.info("Uses CosineAnnealingLR scheduler.")
    else:
        logger.fatal(f"Unsupported Scheduler: {cfg.scheduler}.")
        sys.exit(1)

    net.to(DEVICE)
    logger.info('predicting baseline using pretrained model')
    baseline_dataset = CityscapesDataset(rootdir=cfg.validation_dataset, city=cfg.city, labeldir='labels_coco91', 
                                        labelfile='preprocess/coco-paper-labels.txt', 
                                        transform=train_transform, target_transform=target_transform, mode='ALL')
    cfg_fn_wo_extension = os.path.splitext(os.path.basename(args.config))
    exit()
    predict(net=net, dataset=baseline_dataset, savepath=f'evaluate/{cfg_fn_wo_extension}_baseline.csv')
    logger.info('predicting first window using pretrained model')
    predict(net=net, dataset=val_dataset, savepath=f'evaluate/{cfg_fn_wo_extension}_{val_dataset.cur_window + 1}_{val_dataset.num_window}.csv')
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    while train_dataset.next_window() and val_dataset.next_window():
        if cfg.scheduler == 'multi-step':
            milestones = [int(v.strip()) for v in cfg.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=cfg.gamma, last_epoch=-1)
        elif cfg.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.t_max, last_epoch=last_epoch)
        
        train(net=net, dataset=train_dataset, optimizer=optimizer, scheduler=scheduler, device=DEVICE)
        predict(net=net, dataset=val_dataset, savepath=f'evaluate/{cfg_fn_wo_extension}_{val_dataset.cur_window + 1}_{val_dataset.num_window}.csv')
        model_path = os.path.join(cfg.checkpoint_folder, f"ssdlite320-mb3L_{cfg_fn_wo_extension}_{val_dataset.cur_window + 1}_{val_dataset.num_window}.pth")
        torch.save(net.state_dict(), model_path)
        logger.info(f"Saved model {model_path}")
