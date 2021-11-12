import argparse
import os
import logging
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


def predict(net, dataset, device, savepath, args):
    print('predicting...')
    net.eval()
    
    df = pd.DataFrame(columns=['image', 'prediction', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    loader = DataLoader(dataset, batch_size=4, num_workers=args.num_workers, shuffle=False, collate_fn=lambda a: list(map(list, zip(*a))))

    for _, data in enumerate(loader):
        ids, images, _, _ = data
        images = torch.stack(images).to(device)
        predictions = net(images)
        for id, pred in zip(ids, predictions):
            indices = torch.nonzero(pred['scores'] >= 0.5).squeeze(1)
            labels = pred['labels'][indices].detach().numpy()
            boxes = pred['boxes'][indices].detach().numpy()
            scores = pred['scores'][indices].detach().numpy()
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
                print(df)

    df.to_csv(savepath, index=None)


def test(net, dataset, criterion, device, savepath):
    print(f'testing...')
    df = pd.DataFrame(columns=['image', 'prediction', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    loader = DataLoader(dataset, batch_size=4, num_workers=args.num_workers, shuffle=False)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for _num, data in enumerate(loader):
        ids, images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            """
            confidence, locations = net(images)
            confidence = confidence.squeeze(0)
            locations = locations.squeeze(0)
            predictions = torch.argmax(confidence, dim=1)
            scores = confidence[torch.arange(confidence.shape[0]), predictions]
            # print(images.shape, confidence.shape, locations.shape, predictions.shape, scores.shape)
            # scores = torch.gather(input=confidence, dim=-1, index=predictions).squeeze(-1)
            masks = scores > 0.2
            indices = torch.nonzero(masks).squeeze(1)
            # print(indices.shape)
            predictions = predictions[indices]
            scores = scores[indices]
            locations = locations[indices]
            """
            confidence, locations = net(images)
            predictions = torch.argmax(confidence, dim=-1, keepdim=True)
            scores = torch.gather(input=confidence, dim=-1, index=predictions).squeeze(-1)
            # print(confidence, scores)
            # exit()
            print(predictions.shape, scores.shape)
            masks = scores > 0.5
            indices = torch.nonzero(masks)
            print(masks.shape, indices.shape)
            indices0, indices1 = indices.split(1, dim=1)
            print(indices0.shape, indices1.shape)
            scores = scores[indices0, indices1]
            predictions = predictions.squeeze(-1)[indices0, indices1]
            locations = locations[indices0, indices1, :]
            print(scores.shape, predictions.shape, locations.shape)
            imageids = [ids[ind0] for ind0 in indices0.cpu().detach().numpy()]
        
            # exit()
        for idx in range(predictions.shape[0]):
            location = locations.cpu()[idx].detach().numpy()
            prediction = predictions.cpu()[idx].detach().numpy()
            score = scores.cpu()[idx]
            df.append({
                'image': ids[0], 
                'prediction': prediction, 
                'score': score, 
                'xmin': location[0], 
                'ymin': location[1],
                'xmax': location[2],
                'ymax': location[3]
            }, ignore_index=True)

            # exit(0)
            regression_loss, classification_loss = criterion(confidence.unsqueeze(0), locations.unsqueeze(0), labels, boxes)
            loss = regression_loss + classification_loss
        if _num % 20 == 0:
            print(f'finish {_num} evaluations')
        # running_loss += loss.item()
        # running_regression_loss += regression_loss.item()
        # running_classification_loss += classification_loss.item()
    df.to_csv(savepath)
    print(f'save results to {savepath}')
    # return running_loss / len(dataset), running_regression_loss / len(dataset), running_classification_loss / len(dataset)


def train(net, dataset, optimizer, device, debug_steps=10):
    print('training...')
    net.train()
    index = 0
    loader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=lambda a: list(map(list, zip(*a))))
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for _, data in enumerate(loader):
            index += 1
            _, images, boxes, labels = data
            images = torch.stack(images).to(device)
            targets = [{
                'boxes': torch.tensor(box, device=device),
                'labels': torch.tensor(label, device=device)
            } for box, label in zip(boxes, labels)]
            optimizer.zero_grad()
            res = net(images, targets)
            loss = res['bbox_regression'] + res['classification']
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if index and index % debug_steps == 0:
                logging.info(f"Epoch: {epoch}, Global Step: {index}, Average Loss: {running_loss / debug_steps:.4f}")
                running_loss = 0.0



if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = arg_parser()
    args = parser.parse_args()
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")
    timer = Timer()

    logging.info(args)
    if args.net == 'vgg16-ssd':
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    elif args.net == 'mb3-large-ssd-lite':
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    elif args.net == 'mb3-small-ssd-lite':
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

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

    assert len(args.datasets) == 1, "continual learning not supports multiply dataset"
    
    dataset_path = args.datasets[0]
    if args.dataset_type == 'city_scapes':
        train_dataset = ContinualCityscapesDataset(dataset_path, city=args.city, labeldir='labels_coco91', labelfile='preprocess/coco-paper-labels.txt',
                                            transform=train_transform, target_transform=target_transform, mode='TRAIN')
        val_dataset = ContinualCityscapesDataset(args.validation_dataset, city=args.city, labeldir='labels_coco91', labelfile='preprocess/coco-paper-labels.txt',
                                                transform=test_transform, target_transform=target_transform, mode='TEST')
        num_classes = len(train_dataset.class_names)
    else:
        raise ValueError(f"Dataset type {args.dataset_type} is not supported.")

    net = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.backbone.features)
        params = [
            {'params': net.backbone.extra.parameters(), 'lr': extra_layers_lr},
            {'params': net.head.parameters()}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.backbone)
        params = net.head.parameters()
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.backbone.features.parameters(), 'lr': base_net_lr},
            {'params': net.backbone.extra.parameters(), 'lr': extra_layers_lr},
            {'params': net.head.parameters()}
        ]

    net.to(DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    logging.info(f"Start training from epoch {last_epoch + 1}.")

    baseline_dataset = CityscapesDataset(rootdir='data', city=args.city, labeldir='labels_coco91', labelfile='preprocess/coco-paper-labels.txt', 
                                        transform=train_transform, target_transform=target_transform, mode='ALL')
    
    # predict(net, baseline_dataset, DEVICE, f'evaluate/continual_{args.city}_baseline.csv', args)

    # predict(net, val_dataset, DEVICE, f'evaluate/continual_{args.city}_{val_dataset.cur_window + 1}_{val_dataset.num_window}.csv', args)
    while train_dataset.next_window() and val_dataset.next_window():
        train(net, train_dataset, optimizer, DEVICE)
        predict(net, val_dataset, DEVICE, f'evaluate/continual_{args.city}_{val_dataset.cur_window + 1}_{val_dataset.num_window}.csv', args)
        model_path = os.path.join(args.checkpoint_folder, f"{args.net}_{args.city}_{val_dataset.cur_window + 1}_{val_dataset.num_window}.pth")
        torch.save(net.state_dict(), model_path)
        logging.info(f"Saved model {model_path}")
