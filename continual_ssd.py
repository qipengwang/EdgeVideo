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


def config_parameters(net, args):
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {
                'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ),
                'lr': extra_layers_lr
            },
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {
                'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ),
                'lr': extra_layers_lr
            },
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    
    return params


def predict(net, dataset, device, savepath, args):
    print('predicting...')
    net.test(True)

    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    elif args.net == 'mb2-ssd-lite' or args.net == "mb3-large-ssd-lite" or args.net == "mb3-small-ssd-lite":
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        logging.fatal("The net type is wrong.")
        sys.exit(1)
    
    df = pd.DataFrame(columns=['image', 'prediction', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    loader = DataLoader(dataset, batch_size=4, num_workers=args.num_workers, shuffle=False)
    _num = 0
    for _, data in enumerate(loader):
        ids, _, _, _ = data
        for id in ids:
            orig_image = cv2.imread(os.path.join(dataset.image_rootdir, id))
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                df_row = {
                    'image': ids[0], 
                    'prediction': labels.numpy()[i], 
                    'score': probs.numpy()[i], 
                    'xmin': box.numpy()[0], 
                    'ymin': box.numpy()[1],
                    'xmax': box.numpy()[2],
                    'ymax': box.numpy()[3]
                }
                df = df.append(df_row, ignore_index=True)
                # print(df)
    df.to_csv(savepath, index=None)


def test(net, dataset, criterion, device, savepath):
    print(f'testing...')
    df = pd.DataFrame(columns=['image', 'prediction', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    loader = DataLoader(dataset, batch_size=4, num_workers=args.num_workers, shuffle=False)
    net.test(True)
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


def train(net, dataset, criterion, optimizer, device, debug_steps=10):
    print('training...')
    net.test(False)
    index = 0
    loader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        for _, data in enumerate(loader):
            index += 1
            _, images, boxes, labels = data
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
            loss = regression_loss + classification_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            if index and index % debug_steps == 0:
                avg_loss = running_loss / debug_steps
                avg_reg_loss = running_regression_loss / debug_steps
                avg_clf_loss = running_classification_loss / debug_steps
                logging.info(
                    f"Epoch: {epoch}, Global Step: {index}, " +
                    f"Average Loss: {avg_loss:.4f}, " +
                    f"Average Regression Loss {avg_reg_loss:.4f}, " +
                    f"Average Classification Loss: {avg_clf_loss:.4f}"
                )
                running_loss = 0.0
                running_regression_loss = 0.0
                running_classification_loss = 0.0



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
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    elif args.net == 'mb3-large-ssd-lite':
        create_net = lambda num: create_mobilenetv3_large_ssd_lite(num)
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    elif args.net == 'mb3-small-ssd-lite':
        create_net = lambda num: create_mobilenetv3_small_ssd_lite(num)
        config = mobilenetv3_ssd320_config if args.ssd320 else mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    # net = create_net(21)
    # net.train()
    # output = net(torch.rand(4, 3, 300, 300))
    # print(output[0].shape, output[1].shape)
    # exit(0)

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)


    assert len(args.datasets) == 1, "continual learning not supports multiply dataset"
    dataset_path = args.datasets[0]
    if args.dataset_type == 'city_scapes':
        train_dataset = ContinualCityscapesDataset(dataset_path, city=args.city, labeldir='labels_voc', transform=train_transform,
                                            target_transform=target_transform, mode='TRAIN')
        val_dataset = ContinualCityscapesDataset(args.validation_dataset, city=args.city, labeldir='labels_voc', transform=test_transform, 
                                                target_transform=target_transform, mode='TEST')
        label_file = os.path.join(args.checkpoint_folder, "cityscapes-labels.txt")
        store_labels(label_file, train_dataset.class_names)
        num_classes = len(train_dataset.class_names)
    else:
        raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
    # logging.info("Train dataset size: {}".format(len(train_dataset)))
    # logging.info("Validation dataset size: {}".format(len(val_dataset)))

    # net = create_net(num_classes)
    net = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    min_loss = -10000.0
    last_epoch = -1
    params = config_parameters(net=net, args=args)
    
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
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

    baseline_dataset = CityscapesDataset(rootdir='data', city=args.city, transform=train_transform, target_transform=target_transform, mode='ALL')
    # predict(net, baseline_dataset, DEVICE, f'evaluate/continual_{args.city}_baseline.csv', args)

    # train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)
    # val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)
    predict(net, val_dataset, DEVICE, f'evaluate/continual_{args.city}_{val_dataset.cur_window + 1}_{val_dataset.num_window}.csv', args)
    while train_dataset.next_window() and val_dataset.next_window():
        train(net, train_dataset, criterion, optimizer, DEVICE)
        predict(net, val_dataset, DEVICE, f'evaluate/continual_{args.city}_{val_dataset.cur_window + 1}_{val_dataset.num_window}.csv', args)
        model_path = os.path.join(args.checkpoint_folder, f"{args.net}_{args.city}_{val_dataset.cur_window + 1}_{val_dataset.num_window}.pth")
        net.save(model_path)
        logging.info(f"Saved model {model_path}")
