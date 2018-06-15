from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from optimizers import init_optim
from re_ranking_ranklist import re_ranking

import cv2


parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=180, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=60, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--use-salience', action='store_true', help="use salience maps for net training")
parser.add_argument('--use-parsing', action='store_true', help="use semantic parsing for net training")
parser.add_argument('--use-re-ranking', action='store_true', help="use re-ranking before evauating results")
parser.add_argument('--save-rank', action='store_true', help="save top ranked results for each query")


args = parser.parse_args()

def get_distance_matrix(X, Y):
    '''
    Gets the distance of between each element of X and Y    
   
    input:

    X: tensor of features
    Y: tensor of features

    output: 

    distmat: Matrix(numpy) of size |X|*|Y| with the distances between each element of X and Y
    '''
    m, n = X.size(0), Y.size(0)
    distmat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(Y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, X, Y.t())
    distmat = distmat.numpy()
    return distmat

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(
        name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False
    use_salience = True if args.use_salience else False
    use_parsing = True if args.use_parsing else False
    save_rank = True if args.save_rank else False
    use_re_ranking = True if args.use_re_ranking else False

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train, use_salience = use_salience, use_parsing = use_parsing, salience_base_path = dataset.salience_train_dir, parsing_base_path = dataset.parsing_train_dir),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test, use_salience = use_salience, use_parsing = use_parsing, salience_base_path = dataset.salience_query_dir, parsing_base_path = dataset.parsing_query_dir),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test, use_salience = use_salience, use_parsing = use_parsing, salience_base_path = dataset.salience_gallery_dir, parsing_base_path = dataset.parsing_gallery_dir),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)
    
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("Resuming from epoch {}".format(start_epoch + 1))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu, use_salience = use_salience, use_parsing = use_parsing, save_dir = args.save_dir, epoch = -1, save_rank = save_rank, use_re_ranking = use_re_ranking)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, use_salience, use_parsing)
        train_time += round(time.time() - start_train_time)
        
        if args.stepsize > 0: scheduler.step()
        
        if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu, use_salience = use_salience, use_parsing = use_parsing, save_dir = args.save_dir, epoch = epoch, save_rank = save_rank, use_re_ranking = use_re_ranking)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, use_salience, use_parsing):
    model.train()
    losses = AverageMeter()

    for batch_idx, tuple_i in enumerate(trainloader):
        if use_salience and not use_parsing:
            imgs, pids, _, salience_imgs, _ = tuple_i
        elif not use_salience and use_parsing:
            imgs, pids, _, parsing_imgs, _ = tuple_i
        elif use_salience and use_parsing:
            imgs, pids, _, salience_imgs, parsing_imgs, _ = tuple_i
        else:
            imgs, pids, _, _ = tuple_i

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        
        if use_salience and not use_parsing:
            outputs, features = model(imgs, salience_masks = salience_imgs)
        elif not use_salience and use_parsing:
            outputs, features = model(imgs, parsing_masks = parsing_imgs)
        elif use_salience and use_parsing:
            outputs, features = model(imgs, salience_masks = salience_imgs, parsing_masks = parsing_imgs)
        else:
            outputs, features = model(imgs)
        
        if args.htri_only:
            if isinstance(features, tuple):
                loss = DeepSupervision(criterion_htri, features, pids)
            else:
                loss = criterion_htri(features, pids)
        else:
            if isinstance(outputs, tuple):
                xent_loss = DeepSupervision(criterion_xent, outputs, pids)
            else:
                xent_loss = criterion_xent(outputs, pids)
            
            if isinstance(features, tuple):
                htri_loss = DeepSupervision(criterion_htri, features, pids)
            else:
                htri_loss = criterion_htri(features, pids)
            
            loss = xent_loss + htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Epoch {}/{}\t Batch {}/{}\t Loss {:.6f} ({:.6f})".format(
                epoch+1, args.max_epoch, batch_idx+1, len(trainloader), losses.val, losses.avg
            ))

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], use_salience = False, use_parsing = False, save_dir = "", epoch = -1, save_rank = False, use_re_ranking = False):
    model.eval()
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        qimgs = []
        for batch_idx, tuple_i in enumerate(queryloader):
            if use_salience and not use_parsing:
                imgs, pids, camids, salience_imgs, qimg = tuple_i
            elif not use_salience and use_parsing:
                imgs, pids, camids, parsing_imgs, qimg = tuple_i
            elif use_salience and use_parsing:
                imgs, pids, camids, salience_imgs, parsing_imgs, qimg = tuple_i
            else:
                imgs, pids, camids, qimg = tuple_i
            
            if use_gpu:
                imgs = imgs.cuda()
            
            if use_salience and not use_parsing:
                features = model(imgs, salience_masks = salience_imgs)
            elif not use_salience and use_parsing:
                features = model(imgs, parsing_masks = parsing_imgs)
            elif use_salience and use_parsing:
                features = model(imgs, salience_masks = salience_imgs, parsing_masks = parsing_imgs)
            else:
                features = model(imgs)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            qimgs.extend(qimg)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        qimgs = np.asarray(qimgs)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        gimgs = []
        for batch_idx, tuple_i in enumerate(galleryloader):
            if use_salience and not use_parsing:
                imgs, pids, camids, salience_imgs, gimg = tuple_i
            elif not use_salience and use_parsing:
                imgs, pids, camids, parsing_imgs, gimg = tuple_i
            elif use_salience and use_parsing:
                imgs, pids, camids, salience_imgs, parsing_imgs, gimg = tuple_i
            else:
                imgs, pids, camids, gimg = tuple_i

            if use_gpu:
                imgs = imgs.cuda()

            if use_salience and not use_parsing:
                features = model(imgs, salience_masks = salience_imgs)
            elif not use_salience and use_parsing:
                features = model(imgs, parsing_masks = parsing_imgs)
            elif use_salience and use_parsing:
                features = model(imgs, salience_masks = salience_imgs, parsing_masks = parsing_imgs)
            else:
                features = model(imgs)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            gimgs.extend(gimg)
            
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        gimgs = np.asarray(gimgs)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    
    print("Computing distance matrix")

    if use_re_ranking:
        qg_distmat = get_distance_matrix(qf, gf)
        gg_distmat = get_distance_matrix(gf, gf)
        qq_distmat = get_distance_matrix(qf, qf)
        
        print("Re-ranking feature maps")
        distmat = re_ranking(qg_distmat, qq_distmat, gg_distmat)
    else:
        distmat = get_distance_matrix(qf, gf)

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03, query_img_paths = qimgs, gallery_img_paths = gimgs, save_dir = save_dir, epoch = epoch, save_rank = save_rank)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")
    print("{:.1%}".format(mAP))
    for r in ranks:
        print("{:.1%}".format(cmc[r-1]))
    print("------------------")    

    return cmc[0]

if __name__ == '__main__':
    main()