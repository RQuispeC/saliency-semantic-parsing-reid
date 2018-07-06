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
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
# Architecture
parser.add_argument('-a1', '--arch1', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('-a2', '--arch2', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume1', type=str, default='', metavar='PATH')
parser.add_argument('--resume2', type=str, default='', metavar='PATH')
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
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

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False
    use_salience = models.use_salience(name=args.arch)
    use_parsing = models.use_parsing(name=args.arch)
    save_rank = True if args.save_rank else False
    use_re_ranking = True if args.use_re_ranking else False

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

    print("Initializing model: {}".format(args.arch1))
    model1 = models.init_model(name=args.arch1, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model1.parameters())/1000000.0))
    print("Initializing model: {}".format(args.arch2))
    model2 = models.init_model(name=args.arch2, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model2.parameters())/1000000.0))

    print("Loading checkpoint from '{}'".format(args.resume1))
    checkpoint = torch.load(args.resume1)
    model1.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    print("Resuming model 1 from epoch {}".format(start_epoch + 1))

    print("Loading checkpoint from '{}'".format(args.resume2))
    checkpoint = torch.load(args.resume2)
    model2.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    print("Resuming model 2 from epoch {}".format(start_epoch + 1))

    if use_gpu:
        model1 = nn.DataParallel(model1).cuda()
        model2 = nn.DataParallel(model2).cuda()

    test(model1, model2, queryloader, galleryloader, use_gpu, use_salience = use_salience, use_parsing = use_parsing, save_dir = args.save_dir, epoch = -1, save_rank = save_rank, use_re_ranking = use_re_ranking)

def test(model1, model2, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], use_salience = False, use_parsing = False, save_dir = "", epoch = -1, save_rank = False, use_re_ranking = False):
    model1.eval()
    model2.eval()
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
                features = model1(imgs, salience_masks = salience_imgs)
            elif not use_salience and use_parsing:
                features = model2(imgs, parsing_masks = parsing_imgs)
            elif use_salience and use_parsing:
                features1 = model1(imgs, salience_masks = salience_imgs)
                features2 = model2(imgs, parsing_masks = parsing_imgs)
                features = torch.cat((features1, features2), dim = 1)
            else:
                exit("invalid flags")
                
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
                features = model1(imgs, salience_masks = salience_imgs)
            elif not use_salience and use_parsing:
                features = model2(imgs, parsing_masks = parsing_imgs)
            elif use_salience and use_parsing:
                features1 = model1(imgs, salience_masks = salience_imgs)
                features2 = model2(imgs, parsing_masks = parsing_imgs)
                features = torch.cat((features1, features2), dim = 1)
            else:
                exit("invalid flags")

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

    return cmc[0]

if __name__ == '__main__':
    main()