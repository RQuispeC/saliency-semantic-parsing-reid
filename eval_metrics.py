from __future__ import print_function, absolute_import
import numpy as np
import copy
from collections import defaultdict
import sys


import pylab as plt
import gc
import os.path as osp
import os

from utils import read_image
from PIL import ImageDraw, Image

def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def plot_rank(qimg_path, gimg_paths, order, matches, save_dir = "", epoch = - 1):
    num_figs = min(5, len(order)) #plot at most top 5 answers
    gimg_paths = gimg_paths[order]
    gimg_paths = gimg_paths[:num_figs]

    rank = 5 #top "rank" images are plotted
    fig = plt.figure(figsize=(30, 20))
    a = fig.add_subplot(1, rank + 1, 1)
    a.set_title("QUERY")
    a.axis("off")
    img = read_image(qimg_path)
    img = img.resize((64, 128), Image.BILINEAR)
    plt.imshow(img)
    for ind, gimg_path in enumerate(gimg_paths):
        a = fig.add_subplot(1, rank + 1, ind + 2)
        message = "CORRECT" if matches[ind] == 1 else "WRONG"
        color = "green" if matches[ind] == 1 else "red"
        a.set_title(message)
        a.axis("off")
        img = read_image(gimg_path)
        img = img.resize((64, 128), Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (img.size[0] - 1, img.size[1] - 1)], outline = color)
        draw.rectangle([(1, 1), (img.size[0] - 2, img.size[1] - 2)], outline = color)
        draw.rectangle([(2, 2), (img.size[0] - 3, img.size[1] - 3)], outline = color)
        plt.imshow(img)
        del draw

    save_dir = osp.join(save_dir, '{}'.format(epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    q_imgname = qimg_path.split('/')[-1]
    
    fig.savefig(os.path.join(save_dir, q_imgname), bbox_inches='tight')

    #free ram 
    fig.clf()
    plt.close()
    del a
    gc.collect()

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, query_img_paths = [], gallery_img_paths = [], save_dir = "", epoch = -1, save_rank = False):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        if save_rank:
            #plot rank results
            plot_rank(query_img_paths[q_idx], gallery_img_paths, order[keep], matches[q_idx][keep], save_dir, epoch)

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False, query_img_paths = [], gallery_img_paths = [], save_dir = "", epoch = -1, save_rank = False):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, query_img_paths, gallery_img_paths, save_dir = save_dir, epoch = epoch, save_rank = save_rank)