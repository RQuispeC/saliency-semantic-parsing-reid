import numpy
import cv2
import os.path as osp
import os
import pylab as plt 
import gc
import argparse

import data_manager
import models


parser = argparse.ArgumentParser(description='Plot rank-5 results of S-ReID, SP-ReID and SSP-ReID')

parser.add_argument('-d', '--dataset', type=str, default='market1501')
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50')

args = parser.parse_args()

def plot(images, save_name):
    num_figs = len(images)
    fig = plt.figure(figsize = (30, 20))
    for i, img in enumerate(images):
        a = fig.add_subplot(num_figs, 1, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    fig.savefig(save_name, bbox_inches='tight')
    fig.clf()
    plt.close()
    del a
    gc.collect()

def combine_fig(file_name, salience_dir, parsing_dir, salience_parsing_dir, save_dir):
    salience_file = osp.join(salience_dir, file_name)
    parsing_file = osp.join(parsing_dir, file_name)
    salience_parsing_file = osp.join(salience_parsing_dir, file_name)
    save_file = osp.join(save_dir, file_name)

    images = [cv2.imread(salience_file), cv2.imread(parsing_file), cv2.imread(salience_parsing_file)]
    plot(images, save_file)

def main():
    dataset = args.dataset
    model = args.arch
    salience_dir = osp.join('log/', '{}-salience-{}/-1'.format(model, dataset))
    parsing_dir = osp.join('log/', '{}-parsing-{}/-1'.format(model, dataset))
    salience_parsing_dir = osp.join('log/', '{}-salience-parsing-{}/-1'.format(model, dataset))
    save_dir = osp.join('log/', '{}-improvement-{}'.format(model, dataset))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    list_figs = os.listdir(salience_dir)
    for img_name in list_figs:
        combine_fig(img_name, salience_dir, parsing_dir, salience_parsing_dir, save_dir)

if __name__ == '__main__':
    main()