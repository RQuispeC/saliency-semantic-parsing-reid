from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

import cv2
import random

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_numpy_file(file_path):
    """Keep reading file until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_file = False
    if not osp.exists(file_path):
        raise IOError("{} does not exist".format(file_path))
    while not got_file:
        try:
            file = np.load(file_path)
            got_file = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(file_path))
            pass
    return file

def get_image_name(img_path):
    name = img_path.split('/')[-1] #get name
    name = name[:name.rfind('.')]  #delete extention
    return name

def decode_parsing(mask):
    body_masks = torch.zeros(5, mask.size()[0], mask.size()[1])
    #foreground
    body_masks[0, mask > 0] = 1

    #head = hat, hair, sunglasses, coat, face
    body_masks[1, mask == 1] = 1
    body_masks[1, mask == 2] = 1
    body_masks[1, mask == 4] = 1
    body_masks[1, mask == 7] = 1
    body_masks[1, mask == 13] = 1
    
    #upper body = Upperclothes, dress, jumpsuits, leftarm, rightarm, 
    body_masks[2, mask == 5] = 1
    body_masks[2, mask == 6] = 1
    body_masks[2, mask == 10] = 1
    body_masks[2, mask == 14] = 1
    body_masks[2, mask == 15] = 1
    
    #lower_body = pants, skirt, leftLeg, rightLeg
    body_masks[3, mask == 9] = 1
    body_masks[3, mask == 12] = 1
    body_masks[3, mask == 16] = 1
    body_masks[3, mask == 17] = 1
    
    #shoes = socks, leftshoe, rightshoe
    body_masks[4, mask == 8] = 1
    body_masks[4, mask == 18] = 1
    body_masks[4, mask == 19] = 1

    body_masks = body_masks.numpy()

    #resize to half of the original images
    masks = []
    for body_mask in body_masks:
        masks.append(cv2.resize(body_mask, (64, 128)))
    return np.array(masks)

def decode_parsing_batch(batch):
    '''
    creates 5 parsing masks for each element in the batch
    '''
    batch_mask = []
    for img in batch:
        batch_mask.append(decode_parsing(img))
    return torch.tensor(batch_mask)

def preprocess_salience(img):
    '''
    Resizes each image to 64 x 128 so it can be used inside architectures
    '''
    img = cv2.resize(img, (64, 128))
    return img

def plot_parsing_augmentation(filename, img, img_trans, img_par, img_par_trans):
    import pylab as plt 
    import gc
    fig = plt.figure(figsize = (20, 30))

    a = fig.add_subplot(2, 6, 1)
    plt.imshow(img)
    a = fig.add_subplot(2, 6, 7)
    plt.imshow(img_trans)
    for i, elem in enumerate(img_par):
        a = fig.add_subplot(2, 6, 2 + i)
        plt.imshow(elem)

    for i, elem in enumerate(img_par_trans):
        a = fig.add_subplot(2, 6, 8 + i)
        plt.imshow(elem)

    fig.savefig('log-fix/figs/parsing/' + filename + '.jpg', bbox_inches='tight')
    fig.clf()
    plt.close()
    del a
    gc.collect()

def plot_salience_augmentation(filename, img, img_trans, img_sal, img_sal_trans):
    import pylab as plt 
    import gc
    fig = plt.figure(figsize = (20, 20))

    a = fig.add_subplot(2, 2, 1)
    plt.imshow(img)
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(img_trans)
    a = fig.add_subplot(2, 2, 3)
    plt.imshow(img_sal)
    a = fig.add_subplot(2, 2, 4)
    plt.imshow(img_sal_trans)

    fig.savefig('log-fix/figs/salience/' + filename + '.jpg', bbox_inches='tight')
    fig.clf()
    plt.close()
    del a
    gc.collect()

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, salience_base_path = 'salience/', use_salience = False, parsing_base_path = 'parsing/', use_parsing = False, transform_salience_parsing = None):
        self.dataset = dataset
        self.transform = transform
        self.use_salience = use_salience
        self.use_parsing = use_parsing
        self.salience_base_path = salience_base_path
        self.parsing_base_path = parsing_base_path
        self.transform_salience_parsing = transform_salience_parsing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        seed = random.randint(0,2**32)
        if self.transform is not None:
            random.seed(seed)
            img = self.transform(img)

        if self.use_salience and not self.use_parsing:
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
            else:
                random.seed(seed)
                salience_img = self.transform_salience_parsing(Image.fromarray(read_numpy_file(salience_path)))
                salience_img = salience_img.resize((64, 128), Image.BILINEAR)
                salience_img = np.array(salience_img)

            return img, pid, camid, salience_img, img_path
        elif not self.use_salience and self.use_parsing:
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))

            if self.transform_salience_parsing != None:
                new_parsing_img = []
                for slide in parsing_img:
                    random.seed(seed)
                    img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    img_i = img_i.resize((64, 128), Image.BILINEAR)
                    img_i = np.array(img_i)
                    new_parsing_img.append(img_i)
                parsing_img = np.array(new_parsing_img)

            return img, pid, camid, parsing_img, img_path

        elif self.use_parsing and self.use_salience:
            parsing_path = osp.join(self.parsing_base_path, get_image_name(img_path) + '.npy')
            salience_path = osp.join(self.salience_base_path, get_image_name(img_path) + '.npy')

            if self.transform_salience_parsing == None:
                salience_img = preprocess_salience(read_numpy_file(salience_path))
                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
            else:
                random.seed(seed)
                salience_img = self.transform_salience(Image.fromarray(read_numpy_file(salience_path)))
                salience_img = salience_img.resize((64, 128), Image.BILINEAR)
                salience_img = np.array(salience_img)

                parsing_img = decode_parsing(torch.tensor(read_numpy_file(parsing_path)))
                new_parsing_img = []
                for slide in parsing_img:
                    random.seed(seed)
                    img_i = self.transform_salience_parsing(Image.fromarray(slide))
                    img_i = img_i.resize((64, 128), Image.BILINEAR)
                    img_i = np.array(img_i)
                    new_parsing_img.append(img_i)
                parsing_img = np.array(new_parsing_img)

            return img, pid, camid, salience_img, parsing_img, img_path
        else:
            return img, pid, camid, img_path

class VideoDataset(Dataset):
    """
    Code imported from https://github.com/KaiyangZhou/deep-person-reid
    """
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order
            # comment it to be order-agnostic
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """Evenly sample seq_len items from num items."""
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num/self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid