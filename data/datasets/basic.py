import numpy as np

import torch
from torch.utils.data import Dataset
from utils.util import load_one_image

class Basic(Dataset):
    def __init__(self, imgs, labels, transform, norm_age=True, is_filelist=False, return_ranks=False, std=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.imgs = imgs
        self.labels = labels

        self.n_imgs = len(self.imgs)
        self.is_filelist = is_filelist
        if norm_age:
            self.labels = self.labels - min(self.labels)
        self.return_ranks = return_ranks
        self.std = std

        rank = 0
        self.mapping = dict()
        for cls in np.unique(self.labels):
            self.mapping[cls] = rank
            rank += 1
        self.ranks = np.array([self.mapping[l] for l in self.labels])



    def __getitem__(self, item):
        if self.is_filelist:
            img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
        else:
            img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)

        if self.return_ranks:
            return img, self.labels[item], self.ranks[item], item
        else:
            return img, self.labels[item], item

    def __len__(self):
        return len(self.imgs)
