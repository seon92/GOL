import numpy as np

import torch
from torch.utils.data import Dataset
from utils.util import load_one_image

class OLBasic_Val(Dataset):
    def __init__(self, base_data, ref_data, transform, tau, norm_age=True, is_filelist=False):
        super(Dataset, self).__init__()
        self.transform = transform
        self.ref_imgs, self.ref_labels = ref_data
        self.base_imgs, self.base_labels = base_data

        self.n_base_imgs = len(self.base_imgs)
        self.n_ref_imgs = len(self.ref_imgs)
        self.tau = tau
        self.is_filelist = is_filelist

        if norm_age:
            self.base_labels = self.base_labels - min(self.base_labels)
            self.ref_labels = self.ref_labels - min(self.ref_labels)

    def __getitem__(self, item):
        ref_idx = np.random.choice(self.n_ref_imgs, 1)[0]
        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.base_imgs[item])).astype('uint8')
            ref_img = np.asarray(load_one_image(self.ref_imgs[ref_idx])).astype('uint8')
        else:
            base_img = np.asarray(self.base_imgs[item]).astype('uint8')
            ref_img = np.asarray(self.ref_imgs[ref_idx]).astype('uint8')

        base_img = self.transform(base_img)
        ref_img = self.transform(ref_img)

        # order label generation
        order_labels = self.get_order_labels(item, ref_idx)

        # gt ages
        # base_age = self.base_labels[item]
        # ref_age = self.ref_labels[ref_idx]
        return base_img, ref_img, order_labels, item

    def __len__(self):
        return len(self.base_imgs)

    def get_order_labels(self, base_idx, ref_idx):
        base_ranks = self.base_labels[base_idx]
        ref_ranks = self.ref_labels[ref_idx]

        if base_ranks > ref_ranks + self.tau:
            order_labels = 0
        elif base_ranks < ref_ranks - self.tau:
            order_labels = 1
        else:
            order_labels = 2
        return order_labels