import numpy as np

import torch
from torch.utils.data import Dataset
from utils.util import load_one_image


class OLAngle_Train(Dataset):
    def __init__(self, imgs, labels, transform, tau, norm_age=True, logscale=False, is_filelist=False):
        super(Dataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.n_imgs = len(self.imgs)

        if logscale:
            self.labels = np.log(labels.astype(np.float32))
        else:
            if norm_age:
                self.labels = self.labels - min(self.labels)

        self.max_age = self.labels.max()
        self.min_age = self.labels.min()
        self.tau = tau
        self.is_filelist = is_filelist

        # mapping age to rank : because there are omitted ages
        rank = 0
        self.mapping = dict()
        for cls in np.unique(self.labels):
            self.mapping[cls] = rank
            rank += 1
        self.ranks = np.array([self.mapping[l] for l in self.labels])
        self.max_rank = self.ranks.max()
        self.min_rank = self.ranks.min()
        self.margin = 0   # should smaller than tau
        assert self.margin < self.tau

    def __getitem__(self, item):
        left_idx, center_idx, right_idx, pos_idx = self.find_references(item, self.ranks, margin=self.margin)

        if self.is_filelist:
            l_img = np.asarray(load_one_image(self.imgs[left_idx])).astype('uint8')
            c_img = np.asarray(load_one_image(self.imgs[center_idx])).astype('uint8')
            r_img = np.asarray(load_one_image(self.imgs[right_idx])).astype('uint8')
            p_img = np.asarray(load_one_image(self.imgs[pos_idx])).astype('uint8')

        else:
            l_img = np.asarray(self.imgs[left_idx]).astype('uint8')
            c_img = np.asarray(self.imgs[center_idx]).astype('uint8')
            r_img = np.asarray(self.imgs[right_idx]).astype('uint8')
            p_img = np.asarray(self.imgs[pos_idx]).astype('uint8')

        l_img = self.transform(l_img)
        c_img = self.transform(c_img)
        r_img = self.transform(r_img)
        p_img = self.transform(p_img)

        return l_img, c_img, r_img, p_img, [self.ranks[left_idx], self.ranks[center_idx], self.ranks[right_idx], self.ranks[pos_idx]]

    def __len__(self):
        return self.n_imgs

    def find_references(self, item, ranks, margin=1, epsilon=1e-4):

        def get_indices_in_range(search_range, ages):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))

        base_rank = ranks[item]
        rng = np.random.default_rng()

        if base_rank - self.tau - margin < self.min_rank:
            left_idx = item

            # pick center
            candidates = get_indices_in_range([base_rank+self.tau-margin-epsilon, base_rank+self.tau+margin+epsilon], ranks)
            center_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
            center_rank = ranks[center_idx]

            # pick right
            candidates = get_indices_in_range([center_rank + self.tau -margin-epsilon, center_rank + self.tau + margin+epsilon], ranks)
            right_idx = candidates[rng.choice(len(candidates), 1)[0]][0]

            # pick pos
            candidates = get_indices_in_range(
                [center_rank - epsilon, center_rank + epsilon], ranks)
            pos_idx = candidates[rng.choice(len(candidates), 1)[0]][0]

        elif base_rank + self.tau + margin > self.max_rank:
            right_idx = item

            # pick center
            candidates = get_indices_in_range(
                [base_rank - self.tau - margin - epsilon, base_rank - self.tau + margin + epsilon], ranks)
            center_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
            center_rank = ranks[center_idx]

            # pick left
            candidates = get_indices_in_range([center_rank - self.tau - margin - epsilon,
                                               center_rank - self.tau + margin + epsilon], ranks)
            left_idx = candidates[rng.choice(len(candidates), 1)[0]][0]

            # pick pos
            candidates = get_indices_in_range(
                [center_rank - epsilon, center_rank + epsilon], ranks)
            pos_idx = candidates[rng.choice(len(candidates), 1)[0]][0]

        else:
            center_idx = item

            # pick left
            candidates = get_indices_in_range([base_rank - self.tau - margin - epsilon,
                                               base_rank - self.tau + margin + epsilon], ranks)
            left_idx = candidates[rng.choice(len(candidates), 1)[0]][0]

            # pick right
            candidates = get_indices_in_range([base_rank + self.tau - margin - epsilon,
                                               base_rank + self.tau + margin + epsilon], ranks)
            right_idx = candidates[rng.choice(len(candidates), 1)[0]][0]

            # pick pos
            candidates = get_indices_in_range(
                [base_rank - epsilon, base_rank + epsilon], ranks)
            pos_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
        return left_idx, center_idx, right_idx, pos_idx