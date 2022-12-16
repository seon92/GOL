import numpy as np

import torch
from torch.utils.data import Dataset
from utils.util import load_one_image


class OLHard_Train(Dataset):
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

    def __getitem__(self, item):
        order_label, ref_idx = self.find_reference(self.ranks[item], self.ranks, min_rank=self.min_rank,
                                                   max_rank=self.max_rank)
        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
            ref_img = np.asarray(load_one_image(self.imgs[ref_idx])).astype('uint8')
        else:
            base_img = np.asarray(self.imgs[item]).astype('uint8')
            ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        base_img = self.transform(base_img)
        ref_img = self.transform(ref_img)

        base_age = self.labels[item]
        ref_age = self.labels[ref_idx]

        # gt ranks
        base_rank = self.ranks[item]
        ref_rank = self.ranks[ref_idx]

        return base_img, ref_img, order_label, [base_rank, ref_rank], item

    def __len__(self):
        return self.n_imgs

    def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):

        def get_indices_in_range(search_range, ages):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))

        rng = np.random.default_rng()
        order = np.random.randint(0, 3)
        ref_idx = -1
        debug_flag = 0
        while ref_idx == -1:
            if debug_flag == 3:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if order == 0:  # base_rank > ref_rank + tau
                ref_range_min = max(min_rank, base_rank - 2*self.tau - epsilon)
                ref_range_max = base_rank - self.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1:  # base_rank < ref_rank - tau
                ref_range_min = base_rank + self.tau + epsilon
                ref_range_max = min(max_rank, base_rank + 2*self.tau + epsilon)
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # |base_rank - ref_rank| <= tau
                ref_range_min = base_rank - self.tau - epsilon
                ref_range_max = base_rank + self.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx