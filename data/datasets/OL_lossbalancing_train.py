import numpy as np

import torch
from torch.utils.data import Dataset
from utils.util import load_one_image


class OLLossBalancing_Train(Dataset):
    def __init__(self, imgs, labels, transform, tau, norm_age=True, logscale=False, is_filelist=False):
        super(Dataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.n_imgs = len(self.imgs)
        self.min_age_bf_norm = self.labels.min()
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
        self.n_ranks = self.ranks.max()+1
        self.probs = None
        self.sample_idxs_per_rank = dict()
        for r in range(self.n_ranks):
            self.sample_idxs_per_rank[r] = np.argwhere(self.ranks==r).flatten()

    def __getitem__(self, item):
        if self.probs is not None:
            rng = np.random.default_rng()
            base_rank = rng.choice(np.arange(self.n_ranks), 1, p=self.probs)[0]
            base_idx = rng.choice(self.sample_idxs_per_rank[base_rank], 1)[0]
        else:
            base_idx = item
        order_label, ref_idx = self.find_reference(self.labels[base_idx], self.labels, min_rank=self.min_age,
                                                   max_rank=self.max_age)
        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[base_idx])).astype('uint8')
            ref_img = np.asarray(load_one_image(self.imgs[ref_idx])).astype('uint8')
        else:
            base_img = np.asarray(self.imgs[base_idx]).astype('uint8')
            ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        base_img = self.transform(base_img)
        ref_img = self.transform(ref_img)

        # gt ranks
        base_rank = self.ranks[base_idx]
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
                ref_range_min = min_rank
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
                ref_range_max = max_rank
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

    def update_probs(self, loss_record):
        self.probs = np.exp(loss_record) / np.sum(np.exp(loss_record))