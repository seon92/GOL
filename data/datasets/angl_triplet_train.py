import numpy as np

import torch
from torch.utils.data import Dataset
from utils.util import load_one_image


class AngleTriplet_Train(Dataset):
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

        self.hardness_multiplier = 20  # thus, interval is 4 (=5-1) tau

    def __getitem__(self, item):
        order_label, ref_idx = self.find_reference_triplet(self.labels[item], self.labels, min_rank=self.min_age,
                                                   max_rank=self.max_age)
        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
            ref_img1 = np.asarray(load_one_image(self.imgs[ref_idx[0]])).astype('uint8')
            ref_img2 = np.asarray(load_one_image(self.imgs[ref_idx[1]])).astype('uint8')
        else:
            base_img = np.asarray(self.imgs[item]).astype('uint8')
            ref_img1 = np.asarray(self.imgs[ref_idx[0]]).astype('uint8')
            ref_img2 = np.asarray(self.imgs[ref_idx[1]]).astype('uint8')
        base_img = self.transform(base_img)
        ref_img1 = self.transform(ref_img1)
        ref_img2 = self.transform(ref_img2)

        base_age = self.labels[item]
        ref_age1 = self.labels[ref_idx[0]]
        ref_age2 = self.labels[ref_idx[1]]

        # gt ranks
        base_rank = self.ranks[item]
        ref_rank1 = self.ranks[ref_idx[0]]
        ref_rank2 = self.ranks[ref_idx[1]]

        return base_img, ref_img1, ref_img2, order_label, [base_age, ref_age1, ref_age2, base_rank, ref_rank1, ref_rank2], item

    def __len__(self):
        return self.n_imgs

    def find_reference_triplet(self, base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):

        def get_indices_in_range(search_range, ages):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))

        rng = np.random.default_rng()
        case = rng.choice([0,1], 1, p=[2/3, 1/3])
        ref_idx2 = -1
        debug_flag = 0
        while ref_idx2 == -1:
            if debug_flag == 2:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if case == 0:  # ref1_rank + tau < base_rank < ref2_rank - tau
                ref_range_min = max(base_rank - (self.tau*self.hardness_multiplier), min_rank)
                ref_range_max = base_rank - self.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx1 = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    case = (case + 1) % 2
                    debug_flag += 1
                    continue

                ref_range_min = base_rank + self.tau + epsilon
                ref_range_max = min(base_rank + (self.tau*self.hardness_multiplier), max_rank)
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx2 = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    case = (case + 1) % 2
                    debug_flag += 1
                    continue

            elif case == 1:  # |base_rank - ref_rank| <= tau
                ref_range_min = base_rank - self.tau - epsilon
                ref_range_max = base_rank + self.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx1 = candidates[rng.choice(len(candidates), 1)[0]][0]
                    ref_idx2 = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    case = (case + 1) % 2
                    debug_flag += 1
            else:
                raise ValueError(f'[!] something is wrong... base rank{base_rank}, case{case}')

        if case == 0:
            order = [0, 1]
        else:
            order = [2, 2]

        return order, [ref_idx1, ref_idx2]