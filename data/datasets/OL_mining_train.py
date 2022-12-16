import numpy as np

import torch
from torch.utils.data import Dataset
from utils.util import load_one_image

class OLMining_Train(Dataset):
    def __init__(self, imgs, labels, transform, tau, norm_age=True, is_filelist=False, max_epoch=350):
        super(Dataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.n_imgs = len(self.imgs)
        if norm_age:
            self.labels = self.labels - min(self.labels)

        self.max_age = self.labels.max()
        self.min_age = self.labels.min()
        self.tau = tau
        self.is_filelist = is_filelist

        self.max_hard_sample_prob = 0.75
        self.init_hardness_multiplier = 3
        self.hard_sample_prob = 0
        self.hardness_multiplier = 10

        self.epoch = 0
        self.max_epoch = max_epoch

    def __getitem__(self, item):
        order_label, ref_idx = self.find_reference(self.labels[item], self.labels, min_rank=self.min_age,
                                                   max_rank=self.max_age)
        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
            ref_img = np.asarray(load_one_image(self.imgs[ref_idx])).astype('uint8')
        else:
            base_img = np.asarray(self.imgs[item]).astype('uint8')
            ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')
        base_img = self.transform(base_img)
        ref_img = self.transform(ref_img)

        one_hot_vector = self.convert_to_onehot(order_label)

        # gt ages
        base_age = self.labels[item]
        ref_age = self.labels[ref_idx]
        return base_img, ref_img, one_hot_vector, order_label, [base_age, ref_age], item

    def __len__(self):
        return self.n_imgs

    def convert_to_onehot(self, order):
        if order == 0:
            one_hot_vector = torch.tensor([1, 0], dtype=torch.float32)
        elif order == 1:
            one_hot_vector = torch.tensor([0, 1], dtype=torch.float32)
        elif order == 2:
            one_hot_vector = torch.tensor([0.5, 0.5], dtype=torch.float32)
        else:
            raise ValueError(f'order value {order} is out of expected range.')
        return one_hot_vector

    def update_mining_params(self, ):
        self.epoch += 1

        self.hard_sample_prob = self.max_hard_sample_prob*self.epoch/self.max_epoch
        self.hardness_multiplier = -((self.init_hardness_multiplier - 1)/self.max_epoch*self.epoch) + self.init_hardness_multiplier

    def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=32, epsilon=1e-4):

        def get_indices_in_range(search_range, ages):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))

        is_normal = np.random.choice([True, False], p=[1-self.hard_sample_prob, self.hard_sample_prob])

        order = np.random.randint(0, 3)
        ref_idx = -1
        debug_flag = 0
        while ref_idx == -1:
            if debug_flag == 3:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if order == 0:  # base_rank > ref_rank
                ref_range_min = min_rank if is_normal else max(base_rank - (self.tau*self.hardness_multiplier), min_rank)
                ref_range_max = base_rank - self.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1: # base_rank < ref_rank
                ref_range_min = base_rank + self.tau + epsilon
                ref_range_max = max_rank if is_normal else min(base_rank + (self.tau*self.hardness_multiplier), max_rank)
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # base_rank = ref_rank
                ref_range_min = base_rank - self.tau - epsilon
                ref_range_max = base_rank + self.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx