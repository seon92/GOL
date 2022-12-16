import pickle
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from data.datasets import basic


def get_datasets(cfg):
    if cfg.is_filelist:
        img_root = '/hdd/2020/Research/datasets/Agedataset/img/morph'
        tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        tr_list = np.array(tr_list)
        tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
        tr_ages = tr_list[:, cfg.lb_idx]

        te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
        te_list = np.array(te_list)
        te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
        te_ages = te_list[:, cfg.lb_idx]

    else:
        with open(cfg.train_file, 'rb') as f:
            data = pickle.load(f)
            tr_imgs = data['data']
            tr_ages = data['age']

        with open(cfg.test_file, 'rb') as f:
            data = pickle.load(f)
            te_imgs = data['data']
            te_ages = data['age']

    loader_dict = dict()
    loader_dict['train'] = DataLoader(basic.Basic(tr_imgs, tr_ages, cfg.transform_tr, cfg.tau, is_filelist=cfg.is_filelist),
                                      batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    loader_dict['val'] = DataLoader(basic.Basic(te_imgs, te_ages, cfg.transform_te, cfg.tau, is_filelist=cfg.is_filelist),
                                     batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    loader_dict['train_for_val'] = DataLoader(basic.Basic(tr_imgs, tr_ages, cfg.transform_tr, cfg.tau, is_filelist=cfg.is_filelist),
                                      batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)

    return loader_dict





