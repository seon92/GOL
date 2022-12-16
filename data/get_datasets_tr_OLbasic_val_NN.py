import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from data.datasets import OL_basic_train, basic



def get_datasets(cfg):
    tr_std = None
    te_std = None
    if cfg.dataset =='morph':
        img_root = cfg.img_root
        tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        tr_list = np.array(tr_list)
        tr_imgs = [f'{img_root}/{i_path}' for i_path in tr_list[:, cfg.img_idx]]
        tr_ages = tr_list[:, cfg.lb_idx]

        te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
        te_list = np.array(te_list)
        te_imgs = [f'{img_root}/{i_path}' for i_path in te_list[:, cfg.img_idx]]
        te_ages = te_list[:, cfg.lb_idx]

    elif cfg.dataset =='clap':
        img_root = cfg.img_root
        tr_list = pd.read_csv(cfg.train_file, sep=cfg.delimeter)
        tr_list = np.array(tr_list)
        tr_ages = tr_list[:, cfg.lb_idx]
        tr_imgs = [f'{img_root}/{tr_list[i, 3]}/{tr_list[i, cfg.img_idx]}' for i in range(len(tr_list))]
        tr_std = tr_list[:, 2]
        #
        # # debug for n_ranks and margin relation
        # idx = np.argwhere(tr_ages < 60).flatten()
        # tr_ages = tr_ages[idx]
        # tr_imgs = np.array(tr_imgs)[idx]
        # tr_std = tr_std[idx]

        te_list = pd.read_csv(cfg.test_file, sep=cfg.delimeter)
        te_list = np.array(te_list)
        te_imgs = [f'{img_root}/{te_list[i, 3]}/{te_list[i, cfg.img_idx]}' for i in range(len(te_list))]
        te_ages = te_list[:, cfg.lb_idx]
        te_std = te_list[:, 2]
        #
        # # debug for n_ranks and margin relation
        # idx = np.argwhere(te_ages < 60).flatten()
        # te_ages = te_ages[idx]
        # te_imgs = np.array(te_imgs)[idx]
        # te_std = te_std[idx]

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
    loader_dict['train'] = DataLoader(OL_basic_train.OLBasic_Train(tr_imgs, tr_ages, cfg.transform_tr, cfg.tau, logscale=cfg.logscale, is_filelist=cfg.is_filelist),
                                      batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    loader_dict['train_for_val'] = DataLoader(basic.Basic(tr_imgs, tr_ages, cfg.transform_te, is_filelist=cfg.is_filelist, norm_age=False),
                                    batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                                    num_workers=cfg.num_workers)

    loader_dict['val'] = DataLoader(basic.Basic(te_imgs, te_ages, cfg.transform_te, is_filelist=cfg.is_filelist, std=te_std, norm_age=False),
                                     batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    return loader_dict





