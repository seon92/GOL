import pickle
from torch.utils.data import DataLoader

from data.datasets import OL_basic_train, OL_basic_val


def get_datasets(cfg):
    with open(cfg.train_file, 'rb') as f:
        data = pickle.load(f)
        tr_imgs = data['data']
        tr_ages = data['age']

    with open(cfg.test_file, 'rb') as f:
        data = pickle.load(f)
        te_imgs = data['data']
        te_ages = data['age']

    loader_dict = dict()
    loader_dict['train'] = DataLoader(OL_basic_train.OLBasic_Train(tr_imgs, tr_ages, cfg.transform_tr, cfg.tau),
                                      batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    loader_dict['val'] = DataLoader(OL_basic_val.OLBasic_Val([te_imgs, te_ages], [tr_imgs, tr_ages], cfg.transform_te, cfg.tau),
                                     batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    return loader_dict





