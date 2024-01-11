import os
import time
import sys
from copy import deepcopy

import numpy as np
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from config.basic import ConfigBasic
from utils.util import write_log, get_current_time, to_np, make_dir, log_configs, save_ckpt, set_wandb
from utils.util import adjust_learning_rate, AverageMeter, ClassWiseAverageMeter, cls_accuracy, extract_embs, print_eval_result_by_groups_and_k
from utils.loss_util import compute_order_loss, compute_metric_loss, compute_center_loss
from utils.comparison_utils import find_kNN
from networks.util import prepare_model
from data.get_datasets_tr_OLbasic_val_NN import get_datasets


def set_local_config(cfg):
    # Dataset
    cfg.dataset = 'morph'
    cfg.setting = 'D'
    cfg.fold = 4

    cfg.logscale = False
    cfg.set_dataset()
    cfg.tau = 1

    # Model
    cfg.model = 'GOL'
    cfg.backbone = 'vgg16v2norm'
    cfg.metric = 'L2'
    cfg.k = np.arange(2, 60, 2)
    cfg.epochs = 100
    cfg.scheduler = 'cosine'
    cfg.lr_decay_epochs = [100, 200, 300]
    cfg.period = 3

    cfg.margin = 0.25
    cfg.ref_mode = 'flex'
    cfg.ref_point_num = 60  # 60 Fold1, 58 Fold0 setting D // 56 setting c // 58 setting B // 55 setting A
    cfg.drct_wieght = 1
    cfg.start_norm = True
    cfg.learning_rate = 0.0001

    # Log
    cfg.wandb = False
    cfg.experiment_name = 'EXP_NAME'
    cfg.save_folder = f'../../RESULT_FOLDER_NAME/{cfg.dataset}/setting{cfg.setting}/{cfg.experiment_name}/PREFIX_{cfg.margin}_tau{cfg.tau}_F{cfg.fold}_{cfg.model}_{cfg.backbone}_{get_current_time()}'
    make_dir(cfg.save_folder)

    cfg.n_gpu = torch.cuda.device_count()
    cfg.num_workers = 1
    return cfg


def main():
    np.random.seed(999)

    cfg = ConfigBasic()
    cfg = set_local_config(cfg)
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')

    # dataloader
    loader_dict = get_datasets(cfg)
    cfg.n_ranks = loader_dict['train'].dataset.ranks.max() + 1
    print(f'[*] {cfg.n_ranks} ranks exist. ')

    # model
    model = prepare_model(cfg)
    if cfg.wandb:
        set_wandb(cfg)
        wandb.watch(model)

    if cfg.adam:
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.learning_rate,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    if cfg.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.learning_rate*0.001)
    elif cfg.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_epochs, gamma=cfg.lr_decay_rate)

    # criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if torch.cuda.is_available():
        if cfg.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True

    val_mae_best = 2.9
    log_dict = dict()
    # init loss matrix
    loss_record = dict()
    loss_record['angle'] = [np.zeros([cfg.n_ranks, cfg.n_ranks]), np.zeros([cfg.n_ranks, cfg.n_ranks])]
    # loss_record['angle'] = [np.zeros([n_ranks, ]), np.zeros([n_ranks, ])]

    for epoch in range(cfg.epochs):
        print("==> training...")

        time1 = time.time()
        train_loss, loss_record = train(epoch, loader_dict['train'], model, optimizer, cfg,
                                                     prev_loss_record=loss_record)
        

        if cfg.scheduler:
            scheduler.step()
        time2 = time.time()
        print('epoch {}, loss {:.4f}, total time {:.2f}'.format(epoch, train_loss, time2 - time1))

        if epoch % cfg.val_freq == 0:
            print('==> validation...')
            val_mae, best_k = validate(loader_dict, model, cfg)
            if val_mae < val_mae_best:
                val_mae_best = val_mae
                save_ckpt(cfg, model, f'ep_{epoch}_val_best_{val_mae:.3f}_k{best_k}.pth')

        # if train_acc > best:
        #     best = train_acc
        #     save_ckpt(cfg, model, f'ep_{epoch}_train_best_{best:.3f}.pth')

        elif epoch % cfg.save_freq == 0:
            save_ckpt(cfg, model, f'ep_{epoch}.pth')

        if cfg.wandb:
            log_dict['Epoch'] = epoch
            log_dict['Train Loss'] = train_loss
            log_dict['Val Mae'] = val_mae
            log_dict['LR'] = scheduler.get_lr()[0] if scheduler else cfg.learning_rate
            wandb.log(log_dict)

    print('[*] Training ends')


def update_loss_matrix(A, loss, base_ranks, ref_ranks=None):
    batch_size = len(base_ranks)
    if ref_ranks is not None:
        for i in range(batch_size):
            A[0][base_ranks[i], ref_ranks[i]] += loss[i]
            A[1][base_ranks[i], ref_ranks[i]] += 1
    else:
        for i in range(batch_size):
            A[0][base_ranks[i]] += loss[i]
            A[1][base_ranks[i]] += 1
    return A


def get_pairs_equally(ranks, tau, m=32):
    orders = []
    base_idx = []
    ref_idx = []
    N = len(ranks)
    for i in range(N):
        for j in range(i+1, N):
            if np.random.rand(1) > 0.5:
                base_idx.append(i)
                ref_idx.append(j)
                order_ij = get_order_labels(ranks[i], ranks[j], tau)
                orders.append(order_ij)
            else:
                base_idx.append(j)
                ref_idx.append(i)
                order_ji = get_order_labels(ranks[j], ranks[i], tau)
                orders.append(order_ji)
    refine = []
    orders = np.array(orders)
    for o in range(3):
        o_idxs = np.argwhere(orders==o).flatten()
        if len(o_idxs) > m:
            sel = np.random.choice(o_idxs, m, replace=False)
            refine.append(sel)
        else:
            refine.append(o_idxs)
    refine = np.concatenate(refine)
    base_idx = np.array(base_idx)[refine]
    ref_idx = np.array(ref_idx)[refine]
    orders = orders[refine]
    return base_idx, ref_idx, orders


def get_order_labels(rank_base, rank_ref, tau):
    if rank_base > rank_ref + tau:
        order = 0
    elif rank_base < rank_ref - tau:
        order = 1
    else:
        order = 2
    return order


def train(epoch, train_loader, model, optimizer, cfg, prev_loss_record):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    angle_losses = AverageMeter()
    dist_losses = AverageMeter()
    center_losses = AverageMeter()
    angle_acc_meter = ClassWiseAverageMeter(2)
    # dist_acc_meter = ClassWiseAverageMeter(2)

    loss_record = deepcopy(prev_loss_record)
    end = time.time()
    for idx, (x_base, x_ref, _, ranks, _) in enumerate(train_loader):

        labels_np = torch.cat(ranks).detach().numpy()

        base_idx, ref_idx, order_labels = get_pairs_equally(labels_np, cfg.tau)

        if torch.cuda.is_available():            
            x_base = x_base.cuda()
            x_ref = x_ref.cuda()

            # order_labels = order_labels.cuda()
        data_time.update(time.time() - end)

        # ===================forward=====================
        embs = model.encoder(torch.cat([x_base, x_ref], dim=0))

        # =====================loss======================
        tic = time.time()
        dist_loss = compute_metric_loss(embs, base_idx, ref_idx, labels_np, model.ref_points, cfg.margin, cfg)
        dist_loss_time = time.time() - tic
        tic = time.time()
        angle_loss, logits, order_gt = compute_order_loss(embs, base_idx, ref_idx, labels_np, model.ref_points, cfg)
        angle_loss_time = time.time() - tic
        center_loss = compute_center_loss(embs, labels_np, model.ref_points, cfg)


        total_loss = (cfg.drct_wieght * angle_loss) + dist_loss + center_loss
        losses.update(total_loss.item(), x_base.size(0))
        angle_losses.update(angle_loss.item(), x_base.size(0))
        dist_losses.update(dist_loss.item(), x_base.size(0))
        center_losses.update(center_loss.item(), x_base.size(0))
        # ===================backward=====================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        acc, cnt = cls_accuracy(nn.functional.softmax(logits, dim=-1), order_gt, n_cls=2)
        # dist_acc, dist_cnt = cls_accuracy(nn.functional.softmax(dist_logits, dim=-1), dist_gt, n_cls=2)

        angle_acc_meter.update(acc, cnt)
        # dist_acc_meter.update(dist_acc, dist_cnt)

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # update loss matrix
        # loss_record['angle'] = update_loss_matrix(loss_record['angle'], to_np(angle_loss), labels_np[base_idx], labels_np[ref_idx])

        # print info
        if idx % cfg.print_freq == 0:
            write_log(cfg.logfile,
                      f'Epoch [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f}\t'
                      f'Data {data_time.val:3f}\t'
                      f'Loss {losses.val:.4f}\t'
                      f'Angle-Loss {angle_losses.val:.4f}\t'
                      f'Dist-Loss {dist_losses.val:.4f}\t'
                      f'Center-Loss {center_losses.val:.4f}\t'
                      f'Angle-Acc [{angle_acc_meter.val[0]:.3f}  {angle_acc_meter.val[1]:.3f}]  [{angle_acc_meter.total_avg:.3f}]\t'
                      )
            sys.stdout.flush()
    write_log(cfg.logfile, f' * Angle-Acc [{angle_acc_meter.avg[0]:.3f}  {angle_acc_meter.avg[1]:.3f}]  [{angle_acc_meter.total_avg:.3f}]\n')
    # write_log(cfg.logfile, f' * Dist-Acc [{dist_acc_meter.avg[0]:.3f}  {dist_acc_meter.avg[1]:.3f}]  [{dist_acc_meter.total_avg:.3f}]\n')

    return losses.avg, loss_record


def validate(loader_dict, model, cfg):
    model.eval()
    data_time = AverageMeter()

    embs_train = extract_embs(model.encoder, loader_dict['train_for_val'])
    embs_train = embs_train.cuda()

    embs_test = extract_embs(model.encoder, loader_dict['val'])
    embs_test = embs_test.cuda()
    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.batch_size))
    test_labels = loader_dict['val'].dataset.labels
    train_labels = loader_dict['train_for_val'].dataset.labels

    preds_all = defaultdict(list)

    with torch.no_grad():
        end = time.time()
        for idx in range(n_batch):
            data_time.update(time.time() - end)
            i_st = idx * cfg.batch_size
            i_end = min(i_st + cfg.batch_size, n_test)

            # ===================meters=====================
            vals, inds = find_kNN(embs_test[i_st:i_end].view(i_end - i_st, -1), embs_train, k=max(cfg.k),
                                  metric=cfg.metric)
            inds = np.squeeze(to_np(inds), 0)
            for k in cfg.k:
                nn_labels = train_labels[inds[:, :k]]
                pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32))
                preds_all[k].append(pred_mean)

    for key in preds_all.keys():
        preds_all[key] = np.concatenate(preds_all[key])

    best_mae, best_k = print_eval_result_by_groups_and_k(test_labels, train_labels, preds_all, cfg.logfile, interval=3)
    acc = np.sum(test_labels==preds_all[best_k])/len(test_labels)
    write_log(cfg.logfile, f'Acc : {acc*100:.2f}')
    sys.stdout.flush()
    return best_mae, best_k


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
