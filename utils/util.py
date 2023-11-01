import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
import wandb
from copy import deepcopy

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, 
                                              size_average=size_average, 
                                              reduce=reduce, 
                                              reduction=reduction,
                                              pos_weight=pos_weight)
    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ClassWiseAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_cls):
        self.n_cls = n_cls
        self.reset()

    def reset(self):
        self.val = np.zeros([self.n_cls,])
        self.avg = np.zeros([self.n_cls,])
        self.sum = np.zeros([self.n_cls,])
        self.count = np.ones([self.n_cls,]) * 1e-7
        self.total_avg = 0

    def update(self, val, n=[1,1,1]):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.total_avg = np.sum(self.sum) / np.sum(self.count)


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cls_accuracy(output, target, n_cls=3):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.view(-1)
        correct = pred.eq(target).cpu().numpy()
        accs = np.zeros([n_cls,])
        cnts = np.ones([n_cls,]) * 1e-5
        target = target.cpu().numpy()
        for i_cls in range(n_cls):
            i_cls_idx = np.argwhere(target == i_cls).flatten()
            if len(i_cls_idx) > 0:
                cnts[i_cls] = len(i_cls_idx)
                accs[i_cls] = np.sum(correct[i_cls_idx])/len(i_cls_idx)*100

        return accs, cnts


def cls_accuracy_bc(output, target, cls=[0,1,2], delta=0.1):
    with torch.no_grad():
        accs = np.zeros([3, ])
        cnts = np.ones([3,])* 1e-7
        _, pred = output.topk(1, 1, True, True)
        pred = pred.view(-1)
        correct = pred.eq(target).cpu().numpy()
        for i in range(len(target)):
            if target[i] == cls[0]:
                accs[0] += correct[i]
                cnts[0] += 1
            elif target[i] == cls[1]:
                accs[1] += correct[i]
                cnts[1] += 1
            elif target[i] == cls[2]:
                i_correct = np.abs(output[i][0].cpu().numpy() - 0.5) < delta
                accs[2] += i_correct
                cnts[2] += 1
            else:
                raise ValueError(f'Out of range error! {target[i]} is given')
        accs = accs/ cnts *100
        return accs, cnts


def get_confusion_matrix_bc(output, target, cls=[-1,0,1], delta=0.1):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.view(-1).cpu().numpy()

        for i in range(len(target)):
            if target[i] == cls[0]:
                if np.abs(output[i][0].cpu().numpy()-0.5) < delta:
                    pred[i] = -1
                else:
                    continue

        pred = np.transpose(pred)
        cm = confusion_matrix(target.cpu().numpy(), pred)

        return cm, np.diag(cm)/np.sum(cm, axis=-1)


def get_confusion_matrix(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        cm = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy())

        return cm, np.diag(cm)/np.sum(cm, axis=-1)


def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)


def cross_entropy_loss_with_one_hot_labels(logits, labels):
    log_probs = nn.functional.log_softmax(logits, dim=1)
    loss = -torch.sum(log_probs*labels, dim=1)
    return loss.mean()


def cross_entropy_loss_with_one_hot_labels_with_weights(logits, labels, weights):
    log_probs = nn.functional.log_softmax(logits, dim=1)
    loss = -torch.sum(log_probs*labels, dim=1) * weights
    return loss.mean()


def mix_ce_and_kl_loss(logits, labels, mask, alpha=1):
    inv_mask = mask.__invert__()
    log_probs = nn.functional.log_softmax(logits, dim=1)
    ce_loss = -torch.sum(log_probs[mask]*labels[mask], dim=1)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(log_probs[inv_mask], labels[inv_mask])
    loss = ce_loss.mean() + alpha*kl_loss
    return loss


def load_one_image(img_path, width=256, height=256):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, height))
    return img


def load_images(img_root, img_name_list, width=256, height=256):
    num_images = len(img_name_list)
    images = np.zeros([num_images, height, width, 3], dtype=np.uint8)
    for idx, img_path in enumerate(img_name_list):
        img = cv2.imread(os.path.join(img_root, img_path), cv2.IMREAD_COLOR)
        images[idx] = cv2.resize(img, (width, height))
    return images

def to_np(x):
    return x.cpu().detach().numpy()


def get_current_time():
    _now = datetime.now()
    _now = str(_now)[:-7]
    return _now


def display_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'], param_group['initial_lr'])


def get_distribution(data):
    cls, cnt = np.unique(data, return_counts=True)
    for i_cls, i_cnt in zip(cls, cnt):
        print(f'{i_cls}: {i_cnt} ({i_cnt/len(data)*100:.2f}%)')
    print(f'total: {len(data)}')


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def log_configs(cfg, log_file='log.txt'):
    if os.path.exists(f'{cfg.save_folder}/{log_file}'):
        log_file = open(f'{cfg.save_folder}/{log_file}', 'a')
    else:
        log_file = open(f'{cfg.save_folder}/{log_file}', 'w')
    opt_dict = vars(cfg)
    for key in opt_dict.keys():
        write_log(log_file, f'{key}: {opt_dict[key]}')
    return log_file


def save_ckpt(cfg, model, postfix):
    state = {
        'model': model.state_dict() if cfg.n_gpu <= 1 else model.module.state_dict(),
    }
    save_file = os.path.join(cfg.save_folder, f'{postfix}')
    torch.save(state, save_file)
    print(f'ckpt saved to {save_file}.')


def set_wandb(cfg, key='private_key'):
    wandb.login(key=key)
    wandb.init(project=cfg.experiment_name, tags=[cfg.dataset])
    wandb.config.update(cfg)
    wandb.save('*.py')
    wandb.run.save()


def extract_embs(encoder, data_loader):
    encoder.eval()
    embs = []
    inds = []
    with torch.no_grad():
        for x_base, _, item in data_loader:
            x_base = x_base.cuda()
            embs.append(encoder(x_base).cpu())
            inds.append(item)
    embs = torch.cat(embs)
    inds = torch.cat(inds)
    embs_temp = deepcopy(embs)
    embs[inds] = embs_temp

    return embs


def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x

def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x


def print_eval_result_by_groups_and_k(gt, ref_gt, preds_all, log_file, interval=10):
    test_cls_arr, cnt = np.unique(gt, return_counts=True)
    test_cls_min = test_cls_arr.min()
    test_cls_max = test_cls_arr.max()
    n_groups = int((test_cls_max - test_cls_min + 1) / interval + 0.5)

    title = 'Group \\ K |'
    for k in preds_all.keys():
        title += f" {k:<4} "
    title = title + ' | Best K | #Test | #Train '
    write_log(log_file, title)
    for i_group in range(n_groups):
        min_rank = interval * i_group
        max_rank = min(test_cls_max + 1, min_rank + interval)
        sample_idx_in_group = np.argwhere(np.logical_and(gt >= min_rank, gt < max_rank)).flatten()
        ref_sample_idx_in_group = np.argwhere(np.logical_and(ref_gt >= min_rank, ref_gt < max_rank)).flatten()

        if len(sample_idx_in_group) < 1:
            continue
        to_print = f' {min_rank:<3}~ {max_rank - 1:<3} |'

        best_k = -1
        best_mae = 1000
        for k in preds_all.keys():
            i_group_errors_at_k = np.abs(preds_all[k][sample_idx_in_group] - gt[sample_idx_in_group])
            i_group_mean_at_k = np.mean(i_group_errors_at_k)
            to_print += f' {i_group_mean_at_k:.3f}' if i_group_mean_at_k<10 else f' {i_group_mean_at_k:.2f}'
            if i_group_mean_at_k < best_mae:
                best_mae = i_group_mean_at_k
                best_k = k
        to_print += f' |   {best_k:<2}   | {len(sample_idx_in_group):<4}  | {len(ref_sample_idx_in_group):<4} '
        write_log(log_file, to_print)

    mean_all = '  Total   |'
    best_k = -1
    best_mae = 1000
    for k in preds_all.keys():
        mean_at_k = np.mean(np.abs(preds_all[k] - gt))
        mean_all += f' {mean_at_k:.3f}'
        if mean_at_k < best_mae:
            best_mae = mean_at_k
            best_k = k
    mean_all += f' |   {best_k:<2}   | {len(gt):<5} | {len(ref_gt):<5}'
    write_log(log_file, mean_all)
    write_log(log_file, f'Best Total MAE : {best_mae:.3f}\n')
    return best_mae, best_k



def sample_fdcs(model, fdc_pts, train_labels, cfg):
    to_select = np.unique(train_labels)
    model.select_reference_points(to_select.astype(np.int32), fdc_pts)
    cfg.fiducial_point_num = len(to_select)
    return model, cfg
