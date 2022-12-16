import torch
import torch.nn as nn
import numpy as np
import itertools

from utils.util import to_np


def compute_order_loss(embs, base_idx, ref_idx, rank_labels, fdc_points, cfg, record=False):
    def get_forward_and_backward_idxs(base_idx, ref_idx, ranks, fdc_ranks, cfg):
        batch_size = len(base_idx)
        base_ranks = ranks[base_idx]
        ref_ranks = ranks[ref_idx]
        forward_idxs = []
        backward_idxs = []
        mask = []
        gt = []
        for i in range(batch_size):
            if base_ranks[i] > ref_ranks[i]:
                fdc_1_idx = cfg.fiducial_point_num - np.sum(fdc_ranks - ref_ranks[i] > 0) - 1
                fdc_2_idx = cfg.fiducial_point_num - np.sum(fdc_ranks - base_ranks[i] >= 0)
                fdc_3_idx = fdc_2_idx + 1

                backward_idxs.append([fdc_1_idx, fdc_2_idx])
                if fdc_3_idx >= len(fdc_points):
                    forward_idxs.append([fdc_2_idx, fdc_2_idx-1])
                else:
                    forward_idxs.append([fdc_3_idx, fdc_2_idx])

                mask.append(True)
                gt.append(0)
            elif base_ranks[i] < ref_ranks[i]:
                fdc_1_idx = cfg.fiducial_point_num - np.sum(fdc_ranks - base_ranks[i] > 0) - 1
                fdc_2_idx = cfg.fiducial_point_num - np.sum(fdc_ranks - ref_ranks[i] >= 0)
                fdc_3_idx = fdc_1_idx - 1
                forward_idxs.append([fdc_2_idx, fdc_1_idx])
                if fdc_3_idx < 0:
                    backward_idxs.append([fdc_1_idx, fdc_1_idx+1])
                else:
                    backward_idxs.append([fdc_3_idx, fdc_1_idx])
                mask.append(True)
                gt.append(1)
            else:
                mask.append(False)

        return np.array(forward_idxs), np.array(backward_idxs), torch.tensor(gt).cuda(), base_idx[mask], ref_idx[mask]

    fdc_points = nn.functional.normalize(fdc_points, dim=-1)
    hdim = fdc_points.shape[-1]
    fdc_point_ranks = np.array([((cfg.n_ranks-1) / (cfg.fiducial_point_num-1)) * i for i in range(cfg.fiducial_point_num)])

    direction_matrix = fdc_points.view(cfg.fiducial_point_num, 1, hdim).expand(cfg.fiducial_point_num, cfg.fiducial_point_num, hdim) - fdc_points.view(1, cfg.fiducial_point_num, hdim).expand(cfg.fiducial_point_num, cfg.fiducial_point_num, hdim)
    direction_matrix = nn.functional.normalize(direction_matrix, dim=-1)

    forward_idxs, backward_idxs, gt, base_idx, ref_idx = get_forward_and_backward_idxs(base_idx, ref_idx, rank_labels, fdc_point_ranks, cfg)
    batch_size = base_idx.shape[0]

    v_xy = nn.functional.normalize(embs[ref_idx] - embs[base_idx], dim=-1)
    v_forward = direction_matrix[forward_idxs[:,0], forward_idxs[:,1]]
    v_backward = direction_matrix[backward_idxs[:,0], backward_idxs[:,1]]

    v_fb = torch.stack([v_backward, v_forward], dim=-1)
    logits = 20*torch.matmul(v_xy.view(batch_size, 1, hdim), v_fb).squeeze(1)


    if record:
        loss_per_pair = nn.CrossEntropyLoss(reduction='none')(logits, gt)
        loss = torch.mean(loss_per_pair)
        return loss, logits, gt, to_np(loss_per_pair)
    else:
        loss = nn.CrossEntropyLoss()(logits, gt)
        return loss, logits, gt


def compute_metric_loss(embs, base_idx, ref_idx, rank_labels, fdc_points, margin, cfg, record=False):
    fdc_points = nn.functional.normalize(fdc_points, dim=-1)
    fdc_point_ranks = np.array(
        [((cfg.n_ranks - 1) / (cfg.fiducial_point_num - 1)) * i for i in range(cfg.fiducial_point_num)])

    if cfg.metric == 'L2':
        dists = torch.cdist(fdc_points, embs)
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(fdc_points, embs.transpose(1, 0))
    def get_pos_neg_idxs(base_idx, ref_idx, ranks, fdc_ranks, cfg):
        batch_size = len(base_idx)
        base_ranks = ranks[base_idx]
        ref_ranks = ranks[ref_idx]
        row_idxs = []
        pos_idxs = []
        neg_idxs = []
        split_idxs = []

        sim_row_idxs = []
        sim_pos_idxs = []
        sim_neg_idxs = []

        for i in range(batch_size):
            if base_ranks[i] > (ref_ranks[i] + cfg.tau):
                fdc_1_idx = cfg.fiducial_point_num - np.sum(fdc_ranks - ref_ranks[i] > 0) - 1
                fdc_2_idx = cfg.fiducial_point_num - np.sum(fdc_ranks - base_ranks[i] >= 0)

                row_idxs.append(np.arange(fdc_1_idx+1))
                pos_idxs.append([ref_idx[i]]*(fdc_1_idx+1))
                neg_idxs.append([base_idx[i]]*(fdc_1_idx+1))
                row_idxs.append(np.arange(fdc_2_idx, cfg.fiducial_point_num))
                pos_idxs.append([base_idx[i]]*(cfg.fiducial_point_num-fdc_2_idx))
                neg_idxs.append([ref_idx[i]]*(cfg.fiducial_point_num-fdc_2_idx))
                split_idxs.append(fdc_1_idx + 1 + cfg.fiducial_point_num - fdc_2_idx)

            elif base_ranks[i] < (ref_ranks[i] - cfg.tau):
                fdc_1_idx = cfg.fiducial_point_num - np.sum(fdc_point_ranks - rank_labels[base_idx[i]] > 0) - 1
                fdc_2_idx = cfg.fiducial_point_num - np.sum(fdc_point_ranks - rank_labels[ref_idx[i]] >= 0)

                row_idxs.append(np.arange(fdc_1_idx + 1))
                pos_idxs.append([base_idx[i]] * (fdc_1_idx + 1))
                neg_idxs.append([ref_idx[i]] * (fdc_1_idx + 1))
                row_idxs.append(np.arange(fdc_2_idx, cfg.fiducial_point_num))
                pos_idxs.append([ref_idx[i]] * (cfg.fiducial_point_num-fdc_2_idx))
                neg_idxs.append([base_idx[i]] * (cfg.fiducial_point_num-fdc_2_idx))
                split_idxs.append(fdc_1_idx + 1 + cfg.fiducial_point_num - fdc_2_idx)
            else:
                sim_row_idxs.append(np.arange(cfg.fiducial_point_num))
                sim_pos_idxs.append([base_idx[i]]*cfg.fiducial_point_num)
                sim_neg_idxs.append([ref_idx[i]]*cfg.fiducial_point_num)
                split_idxs.append(cfg.fiducial_point_num)
        row_idxs = np.concatenate(row_idxs)
        pos_idxs = np.concatenate(pos_idxs)
        neg_idxs = np.concatenate(neg_idxs)
        sim_row_idxs = np.concatenate(sim_row_idxs)
        sim_pos_idxs = np.concatenate(sim_pos_idxs)
        sim_neg_idxs = np.concatenate(sim_neg_idxs)
        return row_idxs, pos_idxs, neg_idxs, sim_row_idxs, sim_pos_idxs, sim_neg_idxs, split_idxs

    row_idxs, pos_idxs, neg_idxs, sim_row_idxs, sim_pos_idxs, sim_neg_idxs, split_idxs = get_pos_neg_idxs(base_idx, ref_idx, rank_labels, fdc_point_ranks, cfg)

    violation = dists[row_idxs, pos_idxs] - dists[row_idxs,neg_idxs]
    violation = violation + margin

    if len(sim_row_idxs) > 0:
        if cfg.tau == 0:
            sim_violation = torch.abs(dists[sim_row_idxs, sim_pos_idxs] - dists[sim_row_idxs, sim_neg_idxs])
        else:
            sim_violation = torch.abs(dists[sim_row_idxs,sim_pos_idxs] - dists[sim_row_idxs,sim_neg_idxs]) - margin
        loss = torch.cat([nn.functional.relu(violation), nn.functional.relu(sim_violation)])

    else:
        loss = nn.functional.relu(violation)
    if record:
        loss_per_pairs = torch.tensor([torch.sum(s) for s in torch.split(loss, split_idxs)])
        return torch.sum(loss) / len(base_idx), to_np(loss_per_pairs)
    return torch.sum(loss) / len(base_idx)




def compute_center_loss(embs, rank_labels, fdc_points, cfg, record=False):
    fdc_points = nn.functional.normalize(fdc_points, dim=-1)
    fdc_point_ranks = np.array([((cfg.n_ranks - 1) / (cfg.fiducial_point_num - 1)) * i for i in range(cfg.fiducial_point_num)])

    def get_pos_neg_idxs(ranks, fdc_ranks, cfg):
        adaptive_margin = cfg.n_ranks != cfg.fiducial_point_num
        if adaptive_margin:
            nn_idxs = []
            margins = []
            emb_idxs = []
            emb_idx = 0
            for r in ranks:
                abs_diff = np.abs(fdc_ranks-r)
                min_val = abs_diff.min()
                nn = np.argwhere(abs_diff==min_val).flatten()
                nn_idxs.append(nn)

                margin_val = min_val*cfg.margin/(max(cfg.tau, 1))
                margins.append([margin_val]*len(nn))
                emb_idxs.append([emb_idx]*len(nn))
                emb_idx += 1
            nn_idxs = np.concatenate(nn_idxs)
            margins = np.concatenate(margins)
            emb_idxs = np.concatenate(emb_idxs)
        else:
            nn_idxs = ranks
            margins = np.array([0.5 * cfg.margin / (max(cfg.tau, 1))] * len(nn_idxs))
            emb_idxs = np.arange(len(nn_idxs))

        return nn_idxs, emb_idxs, margins

    nn_idxs, emb_idxs, margins = get_pos_neg_idxs(rank_labels, fdc_point_ranks, cfg)

    if cfg.metric == 'L2':
        dists = torch.cdist(fdc_points, embs)
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(fdc_points, embs.transpose(1, 0))


    loss = dists[nn_idxs, emb_idxs]

    # loss = nn.functional.relu(violation)
    # loss = torch.tensor([torch.sum(s) for s in torch.split(loss, split_idxs)])
    if record:
        return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7), to_np(loss)
    return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7)

