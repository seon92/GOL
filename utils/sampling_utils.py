import numpy as np
import torch


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


def get_pairs_loss_balancing(ranks, loss_record, tau, m=32):
    orders = []
    base_idx = []
    ref_idx = []
    loss_val = []
    N = len(ranks)

    for i in range(N):
        for j in range(i+1, N):
            if np.random.rand(1) > 0.5:
                base_idx.append(i)
                ref_idx.append(j)
                order_ij = get_order_labels(ranks[i], ranks[j], tau)
                orders.append(order_ij)
                loss_val.append(loss_record[ranks[i], ranks[j]])

            else:
                base_idx.append(j)
                ref_idx.append(i)
                order_ji = get_order_labels(ranks[j], ranks[i], tau)
                orders.append(order_ji)
                loss_val.append(loss_record[ranks[j], ranks[i]])
    orders = np.array(orders)
    loss_val = np.array(loss_val)
    refine = []

    for o in range(3):
        o_idxs = np.argwhere(orders==o).flatten()
        if len(o_idxs) > m:
            sel = np.random.choice(o_idxs, m, p=np.exp(loss_val[o_idxs]) / np.sum(np.exp(loss_val[o_idxs])),
                                      replace=False)
            refine.append(sel)
        else:
            refine.append(o_idxs)

    refine = np.concatenate(refine)
    base_idx = np.array(base_idx)[refine]
    ref_idx = np.array(ref_idx)[refine]
    orders = orders[refine]
    return base_idx, ref_idx, orders


class LossTracker:
    def __init__(self, cfg):
        self.n_ranks = cfg.n_ranks

        self.pairwise_loss_record = dict()
        self.pairwise_loss_record['drct'] = np.zeros([self.n_ranks, self.n_ranks])
        self.pairwise_loss_record['dist'] = np.zeros([self.n_ranks, self.n_ranks])
        self.pairwise_loss_record['total'] = np.zeros([self.n_ranks, self.n_ranks])

        self.samplewise_loss_record = dict()
        self.samplewise_loss_record['center'] = np.zeros([self.n_ranks,])

        self.counter = dict()
        self.counter['drct'] = np.zeros([self.n_ranks, self.n_ranks])
        self.counter['dist'] = np.zeros([self.n_ranks, self.n_ranks])
        self.counter['total'] = np.zeros([self.n_ranks, self.n_ranks])
        self.counter['center'] = np.zeros([self.n_ranks, ])
        self.tau = cfg.tau

    def update_pairwise_loss_matrix(self, loss, base_ranks, ref_ranks, losstype='drct'):
        pair_size = len(base_ranks)
        for i in range(pair_size):
            self.pairwise_loss_record[losstype][base_ranks[i], ref_ranks[i]] += loss[i]
            self.counter[losstype][base_ranks[i], ref_ranks[i]] += 1

    def update_pairwise_loss_matrix_total(self, drct_loss, dist_loss, base_ranks, ref_ranks):
        pair_size = len(base_ranks)
        drct_idx = 0
        for i in range(pair_size):
            if abs(base_ranks[i] - ref_ranks[i]) <= self.tau:
                pass
            else:
                self.pairwise_loss_record['drct'][base_ranks[i], ref_ranks[i]] += drct_loss[drct_idx]
                self.pairwise_loss_record['total'][base_ranks[i], ref_ranks[i]] += drct_loss[drct_idx]
                self.counter['drct'][base_ranks[i], ref_ranks[i]] += 1
                drct_idx += 1
            self.pairwise_loss_record['dist'][base_ranks[i], ref_ranks[i]] += dist_loss[i]
            self.pairwise_loss_record['total'][base_ranks[i], ref_ranks[i]] += dist_loss[i]
            self.counter['dist'][base_ranks[i], ref_ranks[i]] += 1
        self.counter['total'] = self.counter['dist']

    def update_samplewise_loss(self, loss, ranks, losstype='center'):
        batch_size = len(ranks)
        for i in range(batch_size):
            self.samplewise_loss_record[losstype][ranks[i]] += loss[i]
            self.counter[losstype][ranks[i]] += 1

    def get_avg_samplewise_loss(self, losstypes=['total']):
        avg_samplewise_loss = np.zeros_like(self.samplewise_loss_record['center'])
        avg_samplewise_loss += (self.samplewise_loss_record['center']/(self.counter['center']+1e-7))
        for losstype in losstypes:
            samplewise_loss_sum = np.sum(self.pairwise_loss_record[losstype], axis=-1)
            samplewise_cnt = np.sum(self.counter[losstype], axis=-1)
            avg_samplewise_loss += samplewise_loss_sum / (samplewise_cnt+1e-7)
        cnt_zero_idx = np.argwhere(self.counter['center']==0).flatten()
        mean_loss_val = np.mean(avg_samplewise_loss)
        avg_samplewise_loss[cnt_zero_idx] = mean_loss_val  # assign mean loss value to some ranks if they didn't occur during previous training period
        return avg_samplewise_loss

    def get_avg_pairwise_loss(self, ):
        avg_pairwise_loss = np.zeros_like(self.pairwise_loss_record['total'])
        avg_pairwise_loss += (self.pairwise_loss_record['total']/(self.counter['total']+1e-7))
        mean_loss_val = np.mean(avg_pairwise_loss)
        cnt_zero_idx = np.argwhere(self.counter['total'] == 0)
        avg_pairwise_loss[cnt_zero_idx[:,0], cnt_zero_idx[:,1]] = mean_loss_val
        avg_samplewise_loss = (self.samplewise_loss_record['center']/(self.counter['center']+1e-7)).reshape(-1,1)
        avg_pairwise_loss = avg_pairwise_loss + avg_samplewise_loss
        return avg_pairwise_loss

    def restart_record(self):
        self.pairwise_loss_record = dict()
        self.pairwise_loss_record['drct'] = np.zeros([self.n_ranks, self.n_ranks])
        self.pairwise_loss_record['dist'] = np.zeros([self.n_ranks, self.n_ranks])
        self.pairwise_loss_record['total'] = np.zeros([self.n_ranks, self.n_ranks])

        self.samplewise_loss_record = dict()
        self.samplewise_loss_record['center'] = np.zeros([self.n_ranks, ])

        self.counter = dict()
        self.counter['drct'] = np.zeros([self.n_ranks, self.n_ranks])
        self.counter['dist'] = np.zeros([self.n_ranks, self.n_ranks])
        self.counter['total'] = np.zeros([self.n_ranks, self.n_ranks])
        self.counter['center'] = np.zeros([self.n_ranks, ])

def update_pairwise_loss_matrix_v2(A, drct_loss, dist_loss, base_ranks, ref_ranks=None):
    batch_size = len(base_ranks)
    if ref_ranks is not None:
        for i in range(batch_size):
            A[0][base_ranks[i], ref_ranks[i]] += (drct_loss[i] + dist_loss[i])
            A[1][base_ranks[i], ref_ranks[i]] += 1
    else:
        for i in range(batch_size):
            A[0][base_ranks[i]] += (drct_loss[i] + dist_loss[i])
            A[1][base_ranks[i]] += 1
    return A



def get_order_labels(rank_base, rank_ref, tau):
    if rank_base > rank_ref + tau:
        order = 0
    elif rank_base < rank_ref - tau:
        order = 1
    else:
        order = 2
    return order
