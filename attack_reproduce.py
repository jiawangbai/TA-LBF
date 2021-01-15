
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.nn as nn
from bitstring import Bits
import numpy as np
import torch.nn.functional as F
import os
import copy
from utils import *

parser = argparse.ArgumentParser(description='TA-LBF (targeted attack with limited bit-flips)')

parser.add_argument('--gpu-id', '-gpu-id', default="7", type=str)


parser.add_argument('--init-lam', '-init_lam', default=100, type=float)
parser.add_argument('--init-k', '-init_k', default=5, type=float)
parser.add_argument('--n-aux', '-n_aux', default=128, type=int)


parser.add_argument('--margin', '-margin', default=10, type=float)
parser.add_argument('--max-search-k', '-max_search_k', default=4, type=int)
parser.add_argument('--max-search-lam', '-max_search_lam', default=8, type=int)
parser.add_argument('--ext-max-iters', '-ext_max_iters', default=2000, type=int)
parser.add_argument('--inn-max-iters', '-inn_max_iters', default=5, type=int)
parser.add_argument('--initial-rho1', '-initial_rho1', default=0.0001, type=float)
parser.add_argument('--initial-rho2', '-initial_rho2', default=0.0001, type=float)
parser.add_argument('--initial-rho3', '-initial_rho3', default=0.00001, type=float)
parser.add_argument('--max-rho1', '-max_rho1', default=50, type=float)
parser.add_argument('--max-rho2', '-max_rho2', default=50, type=float)
parser.add_argument('--max-rho3', '-max_rho3', default=5, type=float)
parser.add_argument('--rho-fact', '-rho_fact', default=1.01, type=float)
parser.add_argument('--inn-lr', '-inn_lr', default=0.001, type=float)
parser.add_argument('--stop-threshold', '-stop_threshold', default=1e-4, type=float)
parser.add_argument('--projection-lp', '-projection_lp', default=2, type=int)


class AugLag(nn.Module):
    def __init__(self, n_bits, w, b, step_size, init=False):
        super(AugLag, self).__init__()

        self.n_bits = n_bits
        self.b = nn.Parameter(torch.tensor(b).float(), requires_grad=True)

        self.w_twos = nn.Parameter(torch.zeros([w.shape[0], w.shape[1], self.n_bits]), requires_grad=True)
        self.step_size = step_size
        self.w = w

        base = [2**i for i in range(self.n_bits-1, -1, -1)]
        base[0] = -base[0]
        self.base = nn.Parameter(torch.tensor([[base]]).float())

        if init:
            self.reset_w_twos()

    def forward(self, x):

        # covert w_twos to float
        w = self.w_twos * self.base
        w = torch.sum(w, dim=2) * self.step_size

        # calculate output
        x = F.linear(x, w, self.b)

        return x

    def reset_w_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_twos.data[i][j] += \
                    torch.tensor([int(b) for b in Bits(int=int(self.w[i][j]), length=self.n_bits).bin])


def project_box(x):
    xp = x
    xp[x>1]=1
    xp[x<0]=0

    return xp


def project_shifted_Lp_ball(x, p):
    shift_vec = 1/2*np.ones(x.size)
    shift_x = x-shift_vec
    normp_shift = np.linalg.norm(shift_x, p)
    n = x.size
    xp = (n**(1/p)) * shift_x / (2*normp_shift) + shift_vec

    return xp


def project_positive(x):
    xp = np.clip(x, 0, None)
    return xp


def loss_func(output, labels, s, t, lam, w, target_thr, source_thr,
              b_ori, k_bits, y1, y2, y3, z1, z2, z3, k, rho1, rho2, rho3):

    l1_1 = torch.max(output[-1][s] - source_thr, torch.tensor(0.0).cuda())
    l1_2 = torch.max(target_thr - output[-1][t], torch.tensor(0.0).cuda())
    l1 = l1_1 + l1_2
    # l1 = output[-1][s] - output[-1][t]
    # l1 = torch.max(output[-1][s] - source_thr, 0)[0] + torch.max(target_thr - output[-1][t], 0)[0]

    # print(target_thr, output[-1][t])
    # print(source_thr, output[-1][s])
    l2 = F.cross_entropy(output[:-1], labels[:-1])
    # print(l1.item(), l2.item())

    y1, y2, y3, z1, z2, z3 = torch.tensor(y1).float().cuda(), torch.tensor(y2).float().cuda(), torch.tensor(y3).float().cuda(), \
                             torch.tensor(z1).float().cuda(), torch.tensor(z2).float().cuda(), torch.tensor(z3).float().cuda()

    b_ori = torch.tensor(b_ori).float().cuda()
    b = torch.cat((w[s].view(-1), w[t].view(-1)))

    l3 = z1@(b-y1) + z2@(b-y2) + z3*(torch.norm(b - b_ori) ** 2 - k + y3)

    l4 = (rho1/2) * torch.norm(b - y1) ** 2 + (rho2/2) * torch.norm(b - y2) ** 2 \
         + (rho3/2) * (torch.norm(b - b_ori)**2 - k_bits + y3) ** 2

    return l1 + lam * l2 + l3 + l4


def attack(auglag_ori, all_data, labels, labels_cuda, clean_output,
           attack_idx, target_class, source_class, aux_idx,
           lam, k, args):
    # set parameters
    n_aux = args.n_aux
    lam = lam
    ext_max_iters = args.ext_max_iters
    inn_max_iters = args.inn_max_iters
    initial_rho1 = args.initial_rho1
    initial_rho2 = args.initial_rho2
    initial_rho3 = args.initial_rho3
    max_rho1 = args.max_rho1
    max_rho2 = args.max_rho2
    max_rho3 = args.max_rho3
    rho_fact = args.rho_fact
    k_bits = k
    inn_lr = args.inn_lr
    margin = args.margin
    stop_threshold = args.stop_threshold

    projection_lp = args.projection_lp

    all_idx = np.append(aux_idx, attack_idx)

    sub_max = clean_output[attack_idx][[i for i in range(len(clean_output[-1])) if i != source_class]].max()
    target_thr = sub_max + margin
    source_thr = sub_max - margin

    auglag = copy.deepcopy(auglag_ori)

    b_ori_s = auglag.w_twos.data[source_class].view(-1).detach().cpu().numpy()
    b_ori_t = auglag.w_twos.data[target_class].view(-1).detach().cpu().numpy()
    b_ori = np.append(b_ori_s, b_ori_t)
    b_new = b_ori

    y1 = b_ori
    y2 = y1
    y3 = 0

    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y1)
    z3 = 0

    rho1 = initial_rho1
    rho2 = initial_rho2
    rho3 = initial_rho3

    stop_flag = False
    for ext_iter in range(ext_max_iters):

        y1 = project_box(b_new + z1 / rho1)
        y2 = project_shifted_Lp_ball(b_new + z2 / rho2, projection_lp)
        y3 = project_positive(-np.linalg.norm(b_new - b_ori, ord=2) ** 2 + k_bits - z3 / rho3)

        for inn_iter in range(inn_max_iters):

            input_var = torch.autograd.Variable(all_data[all_idx], volatile=True)
            target_var = torch.autograd.Variable(labels_cuda[all_idx].long(), volatile=True)

            output = auglag(input_var)
            loss = loss_func(output, target_var, source_class, target_class, lam, auglag.w_twos,
                             target_thr, source_thr,
                             b_ori, k_bits, y1, y2, y3, z1, z2, z3, k_bits, rho1, rho2, rho3)

            loss.backward(retain_graph=True)
            auglag.w_twos.data[target_class] = auglag.w_twos.data[target_class] - \
                                               inn_lr * auglag.w_twos.grad.data[target_class]
            auglag.w_twos.data[source_class] = auglag.w_twos.data[source_class] - \
                                               inn_lr * auglag.w_twos.grad.data[source_class]
            auglag.w_twos.grad.zero_()

        b_new_s = auglag.w_twos.data[source_class].view(-1).detach().cpu().numpy()
        b_new_t = auglag.w_twos.data[target_class].view(-1).detach().cpu().numpy()
        b_new = np.append(b_new_s, b_new_t)

        if True in np.isnan(b_new):
            return -1

        z1 = z1 + rho1 * (b_new - y1)
        z2 = z2 + rho2 * (b_new - y2)
        z3 = z3 + rho3 * (np.linalg.norm(b_new - b_ori, ord=2) ** 2 - k_bits + y3)

        rho1 = min(rho_fact * rho1, max_rho1)
        rho2 = min(rho_fact * rho2, max_rho2)
        rho3 = min(rho_fact * rho3, max_rho3)

        temp1 = (np.linalg.norm(b_new - y1)) / max(np.linalg.norm(b_new), 2.2204e-16)
        temp2 = (np.linalg.norm(b_new - y2)) / max(np.linalg.norm(b_new), 2.2204e-16)

        if max(temp1, temp2) <= stop_threshold and ext_iter > 100:
            stop_flag = True
            break

    auglag.w_twos.data[auglag.w_twos.data > 0.5] = 1.0
    auglag.w_twos.data[auglag.w_twos.data < 0.5] = 0.0

    output = auglag(all_data)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze(1)
    pa_acc = len([i for i in range(len(output)) if labels[i] == pred[i] and i != attack_idx and i not in aux_idx]) / \
                     (len(labels) - 1 - n_aux)

    n_bit = torch.norm(auglag_ori.w_twos.data.view(-1) - auglag.w_twos.data.view(-1), p=0).item()

    ret = {
        "pa_acc": pa_acc,
        "stop": stop_flag,
        "suc": target_class == pred[attack_idx].item(),
        "n_bit": n_bit
    }
    return ret


def main():
    np.random.seed(512)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(args)

    # prepare the data
    print("Prepare data ... ")
    arch = "resnet20_quan"
    bit_length = 8

    weight, bias, step_size = load_model(arch, bit_length)
    all_data, labels = load_data(arch, bit_length)
    labels_cuda = labels.cuda()

    attack_info = np.loadtxt(config.info_root).astype(int)

    auglag = AugLag(bit_length, weight, bias, step_size, init=True).cuda()

    clean_output = auglag(all_data)

    _, pred = clean_output.cpu().topk(1, 1, True, True)
    clean_output = clean_output.detach().cpu().numpy()
    pred = pred.squeeze(1)
    acc_ori = len([i for i in range(len(pred)) if labels[i] == pred[i]]) / len(labels)

    asr = []
    pa_acc = []
    n_bit = []
    n_stop = []
    param_lam = []
    param_k_bits = []

    print("Attack Start")
    for i, (target_class, attack_idx) in enumerate(attack_info):

        source_class = int(labels[attack_idx])
        aux_idx = np.random.choice([i for i in range(len(labels)) if i != attack_idx], args.n_aux, replace=False)

        suc = False

        cur_k = args.init_k
        for search_k in range(args.max_search_k):

            cur_lam = args.init_lam

            for search_lam in range(args.max_search_lam):

                res = attack(auglag, all_data, labels, labels_cuda, clean_output,
                             attack_idx, target_class, source_class, aux_idx,
                             cur_lam, cur_k, args)

                if res == -1:
                    print("Error[{0}]: Lambda:{1} K_bits:{2}".format(i, cur_lam, cur_k))
                    cur_lam = cur_lam / 2.0
                    continue
                elif res["suc"]:
                    n_stop.append(int(res["stop"]))
                    asr.append(int(res["suc"]))
                    pa_acc.append(res["pa_acc"])
                    n_bit.append(res["n_bit"])
                    param_lam.append(cur_lam)
                    param_k_bits.append(cur_k)
                    suc = True
                    break

                cur_lam = cur_lam / 2.0

            if suc:
                break

            cur_k = cur_k * 2.0

        if not suc:
            asr.append(0)
            n_stop.append(0)
            print("[{0}] Fail!".format(i))
        else:
            print("[{0}] PA-ACC:{1:.4f} Success:{2} N_flip:{3} Stop:{4} Lambda:{5} K:{6}".format(
                i, pa_acc[-1]*100, bool(asr[-1]), n_bit[-1], bool(n_stop[-1]), param_lam[-1],
                param_k_bits[-1]))

        if (i+1) % 50 == 0:
            print("END[0] PA-ACC:{1:.4f} ASR:{2} N_flip:{3:.4f}".format(
                i, np.mean(pa_acc)*100, np.mean(asr)*100, np.mean(n_bit)))

    print("END Original_ACC:{0:.4f} PA_ACC:{1:.4f} ASR:{2:.2f} N_flip:{3:.4f}".format(
            acc_ori*100, np.mean(pa_acc)*100, np.mean(asr)*100, np.mean(n_bit)))


if __name__ == '__main__':
    main()
