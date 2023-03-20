import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import pickle
import numpy as np
import argparse

from data_mi import mi_dataset

parser = argparse.ArgumentParser("MI")
parser.add_argument("--feature_path", type=str, default="D://projects//open_cross_entropy//osr_closed_set_all_you_need-main//features//tinyimagenet_msp_optimal_classifier32_0")
parser.add_argument("--model_name", type=str, default="cifar-10-10_msp_optimal_classifier32_0")
parser.add_argument("--feature_name", type=str, default="module.avgpool")

parser.add_argument("--critic", type=str, default="concate")
parser.add_argument("--baseline", type=str, default="constant")
parser.add_argument("--feature_dim", type=int, default=128)
parser.add_argument("--label_dim", type=int, default=20)
parser.add_argument("--hidden_dim", type=int, default=50)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--alpha", type=float, default=0.01)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--save_path", type=str, default="D://projects//open_cross_entropy//osr_closed_set_all_you_need-main//mi.pth")
parser.add_argument("--milestones", type=list, default=[50, 100])
parser.add_argument("--gamma", type=float, default=0.5)

device = torch.device("cpu")
"""
Models for mi estimation
"""

def log_interpolate(log_a, log_b, alpha_logit):
    # replace alfa with 1./(1.+exp(-alfa)), 1-alfa with 1./(1.+exp(alfa))
    #print("alpha_logit", log_a.shape, log_b.shape)
    #alpha_logit = torch.tensor(alpha_logit)
    #alpha_minus_logit = torch.tensor(-alpha_logit)
    #log_alpha = -nn.Softplus(alpha_minus_logit)
    #log_1_minus_alpha = -nn.Softplus(alpha_logit)
    log_alpha = torch.log(torch.tensor(alpha_logit))
    log_1_minus_alpha = torch.log(torch.tensor(1-alpha_logit))
    y = torch.logsumexp(torch.stack((log_alpha+log_a, log_1_minus_alpha+log_b)), axis=0)
    return y


def compute_log_loomean(scores):
    # nce_baseline 
    # This is a numerically stable version of:
    #log_loosum = scores + tfp.math.softplus_inverse(tf.reduce_logsumexp(scores, axis=1, keepdims=True) - scores) 
    # reture a matrix of size [batch_size, batch_size]
    max_scores, _ = torch.max(scores, dim=1, keepdim=True)
    lse_minus_max = torch.logsumexp(scores-max_scores, axis=1, keepdim=True)
    d = lse_minus_max + (max_scores - scores)
    d_ok = torch.eq(d, 0.)
    safe_d = torch.where(d_ok, d, torch.ones_like(d))
    loo_lse = scores + softplus_inverse(safe_d)
    # normalize to get the leave one out mean exp
    loo_me = loo_lse - torch.log(torch.tensor(scores.shape[1]) - 1.)

    return loo_me


def interpolated_lower_bound(scores, bashline, alpha_logit):
    batch_size = scores.shape[0]
    # compute InfoNCE baseline
    nce_baseline = compute_log_loomean(scores)
    # interpolated baseline interpolates the InfoNCE baseline with a learned baseline
    interpolated_baseline = log_interpolate(nce_baseline, torch.tile(bashline[:,None], (1, batch_size)), alpha_logit)     # TODO check

    # Marginal term
    # exp(log(...))
    critic_margin = scores - torch.diag(interpolated_baseline)[:, None]
    marg_term = torch.exp(reduce_logmeanexp_nodiag(critic_margin))

    # joint term
    critic_joint = torch.diag(scores)[:, None] - interpolated_baseline
    # all column except diagonal
    joint_term = (torch.sum(critic_joint) - torch.sum(torch.diag(critic_joint))) / (batch_size*(batch_size-1))

    return 1 + joint_term - marg_term


def reduce_logmeanexp_nodiag(x, dim=None):
    # average the exponentiated critic over all independent samples, 
    # corresponding to the off-diagonal terms in scores
    batch_size = x.shape[0]
    logsumexp = torch.logsumexp(x - torch.diag(np.inf * torch.ones(batch_size)), dim=(0, 1), keepdim=False)    ####
    if dim:
        num_elem = batch_size - 1
    else:
        num_elem = batch_size * (batch_size - 1)
    num_elem = torch.tensor(num_elem * 1.0)

    return logsumexp - torch.log(num_elem)


def softplus_inverse(x):
    return torch.log(torch.expm1(x))


class mi_mlp_seperate(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.hidden_dim = hidden_dim

        self.g = self.mlp(layers)
        self.h = self.mlp(layers)


    def forward(self, x, y):
        scores = torch.matmul(self.g(x), self.h(y))
        return scores

    
    def mlp(self, layers):
        ml = [nn.Linear(self.in_dim, self.hidden_dim), nn.ReLU()]
        for i in range(layers-2):
            ml.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            ml.append(nn.ReLU())
        ml.append(nn.Linear(self.hidden_dim, self.out_dim))
        linears = nn.ModuleList(ml)

        return linears


class mi_mlp_concat(nn.Module):
    # In the setting of x continuos and y discrete (labels)
    # y is one hot code of the labels, but can it just be scalars
    # The returned scores is a matrix of shape [bz, bz]
    def __init__(self, in_dim, hidden_dim, layers):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.hidden_dim = hidden_dim

        self.f = self.mlp(layers)


    def forward(self, x, y):
        batch_size = x.shape[0]
        # Tile all possible combinations of x and y
        batch_size = x.shape[0]
        x_tiled = torch.tile(x[None, :],  (batch_size, 1, 1))
        y_tiled = torch.tile(y[:, None],  (1, batch_size, 1))
        # one x is paired with all y
        xy_pairs = torch.reshape(torch.concat((x_tiled, y_tiled), axis=2), [batch_size * batch_size, -1])
        xy_pairs = xy_pairs.float()
        for i, l in enumerate(self.f):
            xy_pairs = l(xy_pairs)
        scores = torch.reshape(xy_pairs, (batch_size, batch_size))
        return scores

    
    def mlp(self, layers):
        ml = [nn.Linear(self.in_dim, self.hidden_dim), nn.ReLU()]
        for i in range(layers-2):
            ml.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            ml.append(nn.ReLU())
        ml.append(nn.Linear(self.hidden_dim, self.out_dim))
        linears = nn.ModuleList(ml)

        return linears



def estimate_mutual_information(x, y, critic_fn, alpha_logit=None,
                                baseline_fn=None):

    scores = critic_fn(x, y)
    if baseline_fn is not None:
        log_baseline = baseline_fn(y)
        log_baseline = log_baseline.to(device)
    mi = interpolated_lower_bound(scores, log_baseline, alpha_logit)

    return mi

# baselines
# we select constant for the baseline a(y)
def constant_baseline(x):
    # shape of output: [batch_size]
    batch_size = x.shape[0]
    e = torch.ones([batch_size,], requires_grad=False)
    e = np.exp(1.) * e
    return e


def train(dataset, args):

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            num_workers=4, shuffle=True)
    
    in_dim = args.feature_dim + args.label_dim

    if args.critic == "concate":
        _critic = mi_mlp_concat(in_dim=in_dim, hidden_dim=args.hidden_dim, layers=args.layers)
    else:
        _critic = mi_mlp_seperate(in_dim=in_dim, hidden_dim=args.hidden_dim, layers=args.layers)

    _critic.train()
    _critic = _critic.to(device)

    if args.baseline == "constant":
        _baseline = constant_baseline

    optimizer = torch.optim.Adam(params=_critic.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    
    losses = []
    loss_smallest = 1e10
    for epoch in range(args.epochs):
        loss_epoch = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            mi = estimate_mutual_information(x, y, critic_fn=_critic, 
                                             alpha_logit=args.alpha, 
                                             baseline_fn=_baseline)
            loss = -mi
            loss.backward()
            optimizer.step()

            loss_epoch += loss.detach().cpu().item()

        losses.append(loss_epoch)
        scheduler.step()
        loss_avg = loss_epoch / args.batch_size
        if loss_avg < loss_smallest:
            loss_smallest = loss_avg
        print("Epoch ", epoch, "loss is ", loss_epoch, "Avaerage is ", loss_avg)

    model_path = os.path.join(args.save_path, args.model_name + "_" + args.feature_name + "_mi.pth") 
    loss_path = os.path.join(args.save_path, args.model_name + "_" + args.feature_name + "_mi_losses")
    print(model_path)
    print(loss_path)
    torch.save(_critic.cpu().state_dict(), model_path)
    print("Smallest loss is ", loss_smallest) 
    with open(loss_path, "wb") as f:
        pickle.dump(losses, f)

    return losses


if __name__ == "__main__":
    
    args = parser.parse_args()
    dataset = mi_dataset(feature_path=args.feature_path, 
                         feature_name=args.feature_name)

    losses = train(dataset, args)