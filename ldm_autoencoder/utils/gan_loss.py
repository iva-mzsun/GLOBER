import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as st

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


# def hinge_d_loss(logits_real, logits_fake):
#     loss_real = torch.mean(F.relu(1. - logits_real))
#     loss_fake = torch.mean(F.relu(1. + logits_fake))
#     d_loss = 0.5 * (loss_real + loss_fake)
#     return d_loss

# class hinge_d_loss(nn.Module):
#     def __init__(self):
#         super(hinge_d_loss, self).__init__()
#
#     def forward(self, logits, label):
#         # label=1: real sample
#         # label=0: fake sample
#         weight = label.detach() * 2 - 1  # 1: real, -1: fake
#         d_loss = torch.mean(F.relu(1. - logits * weight))
#         return d_loss

def hinge_d_loss(logits, label):
    # label=1: real sample
    # label=0: fake sample
    weight = label.detach() * 2 - 1  # 1: real, -1: fake
    d_loss = torch.mean(F.relu(1. - logits * weight))
    return d_loss

# def hinge_d_loss(logits_real, logits_fake):
#     loss_real = torch.mean(F.relu(1. - logits_real))
#     loss_fake = torch.mean(F.relu(1. + logits_fake))
#     d_loss = 0.5 * (loss_real + loss_fake)
#     return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss