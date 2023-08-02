import torch


def compute_loss(out, target):
    out = out.reshape(-1, out.shape[-1])
    target = target.reshape(-1)
    loss = torch.nn.functional.cross_entropy(out, target)
    return loss

