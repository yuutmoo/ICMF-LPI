import torch
import torch.nn as nn
import torch.nn.functional as F


class GHMCLoss(nn.Module):
    def __init__(self, bins=10, momentum=0):
        super(GHMCLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] += 1e-6
        self.acc_sum = torch.zeros(bins)

    def forward(self, pred, target):
        target_one_hot = F.one_hot(target, num_classes=2).float()
        probs = pred

        grad = torch.abs(probs.detach() - target_one_hot)

        tot = pred.size(0)
        n = 0
        weights = torch.zeros_like(pred)

        for i in range(self.bins):
            inds = (grad >= self.edges[i]) & (grad < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                    weights[inds] = tot / (self.acc_sum[i] + 1e-6)
                else:
                    weights[inds] = tot / num_in_bin
                n += 1

        if n > 0:
            weights = weights / n

        loss = F.nll_loss(pred, target, reduction='none')

        weighted_loss = (loss * weights.mean(dim=1)).sum() / tot

        return weighted_loss
