import torch
import torch.nn as nn
import torch.nn.functional as F


class RaoLoss(torch.nn.Module):
    def __init__(self):
        super(RaoLoss, self).__init__()

    def forward(self, preds, nat):
        eps = 1e-6
        sqroot_prod = ((nat.softmax(1) * preds.softmax(1)) ** 0.5).sum(1)
        rao = (torch.acos(torch.clamp(sqroot_prod - eps, 0, 1))).sum(0)
        return rao


class gLoss(torch.nn.Module):
    def __init__(self):
        super(gLoss, self).__init__()

    def forward(self, preds):
        pyx = (preds.softmax(1) ** 2).sum(1)
        return (1 - (pyx ** 0.5)).sum(0)


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, preds, true):
        true = true.long()
        return nn.CrossEntropyLoss()(preds, true)


class KLLoss(torch.nn.KLDivLoss):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, preds, nat):
        criterion_kl = nn.KLDivLoss(reduction='sum')
        loss_kl = criterion_kl(F.log_softmax(preds, dim=1), F.softmax(nat, dim=1))
        return loss_kl


def _get_loss_by_name(loss_name: str):
    assert loss_name in ['CE', 'KL', 'Rao', 'g', 'BCE']
    if loss_name == 'CE':
        return CrossEntropyLoss()
    elif loss_name == 'KL':
        return KLLoss()
    elif loss_name == 'Rao':
        return RaoLoss()
    elif loss_name == 'g':
        return gLoss()
    elif loss_name == 'BCE':
        return torch.nn.BCEWithLogitsLoss()


def global_loss(loss_name, preds, nat, y):
    assert loss_name in ['CE', 'KL', 'Rao', 'g']

    loss = _get_loss_by_name(loss_name=loss_name)

    if loss_name == 'CE':
        return loss(preds, y)
    elif loss_name == 'KL':
        return loss(preds, nat)
    elif loss_name == 'Rao':
        return loss(preds, nat)
    elif loss_name == 'g':
        return loss(preds)
