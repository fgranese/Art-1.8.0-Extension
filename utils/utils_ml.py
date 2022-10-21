import numpy as np
import torch
from tqdm import tqdm
from typing import Union
import torch.nn.functional as trchfnctnl
from art_wb.estimators.classification.pytorch.art_interface import CustomPyTorchClassifier, PyTorchClassifier
from art_wb.estimators.classification.scikitlearn.art_interface import CustomScikitlearnSVC, ScikitlearnSVC
from art_wb.estimators.regression.scikitlearn.art_interface import CustomScikitlearnRegressor, ScikitlearnRegressor


def compute_accuracy(predictions: torch.tensor, targets: torch.tensor):
    """
    compute the model's accuracy
    :param predictions: tensor containing the predicted labels
    :param targets: tensor containing the target labels
    :return: the accuracy in [0, 1]
    """
    accuracy = torch.div(torch.sum(predictions == targets), len(targets))
    return accuracy


def compute_logits_return_labels_and_predictions(model: Union[torch.nn.Module, PyTorchClassifier, CustomPyTorchClassifier],
                                                 dataloader, device=torch.device("cpu"),
                                                 *args, **kwargs):
    """
    compute the logits given input data loader and model
    :param model: model utilized for the logits computation
    :param dataloader: loader for the training data
    :param device: device used for computation
    :return: logits and targets
    """
    logits = []
    labels = []
    predictions = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, ascii=True, ncols=50, colour='red')):
            # print('m', data[0])
            # exit()
            if type(model) == torch.nn.Module:
                preds_logit = model(data.to(device))
            # isintance() considers every instance an instance of the parent class as well
            elif type(model) in [PyTorchClassifier, CustomPyTorchClassifier]:
                preds_logit = model._get_last_layer_outs(data.to(device))
            elif type(model) in [ScikitlearnSVC, CustomScikitlearnSVC]:
                preds_logit = torch.Tensor(model.predict(data))
            elif type(model) in [CustomScikitlearnRegressor, ScikitlearnRegressor]:
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                preds_logit = torch.Tensor(model.predict(data))
            else:
                raise NotImplementedError

            logits.append(preds_logit.detach().cpu())

            if type(model) in [ScikitlearnSVC, CustomScikitlearnSVC, CustomScikitlearnRegressor, ScikitlearnRegressor]:
                soft_prob = preds_logit
            elif preds_logit.shape[1] >= 2:
                soft_prob = trchfnctnl.softmax(preds_logit, dim=1)
            else:
                soft_prob = trchfnctnl.sigmoid(preds_logit)

            if 'print_sp' in kwargs:
                if kwargs['print_sp']:
                    print(soft_prob)
            if len(preds_logit.shape) == 1 or (len(preds_logit.shape) > 1 and preds_logit.shape[1] == 1):
                if type(model) in [CustomScikitlearnRegressor, ScikitlearnRegressor]:
                    threshold = kwargs.get('threshold')
                    preds = torch.where(soft_prob > threshold, 1, 0)
                else:
                    preds = torch.round(soft_prob)
            else:
                preds = torch.argmax(soft_prob, dim=1)

            predictions.append(preds.detach().cpu().reshape(-1, 1))

            labels.append(target.detach().cpu().reshape(-1, 1))

    logits = torch.vstack(logits)
    labels = torch.vstack(labels).reshape(-1)
    predictions = torch.vstack(predictions).reshape(-1)
    return logits, labels, predictions
