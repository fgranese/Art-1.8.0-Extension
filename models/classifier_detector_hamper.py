import torch
import torch.nn as nn
import torch.nn.functional as F
from depth.utils import depth_from_dict, merge_layers_from_dict
from detectors.hamper.utils import extraction_resnet
import pickle
import logging
import numpy as np

from sklearn import preprocessing


class Detector(nn.Module):
    def __init__(self, path, model, layers_dict_train, y_train, layers, num_classes, K=10000, bs=100, dataset='cifar10'):
        super(Detector, self).__init__()
        self.detect = pickle.load(open(path, 'rb'))
        self.model = model
        self.y_train = y_train
        self.layers = layers
        self.num_classes = num_classes
        self.K = K
        self.layers_dict_train = layers_dict_train
        self.bs = bs
        self.dataset = dataset

    def forward(self, x):
        layers_dic_test = extraction_resnet(x, self.model, bs=self.bs)
        depth_dic = depth_from_dict(self.model, self.layers_dict_train, self.y_train, layers_dic_test, self.K, self.layers, self.num_classes, self.dataset)
        np.save('depth_dic.npy', depth_dic)
        combined_layers = merge_layers_from_dict(depth_dic, self.num_classes, self.layers)  # Combine layers
        combined_layers = preprocessing.normalize(combined_layers)
        with torch.no_grad():
            out = self.detect.predict(combined_layers)
        out = torch.tensor(out)
        return out


class Extract_Depth(nn.Module):
    def __init__(self, model, layers_dict_train, y_train, layers, num_classes, K=10000, bs=100, dataset='cifar10'):
        super(Extract_Depth, self).__init__()
        self.model = model
        self.y_train = y_train
        self.layers = layers
        self.num_classes = num_classes
        self.K = K
        self.layers_dict_train = layers_dict_train
        self.bs = bs
        self.dataset = dataset

    def forward(self, x):
        layers_dic_test = extraction_resnet(x, self.model, bs=self.bs)
        depth_dic = depth_from_dict(self.model, self.layers_dict_train, self.y_train, layers_dic_test, self.K, self.layers, self.num_classes, self.dataset)
        return depth_dic