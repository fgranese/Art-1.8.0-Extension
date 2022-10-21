import os
import logging
import numpy as np
from detectors.hamper.data_depth import DataDepth, sampled_sphere
from tqdm import tqdm

def depth_by_class(depth, X_train, X_test, y_train, c, layer, U=None):
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)
    X_train_c = X_train[np.where(y_train == c)]
    res = depth.halfspace_mass(X=X_train_c, X_test=X_test, U=U)
    return res

def merge_layers_from_dict(dict, num_classes, layers_names):
    depths = np.zeros((dict[next(iter(dict))][0].shape[0], len(layers_names), num_classes))
    for i in range(len(layers_names)):
        layer = layers_names[i]
        for c in range(num_classes):
            depths[:, i, c] = dict[layer][c]
    depths = np.reshape(depths, [-1, len(layers_names) * num_classes])
    return depths

def depth_from_dict(dict_train, y_train, dict_test, K, layers, num_classes):
    from collections import defaultdict
    #logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    features = defaultdict(list)
    depth = DataDepth(K)
    depth_res = {}

    for i in range(len(layers)):
        layer = layers[i]
        depth_res[layer] = []
        print('Depth from dict resnet', layer)
        for c in tqdm(range(num_classes), ncols=70, colour='yellow', ascii=True):
            X_test = dict_test[layer]
            #logging.info(X_test.shape)
            X_train = dict_train[layer]
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            _, dim = X_train.shape
            path = '../detectors/results/depth/U_{}.npy'.format(dim)
            if os.path.exists(path):
                U = np.load(path)
            else:
                from pathlib import Path
                Path('../detectors/results/depth/').mkdir(parents=True, exist_ok=True)
                U = sampled_sphere(K, dim)
                np.save(path, U)
            res = depth_by_class(depth=depth, X_train=X_train, X_test=X_test, y_train=y_train, c=c, U=U, layer=layer)
            depth_res[layer].append(res)
        logging.info(layer)
    return depth_res