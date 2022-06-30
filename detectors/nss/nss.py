import os
import numpy as np
from tqdm import tqdm
from detectors.nss.MSCN import calculate_brisque_features

def extract_nss_features(data: np.ndarray):

    adv_data_f = np.array([])
    for img in data:
        parameters = calculate_brisque_features(img)
        parameters = parameters.reshape((1, -1))
        if adv_data_f.size == 0:
            adv_data_f = parameters
        else:
            adv_data_f = np.concatenate((adv_data_f, parameters), axis=0)

    return adv_data_f

