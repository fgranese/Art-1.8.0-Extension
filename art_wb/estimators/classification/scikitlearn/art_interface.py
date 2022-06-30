import numpy as np
import torch
from art.estimators.classification.scikitlearn import ScikitlearnSVC

class CustomScikitlearnSVC(ScikitlearnSVC):
    def __init__(self, device_type, **kwargs):
        kwargs['model'].kernel = 'linear' # sigmoid not implemented but for the experiments we do not need gradients
        kwargs['model'].probability = True
        self._device_type = device_type
        super().__init__(**kwargs)

    def _to_data_parallel(self):
        if torch.cuda.device_count() > 1 and self._device_type == 'gpu':
            try:
                self._model = torch.nn.DataParallel(self._model)
                assert isinstance(self._model, torch.nn.DataParallel)
            except InterruptedError:
                print('The current model cannot be parallelized.')

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        pass
