import torch
from art.estimators.regression.scikitlearn import ScikitlearnRegressor

class CustomScikitlearnRegressor(ScikitlearnRegressor):
    def __init__(self, device_type, **kwargs):
        self._device_type = device_type
        super().__init__(**kwargs)

    def _to_data_parallel(self):
        if torch.cuda.device_count() > 1 and self._device_type == 'gpu':
            try:
                self._model = torch.nn.DataParallel(self._model)
                assert isinstance(self._model, torch.nn.DataParallel)
            except InterruptedError:
                print('The current model cannot be parallelized.')

