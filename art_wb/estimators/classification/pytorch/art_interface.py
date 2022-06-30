import torch
from art.estimators.classification.pytorch import PyTorchClassifier

class CustomPyTorchClassifier(PyTorchClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _to_data_parallel(self):
        if torch.cuda.device_count() > 1 and self._device_type == 'gpu':
            try:
                self._model = torch.nn.DataParallel(self._model)
                assert isinstance(self._model, torch.nn.DataParallel)
            except InterruptedError:
                print('The current model cannot be parallelized.')

    def _get_last_layer_outs(self, x: torch.Tensor):
        return self._model(x)[-1]
