import torch
import inspect
import logging
import numpy as np
from tqdm import tqdm
from art_wb.attacks.evasion import pgd
from typing import Optional, Union
from art.config import ART_NUMPY_DTYPE
from art.utils import projection, random_sphere
from art.attacks.evasion.fast_gradient import FastGradientMethod

logger = logging.getLogger(__name__)


class FastGradientSignMethod_WB(FastGradientMethod):
    def __init__(self, detectors_dict: dict, classifier_loss_name: str, **kwargs):
        if len(detectors_dict) == 0:
            self.detectors_list = []
            self.alphas_list = []
            self.loss_dtctrs_list = []
        else:
            lsts = [detectors_dict['dtctrs'], detectors_dict['alphas'], detectors_dict['loss_dtctrs']]
            if not all(len(lsts[0]) == len(l) for l in lsts[1:]):
                raise ValueError('The lists have different lengths in: {}'.format(inspect.stack()[1][3]))
            self.detectors_list = detectors_dict['dtctrs']
            self.alphas_list = detectors_dict['alphas']
            self.loss_dtctrs_list = detectors_dict['loss_dtctrs']
        self.classifier_loss_name = classifier_loss_name
        kwargs['num_random_init'] = 1
        super().__init__(**kwargs)

    def _compute_perturbation(self, x: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray], x_init) -> np.ndarray:
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        # grad = self.estimator.loss_gradient(x, y) * (1 - 2 * int(self.targeted))

        grad = pgd.get_composite_gradient(classifier=self.estimator,
                                          classifier_loss_name=self.classifier_loss_name,
                                          detectors_list=self.detectors_list,
                                          alphas_list=self.alphas_list,
                                          loss_dtctrs_list=self.loss_dtctrs_list,
                                          x=x,
                                          x_init=x_init,
                                          y=y)

        # Write summary
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.update(
                batch_id=self._batch_id,
                global_step=self._i_max_iter,
                grad=grad,
                patch=None,
                estimator=self.estimator,
                x=x,
                y=y,
                targeted=self.targeted,
            )

        # Check for NaN before normalisation an replace with 0
        if type(grad) == torch.Tensor:
            grad = grad.detach().cpu().numpy()
        if grad.dtype != object and np.isnan(grad).any():  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad = np.where(np.isnan(grad), 0.0, grad)
        else:
            for i, _ in enumerate(grad):
                grad_i_array = grad[i].astype(np.float32)
                if np.isnan(grad_i_array).any():
                    grad[i] = np.where(np.isnan(grad_i_array), 0.0, grad_i_array).astype(object)

        # Apply mask
        if mask is not None:
            grad = np.where(mask == 0.0, 0.0, grad)

        # Apply norm bound
        def _apply_norm(grad, object_type=False):
            if (grad.dtype != object and np.isinf(grad).any()) or np.isnan(  # pragma: no cover
                    grad.astype(np.float32)
            ).any():
                logger.info("The loss gradient array contains at least one positive or negative infinity.")

            if self.norm in [np.inf, "inf"]:
                grad = np.sign(grad)
            elif self.norm == 1:
                if not object_type:
                    ind = tuple(range(1, len(x.shape)))
                else:
                    ind = None
                grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            elif self.norm == 2:
                if not object_type:
                    ind = tuple(range(1, len(x.shape)))
                else:
                    ind = None
                grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
            return grad

        if x.dtype == object:
            for i_sample in range(x.shape[0]):
                grad[i_sample] = _apply_norm(grad[i_sample], object_type=True)
                assert x[i_sample].shape == grad[i_sample].shape
        else:
            grad = _apply_norm(grad)

        assert x.shape == grad.shape

        return grad

    def _compute(self, x: np.ndarray, x_init: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray],
                 eps: Union[int, float, np.ndarray], eps_step: Union[int, float, np.ndarray], project: bool,
                 random_init: bool, batch_id_ext: Optional[int] = None, ) -> np.ndarray:
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()
            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            if mask is not None:
                random_perturbation = random_perturbation * (mask.astype(ART_NUMPY_DTYPE))
            x_adv = x.astype(ART_NUMPY_DTYPE) + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            if x.dtype == object:
                x_adv = x.copy()
            else:
                x_adv = x.astype(ART_NUMPY_DTYPE)

        # Compute perturbation with implicit batching
        for batch_id in tqdm(range(int(np.ceil(x.shape[0] / float(self.batch_size)))), ascii=True, desc="FGSM - Batches",
                             leave=True):
            if batch_id_ext is None:
                self._batch_id = batch_id
            else:
                self._batch_id = batch_id_ext
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch_index_2 = min(batch_index_2, x.shape[0])
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels, mask_batch,
                                                      x_init=x_init[batch_index_1:batch_index_2])

            # Compute batch_eps and batch_eps_step
            if isinstance(eps, np.ndarray) and isinstance(eps_step, np.ndarray):
                if len(eps.shape) == len(x.shape) and eps.shape[0] == x.shape[0]:
                    batch_eps = eps[batch_index_1:batch_index_2]
                    batch_eps_step = eps_step[batch_index_1:batch_index_2]

                else:
                    batch_eps = eps
                    batch_eps_step = eps_step

            else:
                batch_eps = eps
                batch_eps_step = eps_step

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, batch_eps_step)

            if project:
                if x_adv.dtype == object:
                    for i_sample in range(batch_index_1, batch_index_2):
                        if isinstance(batch_eps, np.ndarray) and batch_eps.shape[0] == x_adv.shape[0]:
                            perturbation = projection(
                                x_adv[i_sample] - x_init[i_sample], batch_eps[i_sample], self.norm
                            )

                        else:
                            perturbation = projection(x_adv[i_sample] - x_init[i_sample], batch_eps, self.norm)

                        x_adv[i_sample] = x_init[i_sample] + perturbation

                else:
                    perturbation = projection(
                        x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], batch_eps, self.norm
                    )
                    x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv
