import inspect
import logging
import math
import random
import torch
from typing import Optional, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format, get_labels_np_array
from art.attacks.evasion import SquareAttack

logger = logging.getLogger(__name__)


def adv_criterion(logits_class, y_class, logits_det):
    y_pred_class = np.argmax(torch.softmax(logits_class, dim=1).detach().cpu().numpy(), axis=1)
    y_pred_det = np.round(torch.sigmoid(logits_det).detach().cpu().numpy())
    y_class = np.argmax(y_class, axis=1)
    y_det = np.where(y_pred_class == y_class, 0, 1)
    return (y_pred_class != y_class) & (y_pred_det.reshape(y_det.shape) != y_det)


class SquareAttack_WB_network(SquareAttack):
    def __init__(self, detectors_dict: dict, classifier_loss_name: str, **kwargs):

        assert len(detectors_dict['dtctrs']) > 0, 'At least one detector must be passed'
        lsts = [detectors_dict['dtctrs'], detectors_dict['alphas'], detectors_dict['loss_dtctrs']]
        if not all(len(lsts[0]) == len(l) for l in lsts[1:]):
            raise ValueError('The lists have different lengths in: {}'.format(inspect.stack()[1][3]))

        self.detectors_list = detectors_dict['dtctrs']
        self.alphas_list = detectors_dict['alphas']
        self.loss_dtctrs_list = detectors_dict['loss_dtctrs']
        self.classifier_loss_name = classifier_loss_name
        kwargs['adv_criterion'] = adv_criterion

        super().__init__(**kwargs)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`. (classifier)
        :return: An array holding the adversarial examples.
        """
        if x.ndim != 4:
            raise ValueError("Unrecognized input dimension. Attack can only be applied to image data.")

        x_adv = x.astype(ART_NUMPY_DTYPE)
        y_class = y

        if isinstance(self.estimator, ClassifierMixin):
            y_class = check_and_transform_label_format(y_class, self.estimator.nb_classes)

        if y_class is None:
            # Use model predictions as true labels
            logger.info("Using model predictions as true labels.")
            y_class = self.estimator.predict(x, batch_size=self.batch_size)
            if isinstance(self.estimator, ClassifierMixin):
                y_class = get_labels_np_array(y_class)

        if isinstance(self.estimator, ClassifierMixin):
            if self.estimator.nb_classes == 2 and y_class.shape[1] == 1:
                raise ValueError(
                    "This attack has not yet been tested for binary classification with a single output classifier."
                )

        if self.estimator.channels_first:
            channels = x.shape[1]
            height = x.shape[2]
            width = x.shape[3]
        else:
            height = x.shape[1]
            width = x.shape[2]
            channels = x.shape[3]

        for _ in trange(self.nb_restarts, desc="SquareAttack - restarts", disable=not self.verbose, ascii=True):

            # Determine correctly predicted samples classifier and detector (predict returns logits)
            logits_class = torch.Tensor(self.estimator.predict(x_adv, batch_size=self.batch_size))
            detector = self.detectors_list[0]
            logits_det = torch.Tensor(detector.predict(logits_class, batch_size=self.batch_size))
            sample_is_robust = np.logical_not(self.adv_criterion(logits_class, y_class, logits_det))

            if np.sum(sample_is_robust) == 0:
                break

            # x_robust = x_adv[sample_is_robust]
            x_robust = x[sample_is_robust]
            y_robust = y_class[sample_is_robust]
            sample_loss_init = self.loss(x_robust, y_robust)

            if self.norm in [np.inf, "inf"]:

                if self.estimator.channels_first:
                    size = (x_robust.shape[0], channels, 1, width)
                else:
                    size = (x_robust.shape[0], 1, width, channels)

                # Add vertical stripe perturbations
                x_robust_new = np.clip(
                    x_robust + self.eps * np.random.choice([-1, 1], size=size),
                    a_min=self.estimator.clip_values[0],
                    a_max=self.estimator.clip_values[1],
                ).astype(ART_NUMPY_DTYPE)

                sample_loss_new = self.loss(x_robust_new, y_robust)
                loss_improved = (sample_loss_new - sample_loss_init) < 0.0

                x_robust[loss_improved] = x_robust_new[loss_improved]

                x_adv[sample_is_robust] = x_robust

                for i_iter in trange(
                        self.max_iter, desc="SquareAttack - iterations", leave=False, disable=not self.verbose, ascii=True
                ):

                    percentage_of_elements = self._get_percentage_of_elements(i_iter)

                    # Determine correctly predicted samples
                    logits_class = torch.Tensor(self.estimator.predict(x_adv, batch_size=self.batch_size))
                    detector = self.detectors_list[0]
                    logits_det = torch.Tensor(detector.predict(logits_class, batch_size=self.batch_size))
                    sample_is_robust = np.logical_not(self.adv_criterion(logits_class, y_class, logits_det))

                    # y_pred = self.estimator.predict(x_adv, batch_size=self.batch_size)
                    # sample_is_robust = np.logical_not(self.adv_criterion(y_pred, y))

                    if np.sum(sample_is_robust) == 0:
                        break

                    x_robust = x_adv[sample_is_robust]
                    x_init = x[sample_is_robust]
                    y_robust = y_class[sample_is_robust]

                    sample_loss_init = self.loss(x_robust, y_robust)

                    height_tile = max(int(round(math.sqrt(percentage_of_elements * height * width))), 1)

                    height_mid = np.random.randint(0, height - height_tile)
                    width_start = np.random.randint(0, width - height_tile)

                    delta_new = np.zeros(self.estimator.input_shape)

                    if self.estimator.channels_first:
                        delta_new[
                        :, height_mid: height_mid + height_tile, width_start: width_start + height_tile
                        ] = np.random.choice([-2 * self.eps, 2 * self.eps], size=[channels, 1, 1])
                    else:
                        delta_new[
                        height_mid: height_mid + height_tile, width_start: width_start + height_tile, :
                        ] = np.random.choice([-2 * self.eps, 2 * self.eps], size=[1, 1, channels])

                    x_robust_new = x_robust + delta_new

                    x_robust_new = np.minimum(np.maximum(x_robust_new, x_init - self.eps), x_init + self.eps)

                    x_robust_new = np.clip(
                        x_robust_new, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1]
                    ).astype(ART_NUMPY_DTYPE)

                    sample_loss_new = self.loss(x_robust_new, y_robust)
                    loss_improved = (sample_loss_new - sample_loss_init) < 0.0

                    x_robust[loss_improved] = x_robust_new[loss_improved]

                    x_adv[sample_is_robust] = x_robust

            elif self.norm == 2:

                n_tiles = 5

                height_tile = height // n_tiles

                def _get_perturbation(height):
                    delta = np.zeros([height, height])
                    gaussian_perturbation = np.zeros([height // 2, height])

                    x_c = height // 4
                    y_c = height // 2

                    for i_y in range(y_c):
                        gaussian_perturbation[
                        max(x_c, 0): min(x_c + (2 * i_y + 1), height // 2),
                        max(0, y_c): min(y_c + (2 * i_y + 1), height),
                        ] += 1.0 / ((i_y + 1) ** 2)
                        x_c -= 1
                        y_c -= 1

                    gaussian_perturbation /= np.sqrt(np.sum(gaussian_perturbation ** 2))

                    delta[: height // 2] = gaussian_perturbation
                    delta[height // 2: height // 2 + gaussian_perturbation.shape[0]] = -gaussian_perturbation

                    delta /= np.sqrt(np.sum(delta ** 2))

                    if random.random() > 0.5:
                        delta = np.transpose(delta)

                    if random.random() > 0.5:
                        delta = -delta

                    return delta

                delta_init = np.zeros(x_robust.shape, dtype=ART_NUMPY_DTYPE)

                height_start = 0
                for _ in range(n_tiles):
                    width_start = 0
                    for _ in range(n_tiles):
                        if self.estimator.channels_first:
                            perturbation_size = (1, 1, height_tile, height_tile)
                            random_size = (x_robust.shape[0], channels, 1, 1)
                        else:
                            perturbation_size = (1, height_tile, height_tile, 1)
                            random_size = (x_robust.shape[0], 1, 1, channels)

                        perturbation = _get_perturbation(height_tile).reshape(perturbation_size) * np.random.choice(
                            [-1, 1], size=random_size
                        )

                        if self.estimator.channels_first:
                            delta_init[
                            :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
                            ] += perturbation
                        else:
                            delta_init[
                            :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
                            ] += perturbation
                        width_start += height_tile
                    height_start += height_tile

                x_robust_new = np.clip(
                    x_robust + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps,
                    self.estimator.clip_values[0],
                    self.estimator.clip_values[1],
                )

                sample_loss_new = self.loss(x_robust_new, y_robust)
                loss_improved = (sample_loss_new - sample_loss_init) < 0.0

                x_robust[loss_improved] = x_robust_new[loss_improved]

                x_adv[sample_is_robust] = x_robust

                for i_iter in trange(
                        self.max_iter, desc="SquareAttack - iterations", leave=False, disable=not self.verbose, ascii=True
                ):

                    percentage_of_elements = self._get_percentage_of_elements(i_iter)

                    # Determine correctly predicted samples
                    logits_class = torch.Tensor(self.estimator.predict(x_adv, batch_size=self.batch_size))
                    detector = self.detectors_list[0]
                    logits_det = torch.Tensor(detector.predict(logits_class, batch_size=self.batch_size))
                    sample_is_robust = np.logical_not(self.adv_criterion(logits_class, y_class, logits_det))
                    # y_pred = self.estimator.predict(x_adv, batch_size=self.batch_size)
                    # sample_is_robust = np.logical_not(self.adv_criterion(y_pred, y))

                    if np.sum(sample_is_robust) == 0:
                        break

                    x_robust = x_adv[sample_is_robust]
                    x_init = x[sample_is_robust]
                    y_robust = y[sample_is_robust]

                    sample_loss_init = self.loss(x_robust, y_robust)

                    delta_x_robust_init = x_robust - x_init

                    height_tile = max(int(round(math.sqrt(percentage_of_elements * height * width))), 3)

                    if height_tile % 2 == 0:
                        height_tile += 1
                    height_tile_2 = height_tile

                    height_start = np.random.randint(0, height - height_tile)
                    width_start = np.random.randint(0, width - height_tile)

                    new_deltas_mask = np.zeros(x_init.shape)
                    if self.estimator.channels_first:
                        new_deltas_mask[
                        :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
                        ] = 1.0
                        w_1_norm = np.sqrt(
                            np.sum(
                                delta_x_robust_init[
                                :,
                                :,
                                height_start: height_start + height_tile,
                                width_start: width_start + height_tile,
                                ]
                                ** 2,
                                axis=(2, 3),
                                keepdims=True,
                            )
                        )
                    else:
                        new_deltas_mask[
                        :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
                        ] = 1.0
                        w_1_norm = np.sqrt(
                            np.sum(
                                delta_x_robust_init[
                                :,
                                height_start: height_start + height_tile,
                                width_start: width_start + height_tile,
                                :,
                                ]
                                ** 2,
                                axis=(1, 2),
                                keepdims=True,
                            )
                        )

                    height_2_start = np.random.randint(0, height - height_tile_2)
                    width_2_start = np.random.randint(0, width - height_tile_2)

                    new_deltas_mask_2 = np.zeros(x_init.shape)
                    if self.estimator.channels_first:
                        new_deltas_mask_2[
                        :,
                        :,
                        height_2_start: height_2_start + height_tile_2,
                        width_2_start: width_2_start + height_tile_2,
                        ] = 1.0
                    else:
                        new_deltas_mask_2[
                        :,
                        height_2_start: height_2_start + height_tile_2,
                        width_2_start: width_2_start + height_tile_2,
                        :,
                        ] = 1.0

                    norms_x_robust = np.sqrt(np.sum((x_robust - x_init) ** 2, axis=(1, 2, 3), keepdims=True))
                    w_norm = np.sqrt(
                        np.sum(
                            (delta_x_robust_init * np.maximum(new_deltas_mask, new_deltas_mask_2)) ** 2,
                            axis=(1, 2, 3),
                            keepdims=True,
                        )
                    )

                    if self.estimator.channels_first:
                        new_deltas_size = [x_init.shape[0], channels, height_tile, height_tile]
                        random_choice_size = [x_init.shape[0], channels, 1, 1]
                        perturbation_size = (1, 1, height_tile, height_tile)
                    else:
                        new_deltas_size = [x_init.shape[0], height_tile, height_tile, channels]
                        random_choice_size = [x_init.shape[0], 1, 1, channels]
                        perturbation_size = (1, height_tile, height_tile, 1)

                    delta_new = (
                            np.ones(new_deltas_size)
                            * _get_perturbation(height_tile).reshape(perturbation_size)
                            * np.random.choice([-1, 1], size=random_choice_size)
                    )

                    if self.estimator.channels_first:
                        delta_new += delta_x_robust_init[
                                     :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
                                     ] / (np.maximum(1e-9, w_1_norm))
                    else:
                        delta_new += delta_x_robust_init[
                                     :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
                                     ] / (np.maximum(1e-9, w_1_norm))

                    diff_norm = (self.eps * np.ones(delta_new.shape)) ** 2 - norms_x_robust ** 2
                    diff_norm[diff_norm < 0.0] = 0.0

                    if self.estimator.channels_first:
                        delta_new /= np.sqrt(np.sum(delta_new ** 2, axis=(2, 3), keepdims=True)) * np.sqrt(
                            diff_norm / channels + w_norm ** 2
                        )
                        delta_x_robust_init[
                        :,
                        :,
                        height_2_start: height_2_start + height_tile_2,
                        width_2_start: width_2_start + height_tile_2,
                        ] = 0.0
                        delta_x_robust_init[
                        :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
                        ] = delta_new
                    else:
                        delta_new /= np.sqrt(np.sum(delta_new ** 2, axis=(1, 2), keepdims=True)) * np.sqrt(
                            diff_norm / channels + w_norm ** 2
                        )
                        delta_x_robust_init[
                        :,
                        height_2_start: height_2_start + height_tile_2,
                        width_2_start: width_2_start + height_tile_2,
                        :,
                        ] = 0.0
                        delta_x_robust_init[
                        :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
                        ] = delta_new

                    x_robust_new = np.clip(
                        x_init
                        + self.eps
                        * delta_x_robust_init
                        / np.sqrt(np.sum(delta_x_robust_init ** 2, axis=(1, 2, 3), keepdims=True)),
                        self.estimator.clip_values[0],
                        self.estimator.clip_values[1],
                    )

                    sample_loss_new = self.loss(x_robust_new, y_robust)
                    loss_improved = (sample_loss_new - sample_loss_init) < 0.0

                    x_robust[loss_improved] = x_robust_new[loss_improved]

                    x_adv[sample_is_robust] = x_robust

        return x_adv
