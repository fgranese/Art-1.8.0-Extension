from __future__ import absolute_import, division, print_function, unicode_literals

from art.attacks.evasion import CarliniLInfMethod
import numpy as np
import logging
from typing import Optional

from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.utils import (
    compute_success,
    get_labels_np_array,
    tanh_to_original,
    original_to_tanh,
)
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)


class MyCarliniLInfMethod(CarliniLInfMethod):
    def __init__(self, tanh_smoother=0.999999, **kwargs):
        super().__init__(**kwargs)

        self._tanh_smoother = tanh_smoother

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y_val` represents the target labels. Otherwise, the
                  targets are the original class labels.
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if self.estimator.clip_values is not None:
            clip_min_per_pixel, clip_max_per_pixel = self.estimator.clip_values
        else:
            clip_min_per_pixel, clip_max_per_pixel = np.amin(x), np.amax(x)

        # Assert that, if attack is targeted, y_val is provided:
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in trange(nb_batches, desc="C&W L_inf", disable=not self.verbose, ascii=True, ncols=70):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]

            # Determine values for later clipping
            clip_min = np.clip(x_batch - self.eps, clip_min_per_pixel, clip_max_per_pixel)
            clip_max = np.clip(x_batch + self.eps, clip_min_per_pixel, clip_max_per_pixel)

            # The optimization is performed in tanh space to keep the
            # adversarial images bounded from clip_min and clip_max.
            # exit(self._tanh_smoother)
            # print(clip_min)
            # print(clip_max)
            # exit()
            x_batch_tanh = original_to_tanh(x_batch, clip_min + 1e-5, clip_max, self._tanh_smoother)

            # Initialize perturbation in tanh space:
            x_adv_batch = x_batch.copy()
            x_adv_batch_tanh = x_batch_tanh.copy()

            # Initialize optimization:
            z_logits, loss = self._loss(x_adv_batch, y_batch)
            attack_success = loss <= 0
            learning_rate = self.learning_rate * np.ones(x_batch.shape[0])

            for i_iter in range(self.max_iter):
                logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)
                logger.debug("Average Loss: %f", np.mean(loss))

                logger.debug(
                    "Successful attack samples: %i out of %i",
                    int(np.sum(attack_success)),
                    x_batch.shape[0],
                )
                # print(f"Successful attack samples: {np.sum(attack_success)} out of {x_batch.shape[0]}")

                # only continue optimization for those samples where attack hasn't succeeded yet:
                active = ~attack_success
                if np.sum(active) == 0:
                    break

                # compute gradient:
                logger.debug("Compute loss gradient")
                perturbation_tanh = -self._loss_gradient(
                    z_logits[active],
                    y_batch[active],
                    x_adv_batch[active],
                    x_adv_batch_tanh[active],
                    clip_min[active],
                    clip_max[active],
                )

                # perform line search to optimize perturbation
                # first, halve the learning rate until perturbation actually decreases the loss:
                prev_loss = loss.copy()
                best_loss = loss.copy()
                best_lr = np.zeros(x_batch.shape[0])
                halving = np.zeros(x_batch.shape[0])

                # print(self.max_halving)

                for i_halve in range(self.max_halving):
                    logger.debug(
                        "Perform halving iteration %i out of %i",
                        i_halve,
                        self.max_halving,
                    )
                    do_halving = loss[active] >= prev_loss[active]
                    logger.debug("Halving to be performed on %i samples", int(np.sum(do_halving)))
                    if np.sum(do_halving) == 0:
                        break
                    active_and_do_halving = active.copy()
                    active_and_do_halving[active] = do_halving

                    lr_mult = learning_rate[active_and_do_halving]
                    for _ in range(len(x.shape) - 1):
                        lr_mult = lr_mult[:, np.newaxis]

                    adv_10 = x_adv_batch_tanh[active_and_do_halving]
                    new_x_adv_batch_tanh = adv_10 + lr_mult * perturbation_tanh[do_halving]

                    new_x_adv_batch = tanh_to_original(
                        new_x_adv_batch_tanh,
                        clip_min[active_and_do_halving],
                        clip_max[active_and_do_halving],
                    )
                    _, loss[active_and_do_halving] = self._loss(new_x_adv_batch, y_batch[active_and_do_halving])
                    logger.debug("New Average Loss: %f", np.mean(loss))
                    logger.debug("Loss: %s", str(loss))
                    logger.debug("Prev_loss: %s", str(prev_loss))
                    logger.debug("Best_loss: %s", str(best_loss))

                    best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                    best_loss[loss < best_loss] = loss[loss < best_loss]
                    learning_rate[active_and_do_halving] /= 2
                    halving[active_and_do_halving] += 1
                learning_rate[active] *= 2

                # print(self.max_doubling)
                # if no halving was actually required, double the learning rate as long as this
                # decreases the loss:
                for i_double in range(self.max_doubling):
                    logger.debug(
                        "Perform doubling iteration %i out of %i",
                        i_double,
                        self.max_doubling,
                    )
                    do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                    logger.debug(
                        "Doubling to be performed on %i samples",
                        int(np.sum(do_doubling)),
                    )
                    if np.sum(do_doubling) == 0:
                        break
                    active_and_do_doubling = active.copy()
                    active_and_do_doubling[active] = do_doubling
                    learning_rate[active_and_do_doubling] *= 2

                    lr_mult = learning_rate[active_and_do_doubling]
                    for _ in range(len(x.shape) - 1):
                        lr_mult = lr_mult[:, np.newaxis]

                    x_adv15 = x_adv_batch_tanh[active_and_do_doubling]
                    new_x_adv_batch_tanh = x_adv15 + lr_mult * perturbation_tanh[do_doubling]
                    new_x_adv_batch = tanh_to_original(
                        new_x_adv_batch_tanh,
                        clip_min[active_and_do_doubling],
                        clip_max[active_and_do_doubling],
                    )
                    _, loss[active_and_do_doubling] = self._loss(new_x_adv_batch, y_batch[active_and_do_doubling])
                    logger.debug("New Average Loss: %f", np.mean(loss))
                    best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                    best_loss[loss < best_loss] = loss[loss < best_loss]

                learning_rate[halving == 1] /= 2

                update_adv = best_lr[active] > 0
                # print(best_lr[active])
                logger.debug(
                    "Number of adversarial samples to be finally updated: %i",
                    int(np.sum(update_adv)),
                )

                print(f"Number of adversarial samples to be finally updated: {np.sum(update_adv)}")
                # exit()
                if np.sum(update_adv) > 0:
                    active_and_update_adv = active.copy()
                    active_and_update_adv[active] = update_adv
                    best_lr_mult = best_lr[active_and_update_adv]
                    for _ in range(len(x.shape) - 1):
                        best_lr_mult = best_lr_mult[:, np.newaxis]

                    best_13 = best_lr_mult * perturbation_tanh[update_adv]
                    x_adv_batch_tanh[active_and_update_adv] = x_adv_batch_tanh[active_and_update_adv] + best_13
                    x_adv_batch[active_and_update_adv] = tanh_to_original(
                        x_adv_batch_tanh[active_and_update_adv],
                        clip_min[active_and_update_adv],
                        clip_max[active_and_update_adv],
                    )
                    (z_logits[active_and_update_adv], loss[active_and_update_adv],) = self._loss(
                        x_adv_batch[active_and_update_adv],
                        y_batch[active_and_update_adv],
                    )
                    print(np.min(loss))
                    attack_success = loss <= 0
                    print(int(np.sum(attack_success)))
                    # exit()

            # Update depending on attack success:
            x_adv_batch[~attack_success] = x_batch[~attack_success]
            x_adv[batch_index_1:batch_index_2] = x_adv_batch

        logger.info(
            "Success rate of C&W L_inf attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv