"""
the y in the method generate is the one given in input or it is computed from the model, usually reformatted as the one-hot
encoding of the class classes
"""
import torch
import inspect
import logging
import numpy as np
from tqdm import tqdm
from typing import Union
from typing import Optional
from losses import losses_classifier
from art.config import ART_NUMPY_DTYPE
from art.utils import compute_success, random_sphere, compute_success_array
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import \
    ProjectedGradientDescentPyTorch

logger = logging.getLogger(__name__)


class ProjectedGradientDescent_WB(ProjectedGradientDescentPyTorch):
    def __init__(self, detectors_dict: dict, classifier_loss_name: str, **kwargs):
        #assert len(detectors_dict['dtctrs']) > 0, 'At least one detector must be passed'
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

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        # Check whether random eps is enabled
        self._random_eps()

        # Set up targets
        targets = self._set_targets(x, y)

        # Create dataset
        if mask is not None:
            # Here we need to make a distinction: if the masks are different for each input, we need to index
            # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
            if len(mask.shape) == len(x.shape):
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(mask.astype(ART_NUMPY_DTYPE)),
                )

            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])),
                )

        else:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
            )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Start to compute adversarial examples
        adv_x = x.astype(ART_NUMPY_DTYPE)

        # Compute perturbation with batching
        for (batch_id, batch_all) in enumerate(
                tqdm(data_loader, desc="PGD - Batches", ascii=True, leave=True, disable=not self.verbose)
        ):

            self._batch_id = batch_id

            if mask is not None:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
            else:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            # Compute batch_eps and batch_eps_step
            if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
                if len(self.eps.shape) == len(x.shape) and self.eps.shape[0] == x.shape[0]:
                    batch_eps = self.eps[batch_index_1:batch_index_2]
                    batch_eps_step = self.eps_step[batch_index_1:batch_index_2]

                else:
                    batch_eps = self.eps
                    batch_eps_step = self.eps_step

            else:
                batch_eps = self.eps
                batch_eps_step = self.eps_step

            for rand_init_num in range(max(1, self.num_random_init)):
                if rand_init_num == 0:
                    # first iteration: use the adversarial examples as they are the only ones we have now
                    adv_x[batch_index_1:batch_index_2] = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )
                else:
                    adversarial_batch = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )

                    # return the successful adversarial examples
                    attack_success = compute_success_array(
                        self.estimator,
                        batch,
                        batch_labels,
                        adversarial_batch,
                        self.targeted,
                        batch_size=self.batch_size,
                    )
                    adv_x[batch_index_1:batch_index_2][attack_success] = adversarial_batch[attack_success]

        logger.info(
            "Success rate of attack: %.2f%%",
            100 * compute_success(self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size),
        )
        print("Success rate of attack: {}%".format(
            100 * compute_success(self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size)))

        return adv_x

    def _compute_torch(
            self,
            x: "torch.Tensor",
            x_init: "torch.Tensor",
            y: "torch.Tensor",
            mask: "torch.Tensor",
            eps: Union[int, float, np.ndarray],
            eps_step: Union[int, float, np.ndarray],
            random_init: bool,
    ) -> "torch.Tensor":
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236).
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the
                            original input.
        :return: Adversarial examples.
        """

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()

            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            random_perturbation = torch.from_numpy(random_perturbation).to(self.estimator.device)

            if mask is not None:
                random_perturbation = random_perturbation * mask

            x_adv = x + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = torch.max(
                    torch.min(x_adv, torch.tensor(clip_max).to(self.estimator.device)),
                    torch.tensor(clip_min).to(self.estimator.device),
                )

        else:
            x_adv = x

        # Get perturbation
        perturbation = self._compute_perturbation(x_adv, y, mask, x_init)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation(x_adv, perturbation, eps_step)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm)

        # Recompute x_adv
        x_adv = perturbation + x_init

        return x_adv

    def _compute_perturbation(self, x: "torch.Tensor", y: "torch.Tensor", mask: Optional["torch.Tensor"],
                              x_init: "torch.Tensor") -> "torch.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :param x_init: Original samples corresponding to the current adversarial ones
        :return: Perturbations.
        """

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        # grad = self.estimator.loss_gradient(x=x, y=y) * (1 - 2 * int(self.targeted))

        # x is the current version of the adv samples for the current batch
        # y if passed to the generate method is the one-hot encoding of the labels (usually true) passed to generate,
        # otherwise is the one-hot of the predictions for the model

        grad = get_composite_gradient(classifier=self.estimator,
                                      classifier_loss_name=self.classifier_loss_name,
                                      detectors_list=self.detectors_list,
                                      alphas_list=self.alphas_list,
                                      loss_dtctrs_list=self.loss_dtctrs_list,
                                      x=x,
                                      x_init=x_init,
                                      y=y)

        # Write summary
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.add_scalar(
                "gradients/norm-L1/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.flatten(), ord=1),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-L2/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.flatten(), ord=2),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-Linf/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.flatten(), ord=np.inf),
                global_step=self._i_max_iter,
            )

            if hasattr(self.estimator, "compute_losses"):
                losses = self.estimator.compute_losses(x=x, y=y)

                for key, value in losses.items():
                    self.summary_writer.add_scalar(
                        "loss/{}/batch-{}".format(key, self._batch_id),
                        np.mean(value.detach().cpu().numpy()),
                        global_step=self._i_max_iter,
                    )

        # Check for nan before normalisation an replace with 0
        if torch.any(grad.isnan()):
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad[grad.isnan()] = 0.0

        # Apply mask
        if mask is not None:
            grad = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), grad)

        # Apply norm bound
        if self.norm in ["inf", np.inf]:
            grad = grad.sign()

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + tol)  # type: ignore

        assert x.shape == grad.shape

        return grad

    def _get_num_random_init(self):
        return self.num_random_init


def get_composite_gradient(classifier, classifier_loss_name, detectors_list, alphas_list, loss_dtctrs_list, x, x_init, y):
    if type(x) == np.ndarray:
        x = torch.Tensor(x).to(classifier.device)
    if type(x_init) == np.ndarray:
        x_init = torch.Tensor(x_init).to(classifier.device)
    if type(y) == np.ndarray:
        y = torch.Tensor(y).to(classifier.device)

    x = x.detach().requires_grad_()
    classifier_outs = classifier._get_last_layer_outs(x)
    classifier_outs_init = classifier._get_last_layer_outs(x_init)
    y_cold_version = torch.argmax(y, dim=1)

    loss = losses_classifier.global_loss(loss_name=classifier_loss_name,
                                         preds=classifier_outs, y=y_cold_version, nat=classifier_outs_init)
    for i in range(len(detectors_list)):
        detector = detectors_list[i]
        alpha = alphas_list[i] if alphas_list[i] is not None else 1.
        detector_loss = loss_dtctrs_list[i] if loss_dtctrs_list[
                                                   i] is not None else losses_classifier._get_loss_by_name('BCE')

        dtctr_outs = detector._get_last_layer_outs(classifier_outs)
        # print(dtctr_outs)
        #dtctr_target = torch.zeros(y_cold_version.shape).reshape(-1, 1).to(detector.device)
        dtctr_target = create_labels_detector(logits_class=classifier_outs,
                                               y_class=y,
                                               device=detector.device)
        loss += alpha * detector_loss(dtctr_outs, dtctr_target)
        # print(loss)

    grad = torch.autograd.grad(loss, [x], allow_unused=True)[0]
    return grad


def create_labels_detector(logits_class, y_class, device):
    y_pred_class = np.argmax(torch.softmax(logits_class, dim=1).detach().cpu().numpy(), axis=1)
    if not isinstance(y_class, np.ndarray):
        y_class = y_class.detach().cpu().numpy()
    y_class = np.argmax(y_class, axis=1)
    # print(y_class.shape)
    y_det_1 = np.where(y_pred_class == y_class, 0, 1)
   # print(y_det_1)
    # # print((y_det_1))
    #
    # filter_ = y_pred_class != y_class
    # y_det_2 = np.zeros((y_pred_class.shape[0],))
    # y_det_2[filter_] = 1
    # # print(y_det_2)
    #
    # # print((y_det_1 - y_det_2).sum())
    # # exit()
    # # np.where(y_pred_class == y_class, 0, 1)

    return torch.Tensor(y_det_1).reshape(-1, 1).to(device)
