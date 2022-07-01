import os
import torch
import losses
import numpy as np
from copy import deepcopy
from utils import utils_ml
from utils import utils_general
import torch.backends.cudnn as cudnn

from art_wb.estimators.classification.pytorch import art_interface
from art_wb.attacks.evasion.pgd_wb import ProjectedGradientDescent_WB
from art_wb.attacks.evasion.fgsm_wb import FastGradientSignMethod_WB
from art_wb.attacks.evasion.bim_wb import BasicIterativeMethod_WB
from art_wb.attacks.evasion.sa_wb_network import SquareAttack_WB_network
from art_wb.attacks.evasion.cw_wb import CarliniLInfMethod_WB

from datasets.dataset import get_dataloader
from utils.utils_models import load_classifier, load_detectors


def execute_attack(attack_strategy, eps, norm, common_parameters):
    assert (norm in [1, 2, 'inf', np.inf] and attack_strategy == 'pgd') or (attack_strategy in ['fgsm', 'bim', 'cwi', 'sa']), 'Attack not implemented'

    if attack_strategy == 'pgd':
        if norm == 1:
            eps_step = 4
        elif norm == 2:
            eps_step = .1
        else:
            eps_step = .01
        attack = ProjectedGradientDescent_WB(detectors_dict=common_parameters['detectors_dict'],
                                             classifier_loss_name=common_parameters['classifier_loss_name'],
                                             estimator=common_parameters['estimator'],
                                             norm=norm,
                                             eps=eps,
                                             eps_step=eps_step,
                                             max_iter=10,
                                             batch_size=common_parameters['batch_size']
                                             )
        attack_name = common_parameters['classifier_loss_name'] + "_pgd" + (str(norm) if norm in [1, 2] else "i") + "_" + str(eps)

    elif attack_strategy == 'fgsm':
        attack = FastGradientSignMethod_WB(detectors_dict=common_parameters['detectors_dict'],
                                           classifier_loss_name=common_parameters['classifier_loss_name'],
                                           estimator=common_parameters['estimator'],
                                           norm=np.inf,
                                           eps=eps,
                                           eps_step=.01,
                                           batch_size=common_parameters['batch_size']
                                           )
        attack_name = common_parameters['classifier_loss_name'] + "_fgsm_" + str(eps)

    elif attack_strategy == 'bim':
        attack = BasicIterativeMethod_WB(detectors_dict=common_parameters['detectors_dict'],
                                         classifier_loss_name=common_parameters['classifier_loss_name'],
                                         estimator=common_parameters['estimator'],
                                         norm=np.inf,
                                         eps=eps,
                                         eps_step=.01,
                                         max_iter=int(eps * 256 * 1.25),
                                         batch_size=common_parameters['batch_size']
                                         )
        attack_name = common_parameters['classifier_loss_name'] + "_bim_" + str(eps)

    elif attack_strategy == 'sa':
        attack = SquareAttack_WB_network(detectors_dict=common_parameters['detectors_dict'],
                                         classifier_loss_name=common_parameters['classifier_loss_name'],
                                         estimator=common_parameters['estimator'],
                                         max_iter=200,
                                         norm=np.inf,
                                         batch_size=common_parameters['batch_size']
                                         )
        attack_name = "_sa"

    elif attack_strategy == 'cwi':
        attack = CarliniLInfMethod_WB(detectors_dict=common_parameters['detectors_dict'],
                                      classifier_loss_name=common_parameters['classifier_loss_name'],
                                      classifier=common_parameters['estimator'],
                                      confidence=0.0,
                                      targeted=False,
                                      learning_rate=.01,
                                      max_iter=100, #200
                                      batch_size=common_parameters['batch_size']
                                      )
        attack_name = "_cwi"

    return attack, attack_name



def main_pipeline_wb(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    if args.RUN.seed is not None:
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.RUN.seed)

    # ---------------------- #
    # ---- Load dataset ---- #
    # ---------------------- #

    data_loader = get_dataloader(data_name=args.DATA_NATURAL.data_name, train=False, batch_size=args.RUN.batch_size)

    # ------------------------------- #
    # ---- Load model classifier ---- #
    # ------------------------------- #

    classifier_model = load_classifier(checkpoint_dir=args.CLASSIFIER.classifier_dir, dataset=args.DATA_NATURAL.data_name,
                                       device=device).eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        classifier_input_shape = data.shape[1:]
        detector_input_shape = classifier_model(data).shape[1:]
        break

    # classifier_input_shape = next(classifier_model.parameters()).size()
    print(classifier_input_shape)

    # transform the classifier in an art kind of network using the interface
    classifier = art_interface.CustomPyTorchClassifier(
        model=classifier_model,
        loss=losses.losses_classifier._get_loss_by_name(loss_name=args.CLASSIFIER.loss),
        input_shape=classifier_input_shape,
        nb_classes=args.DATA_NATURAL.num_classes,
        clip_values=[0, 1]
    )

    # adapt the interface to be parallelized
    classifier._to_data_parallel()

    # extract logits, labels and predictions and run some checks
    logits, labels, predictions = utils_ml.compute_logits_return_labels_and_predictions(model=classifier,
                                                                                        dataloader=data_loader,
                                                                                        device=device)
    print(utils_ml.compute_accuracy(predictions=predictions, targets=labels))

    # ------------------------------ #
    # ---- Load model detectors ---- #
    # ------------------------------ #

    # let us consider the case in which this function always only returns a single detector as it has been the case so far
    detectors, _ = load_detectors(args=args,
                                  model=classifier_model,
                                  device=device,
                                  epsilon=args.DETECTOR.epsilon,
                                  loss=args.CLASSIFIER.loss)

    assert len(detectors) == 1, 'load_detectors returns a list of detectors but we want the list to have length 1'

    detector_model = detectors[0]

    # detector_input_shape = next(detector_model.parameters()).size()
    print(detector_input_shape)

    args.DETECTOR.loss_adv = 'BCE' if args.DETECTOR.loss_adv is None else args.DETECTOR.loss_adv

    # transform the classifier in an art kind of network using the interface
    detector = art_interface.CustomPyTorchClassifier(
        model=detector_model,
        loss=losses.losses_classifier._get_loss_by_name(loss_name=args.DETECTOR.loss_adv),
        input_shape=detector_input_shape,
        nb_classes=2,
    )

    # adapt the interface to be parallelized
    detector._to_data_parallel()

    detectors_dict = {'dtctrs': [detector], 'alphas': args.ADV_CREATION.alpha, 'loss_dtctrs': [None]}

    # --------------------------------- #
    # ---- Perform and save attack ---- #
    # --------------------------------- #

    features_ndarray = deepcopy(utils_general.FeaturesDataLoaderToNdarray(loader=data_loader))
    print('features_ndarray.shape:{}'.format(features_ndarray.shape))

    attack_strategy = args.ADV_CREATION.strategy
    parameters_common_attacks = {'detectors_dict': detectors_dict,
                                 'classifier_loss_name': args.CLASSIFIER.loss,
                                 'estimator': classifier,
                                 'batch_size': args.RUN.batch_size,
                                 'verbose' : True}

    attack, attack_name = execute_attack(attack_strategy=attack_strategy, eps=args.ADV_CREATION.epsilon, norm=args.ADV_CREATION.norm, common_parameters=parameters_common_attacks)
    os.makedirs('{}/{}/white-box-salad/'.format(args.ADV_CREATION.adv_file_path, args.DATA_NATURAL.data_name), exist_ok=True)
    adv_file_path = '{}/{}/white-box-salad/{}{}.npy'.format(args.ADV_CREATION.adv_file_path,
                                                            args.DATA_NATURAL.data_name,
                                                            args.DATA_NATURAL.data_name,
                                                            attack_name
                                                            )
    print(adv_file_path)

    # the y in the method generate is the one given in input or it is computed from the model, usually reformatted as the
    # one-hot encoding of the class classes the x in the method generate is updated to create the adversarial attacks, pass
    # a deepcopy of the initial samples
    adv_x = attack.generate(x=features_ndarray, y=labels.detach().cpu().numpy())
    np.save(adv_file_path, adv_x)

    # ------------------------- #
    # ---- Evaluate attack ---- #
    # ------------------------- #

    from utils.utils_general import from_numpy_to_dataloader
    # Compute accuracy of the classifier on the attack
    data_loader_adv_classifier = from_numpy_to_dataloader(adv_x, labels, batch_size=args.RUN.batch_size)
    logits_class, labels_class, predictions_class = utils_ml.compute_logits_return_labels_and_predictions(model=classifier,
                                                                                                          dataloader=data_loader_adv_classifier,
                                                                                                          device=device)
    labels_det = np.where(labels_class == predictions_class, 0, 1)

    # Compute accuracy of the detector on the attack
    data_loader_adv_detector = from_numpy_to_dataloader(logits_class, labels_det, batch_size=args.RUN.batch_size)
    _, labels_det, predictions_det = utils_ml.compute_logits_return_labels_and_predictions(model=detector,
                                                                                           dataloader=data_loader_adv_detector,
                                                                                           device=device)

    print('Classifier', utils_ml.compute_accuracy(predictions=predictions_class, targets=labels_class))
    print(predictions_class)
    print(labels_class)

    print('Detector', utils_ml.compute_accuracy(predictions=predictions_det, targets=labels_det))
    print(predictions_det)
    print(labels_det)


