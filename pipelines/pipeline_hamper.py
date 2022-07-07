import os
import torch
import numpy as np
import pickle
from copy import deepcopy
from utils import utils_ml
from utils import utils_general
import torch.backends.cudnn as cudnn

from losses.losses_classifier import _get_loss_by_name
from art_wb.estimators.classification.pytorch.art_interface import CustomPyTorchClassifier
from art_wb.estimators.regression.scikitlearn.art_interface import CustomScikitlearnRegressor
from art_wb.attacks.evasion.sa_wb_hamper import SquareAttack_WB_hamper

from datasets.dataset import get_dataloader
from utils.utils_models import load_classifier, extraction_resnet


def execute_attack(attack_strategy, common_parameters, threshold, y_train, K, num_classes, layers, dict_train):
    assert attack_strategy == 'sa', 'Attack not implemented'

    attack = SquareAttack_WB_hamper(detectors_dict=common_parameters['detectors_dict'],
                                    classifier_loss_name=common_parameters['classifier_loss_name'],
                                    threshold=threshold,
                                    y_train=y_train,
                                    K=K,
                                    num_classes=num_classes,
                                    layers=layers,
                                    dict_train=dict_train,
                                    estimator=common_parameters['estimator'],
                                    max_iter=1,
                                    norm=np.inf,
                                    batch_size=common_parameters['batch_size']
                                    )
    attack_name = "_sa"

    return attack, attack_name



def main_pipeline_wb(args, alpha=None):
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
        break

    # classifier_input_shape = next(classifier_model.parameters()).size()
    print(classifier_input_shape)

    # transform the classifier in an art kind of network using the interface
    classifier = CustomPyTorchClassifier(
        model=classifier_model,
        loss_train=_get_loss_by_name(loss_name='CE'),
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
    model_path = '{}ridge_2.pt'.format(args.DETECTOR.detector_dir)
    detector_model = pickle.load(open(model_path, 'rb'))

    args.DETECTOR.loss_adv = 'BCE' if args.DETECTOR.loss_adv is None else args.DETECTOR.loss_adv

    # transform the classifier in an art kind of network using the interface
    detector = CustomScikitlearnRegressor(
        model=detector_model,
        clip_values=[0,1],
        device_type=device
    )

    # adapt the interface to be parallelized
    detector._to_data_parallel()

    detectors_dict = {'dtctrs': [detector], 'alphas': args.ADV_CREATION.alpha if alpha is None else [alpha], 'loss_dtctrs': [None]}

    # --------------------------------- #
    # ---- Perform and save attack ---- #
    # --------------------------------- #

    features_ndarray = deepcopy(utils_general.FeaturesDataLoaderToNdarray(loader=data_loader))
    print('features_ndarray.shape:{}'.format(features_ndarray.shape))

    if args.DATA_NATURAL.data_name in ['cifar10', 'svhn']:
        layers = ['block-{}-{}-{}'.format(b_layer, n_layer, n_block)
                  for b_layer in ['bn_1', 'bn_2', 'conv_1', 'conv_2']
                  for n_layer in ['layer4']
                  for n_block in [0, 1]]
        layers += ['layer4', 'convolution_end', 'logits']
        num_classes = 10

    data_loader_train = get_dataloader(data_name=args.DATA_NATURAL.data_name, train=True, batch_size=args.RUN.batch_size)
    y_train = get_dataloader(data_name=args.DATA_NATURAL.data_name, train=True, batch_size=args.RUN.batch_size, return_numpy=True)[1]
    dict_train = extraction_resnet(data_loader_train, classifier, bs=args.RUN.batch_size)

    attack_strategy = args.ADV_CREATION.strategy
    parameters_common_attacks = {'detectors_dict': detectors_dict,
                                 'classifier_loss_name': args.CLASSIFIER.loss,
                                 'estimator': classifier,
                                 'batch_size': args.RUN.batch_size,
                                 'verbose' : True}
    attack, attack_name = execute_attack(attack_strategy=attack_strategy,
                                         common_parameters=parameters_common_attacks,
                                         layers=layers,
                                         threshold=args.DEPTH.threshold,
                                         K=args.DEPTH.K,
                                         dict_train=dict_train,
                                         y_train=y_train,
                                         num_classes=num_classes)
    os.makedirs('{}/{}/white-box-hamper/'.format(args.ADV_CREATION.adv_file_path, args.DATA_NATURAL.data_name), exist_ok=True)
    adv_file_path = '{}/{}/white-box-hamper/{}{}_alpha_{}.npy'.format(args.ADV_CREATION.adv_file_path,
                                                            args.DATA_NATURAL.data_name,
                                                            args.DATA_NATURAL.data_name,
                                                            attack_name,
                                                            args.ADV_CREATION.alpha[0] if alpha is None else alpha
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
    from art_wb.attacks.evasion.sa_wb_hamper import compute_hamper_features
    # Compute accuracy of the classifier on the attack
    data_loader_adv_classifier = from_numpy_to_dataloader(adv_x, labels, batch_size=args.RUN.batch_size)
    logits_class, labels_class, predictions_class = utils_ml.compute_logits_return_labels_and_predictions(model=classifier,
                                                                                                          dataloader=data_loader_adv_classifier,
                                                                                                          device=device)
    labels_det = np.where(labels_class == predictions_class, 0, 1)

    # Compute accuracy of the detector on the attack
    hamper_features = compute_hamper_features(adv_x, classifier, dict_train, y_train, args.DEPTH.K, layers, num_classes, args.RUN.batch_size)
    data_loader_adv_detector = from_numpy_to_dataloader(hamper_features, labels_det, batch_size=args.RUN.batch_size)
    _, labels_det, predictions_det = utils_ml.compute_logits_return_labels_and_predictions(model=detector,
                                                                                           dataloader=data_loader_adv_detector,
                                                                                           device=device)

    print('Classifier', utils_ml.compute_accuracy(predictions=predictions_class, targets=labels_class))
    print(predictions_class)
    print(labels_class)

    print('Detector', utils_ml.compute_accuracy(predictions=1 - predictions_det, targets=labels_det))
    print(predictions_det)
    print(labels_det)


