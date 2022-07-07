import torch
import numpy as np
import math
import torch.nn.functional as F
from models.detector import Detector
import torch.optim as optim
from tqdm import tqdm
from art_wb.estimators.classification.pytorch.art_interface import CustomPyTorchClassifier, PyTorchClassifier

def create_detectors(args, model, device, **kwargs):
    detectors = []
    optimizers = []

    for layer_size in model.intermediate_size():
        use_cuda = torch.cuda.is_available()

        detector = Detector(input_shape=layer_size, drop=0, nodes=args.DETECTOR.DETECTOR_MODEL.nodes,
                            layers=args.DETECTOR.DETECTOR_MODEL.layers)

        if use_cuda:
            detectors.append(detector.to(device))
        else:
            detectors.append(detector)
        optimizers.append(optim.SGD(detector.parameters(), lr=args.DETECTOR.lr,
                                    momentum=args.DETECTOR.momentum,
                                    weight_decay=args.DETECTOR.weight_decay,
                                    nesterov=args.DETECTOR.nesterov))
    return detectors, optimizers


def load_classifier(dataset, checkpoint_dir, device):
    return load_model(dataset_name=dataset, checkpoints_dir=checkpoint_dir, device=device)


def load_model(dataset_name, checkpoints_dir, device):
    if dataset_name == 'cifar10':
        from models.resnet import ResNet18
        path = '{}{}/rn-best.pt'.format(checkpoints_dir, dataset_name)
        model = ResNet18(num_classes=10)
    elif dataset_name == 'svhn':
        from models.resnet import ResNet18
        path = '{}{}/rn-best.pt'.format(checkpoints_dir, dataset_name)
        model = ResNet18(num_classes=10)
    else:
        exit(dataset_name + " not present")

    if torch.cuda.is_available():
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model


def load_detectors(args, model, device, epsilon=None, loss: str = None):
    loss_train = loss

    detectors = [Detector(input_shape=layer_size, drop=0, nodes=args.DETECTOR.DETECTOR_MODEL.nodes,
                          layers=args.DETECTOR.DETECTOR_MODEL.layers) for
                 layer_size in model.intermediate_size()]
    optimizers = [optim.SGD(detectors[i].parameters(), lr=args.DETECTOR.lr,
                            momentum=args.DETECTOR.momentum,
                            weight_decay=args.DETECTOR.weight_decay,
                            nesterov=args.DETECTOR.nesterov) for i in range(len(detectors))]

    use_cuda = torch.cuda.is_available()

    epoch = args.DETECTOR.resume_epoch

    for i in range(len(detectors)):
        if use_cuda:
            detectors[i] = detectors[i].to(device)

        p = '{}/{}/{}_-1_{}/detector_epoch_{}.pt'.format(args.DETECTOR.detector_dir, args.DATA_NATURAL.data_name,
                                                         loss_train,
                                                         args.TRAIN.PGDi.epsilon if epsilon is None else epsilon, epoch)
        print(p)
        checkpoint = torch.load(p, map_location=device)

        if 'optimizer_state_dict' in checkpoint:
            detectors[i].load_state_dict(checkpoint['model_state_dict'])
            optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

        else:
            detectors[i].load_state_dict(checkpoint)

    return detectors, optimizers



def extraction_resnet(loader, model, device="cuda", bs=100):
    device = model.device
    if type(model) in [CustomPyTorchClassifier, PyTorchClassifier]:
        model = model.model
    if isinstance(loader, np.ndarray):
        from utils.utils_general import from_numpy_to_dataloader
        loader = from_numpy_to_dataloader(loader, np.ones(loader.shape[0]), bs)

    all_hidden_states = {}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader, ascii=True, ncols=70)):
            data = data.to(device)
            hidden_states = data
            if batch_idx == 0:
                # Conv1
                hidden_states = model.conv1(hidden_states)

                # Bn1
                hidden_states = model.bn1(hidden_states)
                hidden_states = F.relu(hidden_states)

                # Layer 1
                hidden_states = model.layer1(hidden_states)
                all_hidden_states['layer1'] = hidden_states.detach().cpu().numpy()

                # Layer2
                # Block0
                # Conv1
                hidden_states = model.layer2(hidden_states)
                all_hidden_states['layer2'] = hidden_states.detach().cpu().numpy()

                # Layer3
                # Block0
                # Conv1
                hidden_states = model.layer3(hidden_states)
                all_hidden_states['layer3'] = hidden_states.detach().cpu().numpy()

                # Layer4
                # Block0
                # Conv1
                inside_hidden_states = model.layer4[0].conv1(hidden_states)
                all_hidden_states['block-conv_1-layer4-0'] = inside_hidden_states.detach().cpu().numpy()
                # Bn1
                inside_hidden_states = F.relu(model.layer4[0].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer4-0'] = inside_hidden_states.detach().cpu().numpy()
                # Conv2
                inside_hidden_states = model.layer4[0].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer4-0'] = inside_hidden_states.detach().cpu().numpy()
                # Bn2
                inside_hidden_states = model.layer4[0].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer4-0'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                out_block0_hidden_states = model.layer4[0](hidden_states)

                # Block1
                # Conv1
                inside_hidden_states = model.layer4[1].conv1(out_block0_hidden_states)
                all_hidden_states['block-conv_1-layer4-1'] = inside_hidden_states.detach().cpu().numpy()
                # Bn1
                inside_hidden_states = F.relu(model.layer4[1].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer4-1'] = inside_hidden_states.detach().cpu().numpy()
                # Conv2
                inside_hidden_states = model.layer4[1].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer4-1'] = inside_hidden_states.detach().cpu().numpy()
                # Bn2
                inside_hidden_states = model.layer4[1].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer4-1'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                out_block1_hidden_states = model.layer4[1](out_block0_hidden_states)

                hidden_states = model.layer4(hidden_states)
                all_hidden_states['layer4'] = hidden_states.detach().cpu().numpy()

                hidden_states = F.avg_pool2d(hidden_states, 4).view(data.shape[0], -1)
                all_hidden_states['convolution_end'] = hidden_states.detach().cpu().numpy()

                logits = model.linear(hidden_states)
                all_hidden_states['logits'] = logits.detach().cpu().numpy()
                pred = torch.nn.Softmax(dim=1)(logits)
                all_hidden_states['pred'] = pred.detach().cpu().numpy()
            else:
                # Conv1
                hidden_states = model.conv1(hidden_states)
                # Bn1
                hidden_states = model.bn1(hidden_states)
                # all_hidden_states['bn1'] = np.concatenate(( all_hidden_states['bn1'], hidden_states.detach().cpu().numpy()), axis=0)
                hidden_states = F.relu(hidden_states)

                # Layer1
                # Block0
                # Conv1
                hidden_states = model.layer1(hidden_states)
                all_hidden_states['layer1'] = np.concatenate((all_hidden_states['layer1'], hidden_states.detach().cpu().numpy()), axis=0)

                # Layer2
                # Block0
                # Conv1
                hidden_states = model.layer2(hidden_states)
                all_hidden_states['layer2'] = np.concatenate((all_hidden_states['layer2'], hidden_states.detach().cpu().numpy()), axis=0)

                # Layer3
                # Block0
                # Conv1
                hidden_states = model.layer3(hidden_states)
                all_hidden_states['layer3'] = np.concatenate((all_hidden_states['layer3'], hidden_states.detach().cpu().numpy()), axis=0)

                # Layer4
                # Block0
                # Conv1
                inside_hidden_states = model.layer4[0].conv1(hidden_states)
                all_hidden_states['block-conv_1-layer4-0'] = np.concatenate((all_hidden_states['block-conv_1-layer4-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # Bn1
                inside_hidden_states = F.relu(model.layer4[0].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer4-0'] = np.concatenate((all_hidden_states['block-bn_1-layer4-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # Conv2
                inside_hidden_states = model.layer4[0].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer4-0'] = np.concatenate((all_hidden_states['block-conv_2-layer4-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # Bn2
                inside_hidden_states = model.layer4[0].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer4-0'] = np.concatenate((all_hidden_states['block-bn_2-layer4-0'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                out_block0_hidden_states = model.layer4[0](hidden_states)

                # Block1
                # Conv1
                inside_hidden_states = model.layer4[1].conv1(out_block0_hidden_states)
                all_hidden_states['block-conv_1-layer4-1'] = np.concatenate((all_hidden_states['block-conv_1-layer4-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # Bn1
                inside_hidden_states = F.relu(model.layer4[1].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer4-1'] = np.concatenate((all_hidden_states['block-bn_1-layer4-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # Conv2
                inside_hidden_states = model.layer4[1].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer4-1'] = np.concatenate((all_hidden_states['block-conv_2-layer4-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # Bn2
                inside_hidden_states = model.layer4[1].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer4-1'] = np.concatenate((all_hidden_states['block-bn_2-layer4-1'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                out_block1_hidden_states = model.layer4[1](out_block0_hidden_states)

                hidden_states = model.layer4(hidden_states)
                all_hidden_states['layer4'] = np.concatenate((all_hidden_states['layer4'], hidden_states.detach().cpu().numpy()), axis=0)

                hidden_states = F.avg_pool2d(hidden_states, 4).view(data.shape[0], -1)
                all_hidden_states['convolution_end'] = np.concatenate((all_hidden_states['convolution_end'], hidden_states.detach().cpu().numpy()), axis=0)

                logits = model.linear(hidden_states)
                all_hidden_states['logits'] = np.concatenate((all_hidden_states['logits'], logits.detach().cpu().numpy()), axis=0)
                pred = torch.nn.Softmax(dim=1)(logits)
                all_hidden_states['pred'] = np.concatenate((all_hidden_states['pred'], pred.detach().cpu().numpy()), axis=0)

    return all_hidden_states
