
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as t
from torchvision.datasets import CIFAR100, CIFAR10
from data.sampler import SubsetSequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def get_transform(dataset):
    if dataset == 'cifar10':
        train_transform = t.Compose([
            t.RandomHorizontalFlip(),
            t.RandomCrop(size=32, padding=4),
            t.ToTensor(),
            t.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        test_transform = t.Compose([
            t.ToTensor(),
            t.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        return train_transform, test_transform
    if dataset == 'cifar100':
        train_transform = t.Compose([
            t.RandomHorizontalFlip(),
            t.RandomCrop(size=32, padding=4),
            t.ToTensor(),
            t.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        test_transform = t.Compose([
            t.ToTensor(),
            t.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        return train_transform, test_transform
    else:
        print("Error: No dataset named {}!".format(dataset))
        return -1


def get_dataset(dataset_root, dataset, train_transform, test_transform):
    if dataset == 'cifar10':
        train = CIFAR10(dataset_root, train=True, download=True, transform=train_transform)
        unlabeled = CIFAR10(dataset_root, train=True, download=True, transform=train_transform)
        test = CIFAR10(dataset_root, train=False, download=True, transform=test_transform)
    elif dataset == 'cifar100':
        train = CIFAR100(dataset_root, train=True, download=True, transform=train_transform)
        unlabeled = CIFAR100(dataset_root, train=True, download=True, transform=train_transform)
        test = CIFAR100(dataset_root, train=False, download=True, transform=test_transform)
    else:
        print("Error: No dataset named {}!".format(dataset))
        return -1
    return train, test, unlabeled


def get_training_functions(cfg, models):
    criterion = nn.CrossEntropyLoss(reduction='none')
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=cfg.TRAIN.LR,
                               momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
    optim_module = optim.SGD(models['module'].parameters(), lr=cfg.TRAIN.LR,
                             momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=cfg.TRAIN.MILESTONES)
    sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=cfg.TRAIN.MILESTONES)

    optimizers = {'backbone': optim_backbone, 'module': optim_module}
    schedulers = {'backbone': sched_backbone, 'module': sched_module}
    return criterion, optimizers, schedulers


def get_training_functions_single(cfg, models):
    #criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = nn.BCELoss(reduction='none')
    optim_backbone = optim.SGD(models['backbone'].parameters(), lr=cfg.TRAIN.LR,
                               momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
    sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=cfg.TRAIN.MILESTONES)
    # optim_task_model = torch.optim.SGD(task_model.parameters(), lr=cfg.TRAIN.LR,
    #                                    momentum=cfg.TRAIN.MOMENTUM,
    #                                    weight_decay=cfg.TRAIN.WDECAY)

    optimizers = {'backbone': optim_backbone}
    schedulers = {'backbone': sched_backbone}
    return criterion, optimizers, schedulers

def get_training_functions_isdal(cfg, models):
    optim_model = optim.SGD(models.parameters(), lr=cfg.TRAIN.LR,
                               momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
    sched_model = lr_scheduler.MultiStepLR(optim_model, milestones=cfg.TRAIN.MILESTONES)
    # optim_task_model = torch.optim.SGD(task_model.parameters(), lr=cfg.TRAIN.LR,
    #                                    momentum=cfg.TRAIN.MOMENTUM,
    #                                    weight_decay=cfg.TRAIN.WDECAY)

    optimizers = {'model': optim_model}
    schedulers = {'model': sched_model}
    return  optimizers, schedulers


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()


def update_dataloaders(
        cfg,
        unlabeled_set, labeled_set,
        unlabeled_dataset, train_dataset,
        models, dataloaders
):
    # Randomly sample 10000 unlabeled data points
    random.shuffle(unlabeled_set)
    subset = unlabeled_set[:cfg.ACTIVE_LEARNING.SUBSET]

    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.TRAIN.BATCH,
                                  sampler=SubsetSequentialSampler(subset),
                                  pin_memory=True)

    # Measure uncertainty of each data points in the subset
    uncertainty = get_uncertainty(models, unlabeled_loader)

    # Index in ascending order
    arg = np.argsort(uncertainty)

    # Update the labeled dataset and the unlabeled dataset, respectively
    budget = cfg.ACTIVE_LEARNING.ADDENDUM
    labeled_set += list(torch.tensor(subset)[arg][-budget:].numpy())
    unlabeled_set = list(torch.tensor(subset)[arg][:-budget].numpy()) + unlabeled_set[cfg.ACTIVE_LEARNING.SUBSET:]

    # Create a new dataloader for the updated labeled dataset
    dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH,
                                      sampler=SubsetRandomSampler(labeled_set),
                                      pin_memory=True)

    return dataloaders, unlabeled_loader, unlabeled_set


# def get_my_uncertainty(models, unlabeled_loader):
#     models['backbone'].eval()
#     uncertainty = torch.tensor([]).cuda()
#
#     with torch.no_grad():
#         for (inputs, labels) in unlabeled_loader:
#             inputs = inputs.cuda()
#             # labels = labels.cuda()
#
#             scores, features = models['backbone'](inputs)
#
#             uncertainty = torch.cat((uncertainty, scores.max(1)[0]), 0)
#
#     return uncertainty.cpu()


def get_my_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    uncertainty = torch.tensor([]).cuda()
    scores = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda().long()
    feats = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, label) in unlabeled_loader:
            inputs = inputs.cuda()
            label = label.cuda()

            score, features = models['backbone'](inputs)

            uncertainty = torch.cat((uncertainty, score.max(1)[0]), 0)
            scores = torch.cat((scores, score), 0)
            labels = torch.cat((labels, label), 0)

    return uncertainty.cpu(), scores.cpu(), labels.cpu(), feats.cpu()


def update_dataloaders_single(
        cfg,
        unlabeled_set, labeled_set,
        unlabeled_dataset, train_dataset,
        dataloaders
):
    # Randomly sample 10000 unlabeled data points
    random.shuffle(unlabeled_set)
    subset = unlabeled_set[:cfg.ACTIVE_LEARNING.SUBSET]

    # Create unlabeled dataloader for the unlabeled subset
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.TRAIN.BATCH,
                                  sampler=SubsetSequentialSampler(subset),
                                  pin_memory=True)

    # random select
    arg = np.argsort(np.random.rand(cfg.ACTIVE_LEARNING.SUBSET))

    # Update the labeled dataset and the unlabeled dataset, respectively
    budget = cfg.ACTIVE_LEARNING.ADDENDUM
    labeled_set += list(torch.tensor(subset)[arg][-budget:].numpy())
    unlabeled_set = list(torch.tensor(subset)[arg][:-budget].numpy()) + unlabeled_set[cfg.ACTIVE_LEARNING.SUBSET:]

    # Create a new dataloader for the updated labeled dataset
    dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH,
                                      sampler=SubsetRandomSampler(labeled_set),
                                      pin_memory=True)

    return dataloaders, unlabeled_loader, unlabeled_set
