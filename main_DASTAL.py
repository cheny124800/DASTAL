import argparse
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data.sampler import SubsetSequentialSampler
# Custom
import models.isresnet
import models.vgg16  
from utils.config import get_configs
from utils.dataset import get_dataset, get_transform, get_training_functions_isdal
from utils.train_IDA import *
from utils.test_IDA import *
from utils.util import *
from torch.backends import cudnn
import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from ISAL import ISDALoss
# from torch.nn import functional
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--lambda_0', default=1.0, type=float,
                    help='The hyper-parameter \lambda_0 for ISDA, select from {1, 2.5, 5, 7.5, 10}. '
                         'We adopt 1 for DenseNets and 7.5 for ResNets and ResNeXts, except for using 5 for ResNet-101.')
parser.add_argument('--model', default='Vgg16', type=str,
                    help='Model to be trained.  ResNet18 Vgg16'
                         'Select from resnet{18, 34, 50, 101, 152} / resnext{50_32x4d, 101_32x8d} / '
                         'densenet{121, 169, 201, 265}')
parser.add_argument('--metric', default='ISDAL', type=str,
                    help='Ent, Maxp, ISDAL, Ran')


#Ent, Maxp, ISDAL, Ran

def get_ems(model, unlabeled_loader,criterion_isda,metric="Maxp"): #Ent, Maxp, ISDAL, Ran
    model.eval()    
    ems = torch.tensor([]).cuda()    
    with torch.no_grad():
        for (inputs, _) in unlabeled_loader:
            inputs = inputs.cuda()
            if metric == "Maxp":
                output1= model(inputs)
                pro = F.softmax(output1,dim=1)
                ems_cur = torch.max(pro, 1).values
                #while(1):True
            if metric == "Ent":
                output1= model(inputs)                
                pro = F.softmax(output1,dim=1)
                log_pro = -torch.log(pro)
                ent = pro * log_pro
                ems_cur = ent.sum(dim=-1) 
            if metric == "Ran":
                ems_cur =  torch.rand(len(inputs), out=None).cuda() 
            if metric == "ISDAL":
                target_x = torch.tensor(len(inputs)).cuda()
                ems_cur = criterion_isda(model, inputs, target_x, 1,Val_Flag = True)     
                #ems_cur = output1.max()                                             
            ems = torch.cat((ems, ems_cur), 0)
    return ems.cpu()

# Main
if __name__ == '__main__':
    args = parser.parse_args()
    # hyper params
    port = 9999
    dataset = 'cifar10'
    method = args.metric
    num_classes = 10
    cfg = get_configs(port=port)

    # path and logger
    time_str = time.strftime('%m%d%H%M%S', time.localtime())
    dataset_root = cfg.DATASET.ROOT[dataset]
    checkpoint_dir = os.path.join('output', time_str)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    log_save_file = checkpoint_dir + "/log.txt"
    my_logger = Logger(method, log_save_file).get_log
    my_logger.info("Checkpoint Path: {}".format(checkpoint_dir))
    my_logger.info("Dataset : {}".format(dataset))    
    my_logger.info("Model : {}".format(args.model))    
    my_logger.info("ISDA lambda_0 : {}".format(args.lambda_0))         
    print_config(cfg, my_logger)

    # define dataset and dataloaders
    train_transform, test_transform = get_transform(dataset)
    train_dataset, test_dataset, unlabeled_dataset = get_dataset(
        dataset_root, dataset, train_transform, test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH)

    Performance = np.zeros((10, 10))
    for trial in range(cfg.ACTIVE_LEARNING.TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points
        # from the entire dataset.
        indices = list(range(cfg.DATASET.NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:cfg.ACTIVE_LEARNING.ADDENDUM]
        unlabeled_set = indices[cfg.ACTIVE_LEARNING.ADDENDUM:]

        random.shuffle(unlabeled_set)

        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        if 'Res' in args.model:
            model = eval('models.isresnet.' + args.model)(num_classes=num_classes)

        if 'Vgg' in args.model:
            model = eval('models.vgg16.' + args.model)(num_classes=num_classes)

        #resnet18 = resnet.resnet18(num_classes=num_classes).cuda()
        feature_num = model.feature_num

        torch.backends.cudnn.benchmark = True
        num_images = 50000
        initial_budget = 1000
        all_indices = set(np.arange(num_images))
        initial_indices = random.sample(all_indices, initial_budget)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(initial_indices)
        model = model.cuda()
        criterion_isda = ISDALoss(feature_num, num_classes).cuda()
        criterion_ce = nn.CrossEntropyLoss().cuda()

        # Active learning cycles

        acc =0
        for cycle in range(cfg.ACTIVE_LEARNING.CYCLES):
            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:cfg.ACTIVE_LEARNING.SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.TRAIN.BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)

            optimizers, schedulers = get_training_functions_isdal(
                cfg, model
            )

            # Training and test
            if 1:
                cur_acc = train(
                    cfg, my_logger, model, criterion_isda, optimizers,
                    schedulers, dataloaders, unlabeled_loader,
                    cfg.TRAIN.EPOCH,args, cycle
                    # querry_dataloader, task_model, optim_task_model, val_loader
                )
            torch.save(
                        {
                            'cycle': cycle,
                            'state_dict_backbone': model.state_dict()
                        },
                        '{}/isdal active_resnet18_cifar10_cycle_9epoch_200.pth'.format(checkpoint_dir)
                    )
            my_logger.info("Start Test")
            acc = cur_acc #max(acc, cur_acc) 
            print("acc:", acc)
            my_logger.info("End Test")
            # print(acc1, acc2, acc)
            my_logger.info('Trial {}/{} || Cycle {}/{} || Label set size {}: lambda_0: {} Selecting metric: {} Test acc {}'.format(
                trial + 1, cfg.ACTIVE_LEARNING.TRIALS, cycle + 1,
                cfg.ACTIVE_LEARNING.CYCLES, len(labeled_set),args.lambda_0,args.metric, acc)
            )
            #print("Unlabeled dataset length:",len(unlabeled_set))
            Performance[trial, cycle] = acc

            # update dataloaders
            # Measure emc of each data points in the subset
            print("Selecting data")
            EMC_Array = get_ems(model, unlabeled_loader,criterion_isda,metric=args.metric) #Ent, Maxp, ISDAL, Ran

            # Index in ascending order
            arg = np.argsort(EMC_Array)
            #print(arg)
            # Update the labeled dataset and the unlabeled dataset, respectively
            budget = cfg.ACTIVE_LEARNING.ADDENDUM
            labeled_set += list(torch.tensor(subset)[arg][-budget:].numpy())
            unlabeled_set =\
                list(torch.tensor(subset)[arg][:-budget].numpy()) + unlabeled_set[cfg.ACTIVE_LEARNING.SUBSET:]


            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)
            sampler = torch.utils.data.sampler.SubsetRandomSampler(labeled_set)

            # save a ckpt
            torch.save(
                        {
                            'cycle': cycle,
                            'state_dict_backbone': model.state_dict()
                        },
                        '{}/active_resnet18_cifar10_cycle{}_epoch_199.pth'.format(checkpoint_dir, cycle)
                    )
              
        # np.save(checkpoint_dir + '/l_set.npy', np.array(labeled_set))

    my_logger.info("Performance summary: ")
    my_logger.info("Trail 1: {}".format(Performance[0]))
    my_logger.info("Trail 2: {}".format(Performance[1]))
    my_logger.info("Trail 3: {}".format(Performance[2]))
    my_logger.info("Trail 4: {}".format(Performance[3]))
    my_logger.info("Trail 5: {}".format(Performance[4]))
    my_logger.info("Trail 6: {}".format(Performance[5]))
    my_logger.info("Trail 7: {}".format(Performance[6]))
    my_logger.info("Trail 8: {}".format(Performance[7]))    
    my_logger.info("Trail 9: {}".format(Performance[8]))
    my_logger.info("Trail 10: {}".format(Performance[9])) 