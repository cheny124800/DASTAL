import torch.nn as nn
from utils.util import *
import torch.optim.lr_scheduler as lr_scheduler
import time
from utils.test_IDA import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def train(
        cfg, logger, models, criterion,
        optimizers, schedulers, dataloaders, unlabeled_loader,
        num_epochs,args, cycle):
    logger.info('>> Train a Model.')
    best_prec1 = 0
    for epoch in range(num_epochs):
        train_net(dataloaders,models,criterion,optimizers,epoch,args,cfg,cycle)
        #prec1 = validate(val_loader, model, fc, ce_criterion, epoch)
        # remember best prec@1 and save checkpoint
        #best_prec1 = max(prec1, best_prec1)
        
        schedulers['model'].step()
    prec1 = test_acc(models, dataloaders, args,mode='test') 
    logger.info('>> Finished.')
    return prec1



def train_net(dataloaders, model, criterion, optimizer, epoch, args,cfg,cycle):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.train()
    #total = 0
    #correct1 = 0
    end = time.time()
    for data in dataloaders['train']:
        # measure data loading time
        data_time.update(time.time() - end)

        images = data[0].cuda()
        target = data[1].cuda()
        #target = get_one_hot_label(target, nCls=10)
        #images = images.cuda()
        #target = target.cuda()

        # compute output
        # output = model(images)
        # loss = criterion(output, target)
        
        #loss, output = criterion(model, images, target, args.lambda_0 * ( (epoch +  (cycle-1) * cfg.TRAIN.EPOCH)  /cfg.TRAIN.EPOCH * cfg.ACTIVE_LEARNING.CYCLES))
        loss, output = criterion(model, images, target, args.lambda_0 * ( epoch  /cfg.TRAIN.EPOCH ))
        # measure accuracy and record loss
        _, preds1 = torch.max(output.data, 1)
        #total += target.size(0)
        #correct1 += (preds1 == target).sum().item()
        # compute gradient and do SGD step
        #optimizer.zero_grad()
        optimizer['model'].zero_grad()
        loss.backward()
        #optimizer.step()
        optimizer['model'].step()    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    #acc1 = 100 * correct1 / total
    #print("Train acc:", acc1)
