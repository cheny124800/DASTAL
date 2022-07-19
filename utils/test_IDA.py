
import torch
import time

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Pytorch 1.7
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def test(model, dataloaders,args, mode='val'):
    model.eval()
    with torch.no_grad():
        end = time.time()
        for (inputs, labels) in dataloaders[mode]:
 
            images = inputs.cuda()
            target = labels.cuda()
            
            #images = images.cuda()
            #target = target.cuda()

            # compute output
            output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

    return acc1


def test_acc(model, dataloaders,args, mode='val'):
    model.eval()

    total = 0
    correct1 = 0
   
    with torch.no_grad():
        end = time.time()
        for (inputs, labels) in dataloaders[mode]:
 
            images = inputs.cuda()
            target = labels.cuda()
            
            #images = images.cuda()
            #target = target.cuda()

            # compute output
            output = model(images)
            # measure accuracy and record loss
            _, preds1 = torch.max(output.data, 1)
            total += target.size(0)
            correct1 += (preds1 == target).sum().item()
            acc1 = 100 * correct1 / total
    return acc1

