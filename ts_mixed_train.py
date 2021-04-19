import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from two_steam_resnet import *
from imagefolder_sal import *
from imagefolder_sal_intra import *
# from imagefolder_sal_intra_dynamic import * 
from imagenet_sal_all_bg import *

from utils import Logger

import numpy as np
from tensorboardX import SummaryWriter

from torch.nn import functional as F

CUDA_LAUNCH_BLOCKING="1"


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--arch', default='resnet152',type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr_model', default=0.1, type=float)
parser.add_argument('--sche_epo_model', default=30,type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_acc1_A = 0
best_acc1_B = 0


def main():
    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)


def soft_label_loss(output, target,size_average=True):
    # if not self.training:
    #     # Loss is normal cross entropy loss between the model output and the
    #     # target.
    #     return F.cross_entropy(output, target,
    #                            size_average=self.size_average)

    assert type(output) == tuple and len(output) == 2 and output[0].size() == \
        output[1].size(), "output must a pair of tensors of same size."

    # Target is ignored at training time. Loss is defined as KL divergence
    # between the model output and the refined labels.
    model_output, refined_labels = output
    if refined_labels.requires_grad:
        raise ValueError("Refined labels should not require gradients.")

    model_output_log_prob = F.log_softmax(model_output, dim=1)
    del model_output

    # Loss is -dot(model_output_log_prob, refined_labels). Prepare tensors
    # for batch matrix multiplicatio
    refined_labels = refined_labels.unsqueeze(1)
    model_output_log_prob = model_output_log_prob.unsqueeze(2)

    
    # Compute the loss, and average/sum for the batch.
    cross_entropy_loss = -torch.bmm(refined_labels, model_output_log_prob)
    # import pdb
    # pdb.set_trace()
    if size_average:
        cross_entropy_loss = cross_entropy_loss.mean()
    else:
        cross_entropy_loss = cross_entropy_loss.sum()
    # Return a pair of (loss_output, model_output). Model output will be
    # used for top-1 and top-5 evaluation.
    model_output_log_prob = model_output_log_prob.squeeze(2)
    # import pdb
    # pdb.set_trace()
    # return (cross_entropy_loss, model_output_log_prob)
    return cross_entropy_loss




def main_worker(gpu, ngpus_per_node, args):
    global best_acc1_A
    global best_acc1_B
    args.gpu = gpu


    args.arch = 'ts_resnet34'
    ext_info = '-ts-mixed-test'
    args.batch_size=256
    args.epochs=32
    args.lr_model=0.001
    args.sche_epo_model=8
    args.start_epoch = 0
    
    
    Input = '-bg-obj==bg-obj-'
    pretrained = False
    ts_pretrain = True
    set_low_encoder_grad = False
    args.load_ts_path = '/BS/sun_project_multimodal/nobackup/weiqin/project_code/imagenet_sal/load_pretrain/'+args.arch+'.pkl'
    # args.load_ts_path = '/BS/sun_project_multimodal/nobackup/weiqin/project_code/imagenet_sal/load_pretrain/'+args.arch+'_10.pkl'
    # args.load_ts_path = '/BS/sun_project_multimodal/nobackup/weiqin/project_code/imagenet_sal/model_saved/imagenet-1000ts_resnet18_batsize-256_epochs-48_lr_m-0.001__schel_model-16_pretrain-False_ts_pretr-True_set_low_grad=False-bg-obj==bg-obj--ts-mixed-1th/8_cnn.pkl'
    dataset_type = 'imagenet'          # macro-30  or micro-100
    # dataset_type = 'micro-10-1000'          # macro-30  or micro-100
    class_num = 1000
    idx = dataset_type+'-'+str(class_num)+args.arch+'_batsize-'+str(args.batch_size)+'_epochs-'+str(args.epochs)+'_lr_m-'+str(args.lr_model)+'_'+\
            '_schel_model-'+str(args.sche_epo_model)+'_pretrain-'+str(pretrained)+'_ts_pretr-'+str(ts_pretrain)+'_set_low_grad='+str(set_low_encoder_grad)+Input+ext_info
    print(args)
    print(idx)
    
    log_path = './newlogs/'+idx
    model_saved_path = './model_saved/'+idx+'/'

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)



    # create model
    if args.arch == 'resnet152':
        model = resnet152(pretrained=pretrained,num_classes=class_num).cuda()
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=pretrained,num_classes=class_num).cuda()
    elif args.arch == 'ts_resnet50':
        model = ts_resnet50(pretrained=pretrained,num_classes=class_num).cuda()
    elif args.arch == 'ts_resnet34':
        model = ts_resnet34(pretrained=pretrained, num_classes=class_num).cuda()
    elif args.arch == 'ts_resnet18':
        model = ts_resnet18(pretrained=pretrained, num_classes=class_num).cuda()
    else:
        print('arch wrong')
        model = None
    # fusion_module = Sum_Fusion_2(512*4,1000).cuda()
    
    if set_low_encoder_grad == False:
        print('set false')
        model.set_low_encode_grad_false()
    model = torch.nn.DataParallel(model)
    
    if ts_pretrain:
        model.load_state_dict(torch.load(args.load_ts_path))

    print(model)

    # for param in model.parameters():
    #     param.requires_grad = False

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer_model = torch.optim.SGD(model.parameters(), args.lr_model,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    print('model construction complete')

    cudnn.benchmark = True

    # Data loading code


    if dataset_type == 'imagenet':
        train_image_root = '/BS/sun_project_multimodal/nobackup/weiqin/imagenet/training_imgs'
        train_sal_root = '/BS/sun_project_multimodal/nobackup/weiqin/Dataset/sal_imagenet/Train' 


        val_image_root=  '/BS/sun_project_multimodal/nobackup/weiqin/imagenet/val_images'
        val_sal_root =  '/BS/sun_project_multimodal/nobackup/weiqin/Dataset/sal_imagenet/Val'
    elif dataset_type == 'micro-10-1000':
        train_image_root = '/BS/sun_project_multimodal/nobackup/weiqin/imagenet/training_imgs'
        train_sal_root = '/BS/sun_project_multimodal/nobackup/weiqin/Dataset/new_micro_imagenet/Train' 

        val_image_root = '/BS/sun_project_multimodal/nobackup/weiqin/imagenet/val_images'
        val_sal_root = '/BS/sun_project_multimodal/nobackup/weiqin/Dataset/new_micro_imagenet/Val' 
        

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    transform_val_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    transform_val_sal = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    val_dataset = ImageFolder_Sal(
        root_image=val_image_root,
        root_sal=val_sal_root,
        transform_img = transform_val_img,
        transform_sal = transform_val_sal
        )

    print(val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('val dataloader complete')

    logger = Logger(os.path.join(log_path,'log.txt'))
    logger_best = Logger(os.path.join(log_path,'log_best.txt'))
    logger.set_names(['Epoch', 'Train Loss',  'Train Acc1.', 'Train Acc5.', 'Test Acc1.', 'Test Acc5.', 'Test Loss'])
    logger_best.set_names(['Epoch', 'acc_1', 'acc_5'])

    writer = SummaryWriter(comment = idx)

    for epoch in range(args.start_epoch, args.epochs):
        print(epoch)
        adjust_learning_rate(optimizer_model, epoch, args)

        print('lr:')
        for param_group in optimizer_model.param_groups:
            print(param_group['lr'])


        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        transform_train_img = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        transform_train_sal = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

        train_dataset = ImageFolder_Sal_Intra(
            root_image=train_image_root,
            root_sal=train_sal_root,
            intra_num=1,
            transform_img = transform_train_img,
            transform_sal = transform_train_sal
            )

        print(train_dataset.__len__())
        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        print('train dataloader complete')

        # train for one epoch
        train_acc1, train_acc5, train_loss, test_acc1_A, test_acc5_A, test_loss_A = train(train_loader, val_loader, model, criterion, optimizer_model, epoch, args)
        # train_acc1, train_acc5, train_loss = 0,0,0

        # evaluate on validation set
        test_acc1, test_acc5, test_loss = validate_A(val_loader, model, criterion, args)
        # acc1_B, loss_B, acc1_C = 0, 0, 0

        writer.add_scalars('scalar/train_acc',{'train_acc@1':train_acc1,'train_acc@5':train_acc5},epoch*2+1)
        writer.add_scalar('scalar/train_loss',train_loss,epoch*2+1)

        writer.add_scalars('scalar/val_acc',{'val_acc@1':test_acc1_A,'val_acc@5':test_acc5_A},epoch*2)
        writer.add_scalar('scalar/val_loss',test_loss_A,epoch*2)

        writer.add_scalars('scalar/val_acc',{'val_acc@1':test_acc1,'val_acc@5':test_acc5},epoch*2+1)
        writer.add_scalar('scalar/val_loss',test_loss,epoch*2+1)

        logger.append([epoch*2,train_loss, train_acc1, train_acc5,test_acc1_A, test_acc5_A,test_loss_A])
        logger.append([epoch*2+1,train_loss, train_acc1, train_acc5,test_acc1, test_acc5,test_loss])

        # remember best acc@1 and save checkpoint
        if test_acc1 > best_acc1_A:
            best_acc1_A = test_acc1
            logger_best.append([epoch,test_acc1,test_acc5])
            torch.save(model.state_dict(),model_saved_path+'max_acc_cnn.pkl')
        # if acc1_B > best_acc1_B:
        #     best_acc1_B = acc1_B
        #     logger_best.append([epoch,acc1_A,acc1_B])
        #     torch.save(model.state_dict(),model_saved_path+'max_acc_cnn_B.pkl')
        #     torch.save(fusion_module.state_dict(),model_saved_path+'max_acc_fusion_B.pkl')
        if epoch % 2 == 0:
            torch.save(model.state_dict(),model_saved_path+str(epoch)+'_cnn.pkl')


    writer.close()



def train(train_loader, val_loader, model, criterion, optimizer_model , epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (image_main, sal_main, image_bg, sal_bg, fname, targets) in enumerate(train_loader):
    # for i, (main_obj, sup_bg, fname, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)
        
        main_image = image_main.cuda()
        main_sal = sal_main.cuda()
        sup_image = image_bg.cuda()
        sup_sal = sal_bg.cuda()
        targets = targets.cuda()


        main_sal_x = main_sal.expand_as(main_image)
        main_obj = main_image * main_sal_x
        # main_bg = main_image * (1 - main_sal_x)

        sup_sal_x = sup_sal.expand_as(sup_image)
        # sup_obj = sup_image * sup_sal_x
        sup_bg = sup_image * (1 - sup_sal_x)
        
        # sup_bg = sup_bg.cuda()
        # main_obj = main_obj.cuda()
        # targets = targets.cuda()

        # output,_ = model(sup_bg,main_obj)
        output = model(sup_bg,main_obj)


        # loss = soft_label_loss((output,soft_label),targets)
        loss = criterion(output, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.detach(), targets, topk=(1, 5))
        losses.update(loss.item(), targets.size(0))
        top1.update(acc1[0], targets.size(0))
        top5.update(acc5[0], targets.size(0))

        # compute gradient and do SGD step

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if i == 4852:
            test_acc1_A, test_acc5_A, test_loss_A = validate_A(val_loader, model, criterion, args)
            model.train()
        test_acc1_A, test_acc5_A, test_loss_A = 0.0, 0.0, 0.0

    return top1.avg, top5.avg, losses.avg, test_acc1_A, test_acc5_A, test_loss_A

def validate_A(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # top1_B = AverageMeter('Acc_A@B', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (main_image, main_sal,_, targets) in enumerate(val_loader):

            # if i == 20:
            #     break

            main_image = main_image.cuda()
            main_sal = main_sal.cuda()
            targets = targets.cuda()

            main_sal_x = main_sal.expand_as(main_image)
            main_obj = main_image * main_sal_x
            main_bg = main_image * (1 - main_sal_x)


            output= model(main_bg,main_obj)

            loss = criterion(output, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            # acc1_A, _ = accuracy(output1, target, topk=(1,5))
            # acc1_B, _ = accuracy(output2, target, topk=(1,5))
            losses.update(loss.item(), targets.size(0))
            top1.update(acc1[0], targets.size(0))
            top5.update(acc5[0], targets.size(0))
            # top1_A.update(acc1_A[0], images_bg.size(0))
            # top1_B.update(acc1_B[0], images_bg.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc5@1 {top5.avg:.3f} loss {losses.avg:.3f}'
              .format(top1=top1, top5=top5, losses=losses))

    return top1.avg, top5.avg, losses.avg




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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer_model, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 6 epochs"""
    lr_model = args.lr_model * (0.1 ** (epoch // args.sche_epo_model))
    
    for param_group in optimizer_model.param_groups:
        param_group['lr'] = lr_model


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()