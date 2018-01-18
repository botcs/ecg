#! /usr/bin/python3.5
import torch
import torch.backends.cudnn as cudnn

import data_handler
import models
import trainer as T

import sys, os
from os.path import basename, splitext
import argparse


parser = argparse.ArgumentParser(description='PyTorch AF-detector Training')
parser.add_argument('--debug', '-d', action='store_true', help='print stats')
parser.add_argument('--arch', '-a', default='vgg16_bn')
parser.add_argument('--gpu_id', default=0, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
torch.multiprocessing.set_sharing_strategy('file_system')
#name = splitext(basename(sys.argv[0]))[0]
name = args.arch

transformations = [
    data_handler.Crop(2400),
    data_handler.RandomMultiplier(-1),
]

use_cuda = torch.cuda.is_available()

dataset = data_handler.DataSet(
    'data/REFERENCE.csv', data_handler.load_composed,
    transformations=transformations,
    path='data/',
    remove_noise=True, tokens='NAO')
train_set, eval_set = dataset.disjunct_split(.9)

train_producer = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=32, shuffle=True,
        num_workers=8, collate_fn=data_handler.batchify)
test_producer = torch.utils.data.DataLoader(
        dataset=eval_set, batch_size=32, shuffle=True,
        num_workers=8, collate_fn=data_handler.batchify)
print("=> Building model %30s"%(args.arch))
net = models.__dict__[args.arch](in_channels=1, num_classes=3)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

trainer = T.Trainer('saved/'+name, class_weight=[1, 5, 3],
                    dryrun=args.debug)
trainer(net, train_producer, test_producer, gpu_id=0, useAdam=True, epochs=1200)
