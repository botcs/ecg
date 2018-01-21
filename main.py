#! /usr/bin/python3.5
import torch
import torch.backends.cudnn as cudnn

import data_handler
import models
import trainer as T

import sys, os
import argparse


parser = argparse.ArgumentParser(description='PyTorch AF-detector Training')
parser.add_argument('--spectrogram', '-s', type=int, help='Use spectrogram with [NFFT]')
parser.add_argument('--equal_batch', '-e', action='store_true', help='Oversample less frequent class')
parser.add_argument('--debug', '-d', action='store_true', help='print stats')
parser.add_argument('--arch', '-a', default='vgg16_bn')
parser.add_argument('--gpu_id', default='0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.multiprocessing.set_sharing_strategy('file_system')
#name = splitext(basename(sys.argv[0]))[0]
name = args.arch

train_transformations = [
    data_handler.Crop(3000),
    data_handler.RandomMultiplier(-1),
]

test_transformations = [

]


in_channels = 1
if args.spectrogram is not None:
    spectral_transformations = [
        data_handler.Spectrogram(args.spectrogram),
        data_handler.Logarithm()
    ]
    train_transformations += spectral_transformations
    test_transformations += spectral_transformations
    name += "_freq%d" % args.spectrogram
    in_channels = args.spectrogram // 2 + 1

use_cuda = torch.cuda.is_available()

train_set = data_handler.DataSet(
    'data/train.csv',
    transformations=train_transformations,
    data_handler.load_composed,
    path='data/',
    tokens='NAO~')

test_set = data_handler.DataSet(
    'data/test.csv',
    transformations=test_transformations,
    data_handler.load_composed,
    path='data/',
    tokens='NAO~')


if args.equal_batch:
    train_set.equal_batch = True
    name += "_equal"


train_producer = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=32, shuffle=True,
        num_workers=32, collate_fn=data_handler.batchify)
test_producer = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=4, shuffle=True,
        num_workers=4, collate_fn=data_handler.batchify)
print("=> Building model %30s"%(args.arch))
net = models.__dict__[args.arch](in_channels=in_channels, num_classes=dataset.num_classes)

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

trainer = T.Trainer('saved/'+name, class_weight=[1, 1, 1], dryrun=args.debug)
if args.debug:
    print(net)
trainer(net, train_producer, test_producer, useAdam=True, epochs=1200)
