#! /usr/bin/python3.5
import torch
import torch.backends.cudnn as cudnn

import data_handler
import models
import trainer as T

import sys, os
import argparse


seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available:
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='PyTorch AF-detector Training')
parser.add_argument('--spectrogram', '-s', type=int, help='Use spectrogram with [NFFT]', required=True)
parser.add_argument('--spectrogram_arch', help='Architecture to use on spectrogram feature extraction', required=True)
parser.add_argument('--spectrogram_ckpt', help='Load trained weights for spectrogram feature extractor')

parser.add_argument('--time_arch', help='Architecture to use on time feature extraction', required=True)
parser.add_argument('--time_ckpt', help='Load trained weights for time feature extractor')

parser.add_argument('--equal_batch', '-e', action='store_true', help='Oversample less frequent class')
parser.add_argument('--debug', '-d', action='store_true', help='print stats')
parser.add_argument('--gpu_id', default='0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.multiprocessing.set_sharing_strategy('file_system')
name = args.time_arch + '_joint_freq%d' % args.spectrogram + '_' + args.spectrogram_arch

time_transformations = [

]

freq_transformations = [
    data_handler.Spectrogram(args.spectrogram),
    data_handler.Logarithm()
]

train_transformations = [
    data_handler.Crop(3000),
    data_handler.RandomMultiplier(-1),
    data_handler.Fork({
        'time': time_transformations,
        'freq': freq_transformations
    })
]

test_transformations = [
    data_handler.Fork({
            'time': time_transformations,
            'freq': freq_transformations
        })
]

train_set = data_handler.DataSet(
    'data/train.csv',
    load=data_handler.load_forked,
    transformations=train_transformations,
    path='data/',
    tokens='NAO')

test_set = data_handler.DataSet(
    'data/test.csv',
    load=data_handler.load_forked,
    transformations=test_transformations,
    path='data/',
    tokens='NAO')

use_cuda = torch.cuda.is_available()

if args.equal_batch:
    train_set.equal_batch = True
    name += "_equal"


train_producer = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=32, shuffle=True,
        num_workers=16, collate_fn=data_handler.batchify_forked)
test_producer = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=32, shuffle=True,
        num_workers=16, collate_fn=data_handler.batchify_forked)
print("=> Loading time model %30s"%(args.time_arch))
time_net = models.__dict__[args.time_arch](in_channels=1, num_classes=train_set.num_classes)
if args.time_ckpt is not None:
    time_net.load_state_dict(torch.load(args.time_ckpt))
# classifier supression
time_net.classifier = torch.nn.Dropout(0)

print("=> Loading spectrogram model %23s"%(args.spectrogram_arch))
freq_net = models.__dict__[args.spectrogram_arch](
    in_channels=args.spectrogram//2+1, num_classes=train_set.num_classes)
if args.spectrogram_ckpt is not None:
    freq_net.load_state_dict(torch.load(args.spectrogram_ckpt))
# classifier supression
freq_net.classifier = torch.nn.Dropout(0)

if use_cuda:
    time_net.cuda()
    freq_net.cuda()
    cudnn.benchmark = True

num_time_features = time_net.num_features
num_freq_features = freq_net.num_features
print('# Time features: %5d\n# Spectrogram features: %5d'%(num_time_features, num_freq_features))
num_features = num_time_features + num_freq_features
classifier = torch.nn.Sequential(
    torch.nn.AlphaDropout(0.1),
    torch.nn.BatchNorm1d(num_features),
    torch.nn.SELU(),
    torch.nn.Conv1d(num_features, num_features, 1),
    torch.nn.AlphaDropout(0.05),
    torch.nn.BatchNorm1d(num_features),
    torch.nn.SELU(),
    torch.nn.Conv1d(num_features, train_set.num_classes, 1),
    torch.nn.AdaptiveAvgPool1d(1)
)

pretrained = args.spectrogram_ckpt is not None and args.time_ckpt is not None


net = models.ForkedModel(
    pretrained=pretrained,
    gpu_id=args.gpu_id,
    classifier=classifier,
    time=time_net,
    freq=freq_net
)


trainer = T.Trainer('saved/'+name, class_weight=[1]*train_set.num_classes, dryrun=args.debug)
if args.debug:
    print(net)
trainer(net, train_producer, test_producer, useAdam=True, epochs=1200)
