#! /usr/bin/python3.5
import torch
import data_handler
from models import vgg16
import trainer as T
import sys, os
from os.path import basename, splitext

torch.multiprocessing.set_sharing_strategy('file_system')
name = splitext(basename(sys.argv[0]))[0]

transformations = [
    data_handler.Crop(2400),
    data_handler.RandomMultiplier(-1),
]

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

net = vgg16(in_channels=1, num_classes=3)

DRYRUN = '--dryrun' in sys.argv
RESTORE = '--restore' in sys.argv
print('DRYRUN:', DRYRUN)

trainer = T.Trainer('saved/'+name, class_weight=[1, 1, 1], dryrun=DRYRUN, restore=True)
trainer(net, train_producer, test_producer, gpu_id=0, useAdam=True)
