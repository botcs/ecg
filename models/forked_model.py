import math
import torch
import torch.nn as nn

from collections import OrderedDict


class ForkedModel(nn.Module):
    def __init__(self, pretrained, classifier, gpu_id=None, **models):
        super(ForkedModel, self).__init__()
        self.pretrained = pretrained
        models = OrderedDict(sorted(models.items(), key=lambda t: t[0]))
        self.key2ind = {k:i for i, k in enumerate(models.keys())}
        self.models = torch.nn.ModuleList(list(models.values()))
        self.classifier = classifier
        self.gpu_id = gpu_id

        if pretrained:
            for p in self.models.parameters():
                p.requres_grad = False

    def parameters(self):
        if self.pretrained:
            return self.classifier.parameters()
        return super(ForkedModel, self).parameters()

    def forward_fork(self, x, key):
        #if self.gpu_id is not None:
        x = {x_key:x_val.cuda(self.gpu_id) for x_key,x_val in x.items()}
        x = self.models[self.key2ind[key]].forward(x[key])
        return x

    def forward_features(self, x):
        res = [self.forward_fork(x, key) for key in self.key2ind.keys()]
        return torch.cat(res, 1)

    def forward(self, x):
        features = self.forward_features(x)
        if self.pretrained:
            return self.classifier(features.detach()).squeeze()
        return self.classifier(features).squeeze()


    def cuda(self, gpu_id=None):
        super(ForkedModel, self).cuda(gpu_id)
        self.gpu_id = gpu_id

    def cpu(self, *args, **kwargs):
        super(ForkedModel, self).cpu(*args, **kwargs)
        self.gpu_id = None
