#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import unittest

from focal_loss_with_smoothing import FocalLossWithSmoothing


class Model(nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()
        net = torchvision.models.resnet18(pretrained=False)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.maxpool = net.maxpool
        self.relu = net.relu
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.fc = nn.Conv2d(512, n_classes, 3, 1, 1)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        feat = self.fc(feat)
        out = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
        return out


class TestStringMethods(unittest.TestCase):

    def test_predict(self):
        torch.manual_seed(15)
        random.seed(15)
        np.random.seed(15)

        height = 416
        width = 416
        batch_size = 12
        classes = 4
        n_iter = 100

        model = Model(classes)
        criteria = FocalLossWithSmoothing(classes)
        model.cuda()
        model.train()
        criteria.cuda()
        optim = torch.optim.SGD(model.parameters(), lr=1e-2)

        for it in range(n_iter):
            inten = torch.randn(batch_size, 3, height, width).cuda()
            lbs = torch.randint(0, classes, (batch_size, height, width)).cuda()
            logits = model(inten)
            loss = criteria(logits, lbs)
            optim.zero_grad()
            loss.backward()
            optim.step()
            with torch.no_grad():
                print('loss: ', loss.item())


if __name__ == '__main__':
    unittest.main()
