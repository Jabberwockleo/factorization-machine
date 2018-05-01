#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: mlpmf.py
Author: leowan(leowan)
Date: 2018/05/01 20:55:43
"""

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon, nd
import util

class FactorizationBlock(gluon.Block):
    def __init__(self, user_cnt, item_cnt, rank, type='regression'):
        super(FactorizationBlock, self).__init__()
        self.user_cnt = user_cnt
        self.item_cnt = item_cnt
        self.rank = rank
        self.type = type
        with self.name_scope():
            self.user_layer = gluon.nn.Embedding(self.user_cnt, rank)
            self.item_layer = gluon.nn.Embedding(self.item_cnt, rank)
            self.dense_layer = gluon.nn.Dense(rank, activation='relu')
            if self.type == 'classification':
                self.output_layer = gluon.nn.Dense(2)

    def forward(self, users, items):
        uo = self.user_layer(users)
        uo = self.dense_layer(uo)
        
        io = self.item_layer(items)
        io = self.dense_layer(io)
        
        if self.type == 'regression':
            score_pred = uo * io
            score_pred = nd.sum(score_pred, axis=1)
            return score_pred
        elif self.type == 'classification':
            return self.output_layer(uo * io)

def evaluate_network(model, data_iter, ctx=mx.cpu(), type='regression'):
    metric = mx.metric.RMSE()
    if type == 'classification':
        metric = mx.metric.Accuracy()
    for user, item, score in data_iter:
        user = user.as_in_context(ctx).reshape((user.shape[0],))
        item = item.as_in_context(ctx).reshape((user.shape[0],))
        score = score.as_in_context(ctx).reshape((user.shape[0],))
        predictions = model.forward(user, item)
        metric.update(preds=[predictions], labels=[score])
    return metric.get()[1]

def train(model, dataiter_train, dataiter_dev, dataiter_test, epoches=10, type='regression'):
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.02, 'wd': 0.01})
    loss_function = gluon.loss.L2Loss()
    if type == 'classification':
        loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    loss_trace = []
    for ep in range(epoches):
        for i, (user, item, score) in enumerate(dataiter_train):
            user = user.as_in_context(mx.cpu()).reshape((user.shape[0],))
            item = item.as_in_context(mx.cpu()).reshape((user.shape[0],))
            score = score.as_in_context(mx.cpu()).reshape((user.shape[0],))
            with mx.autograd.record():
                output = model.forward(user, item)
                loss = loss_function(output, score)
                if i % 100 == 0:
                    loss_trace.append(nd.sum(loss).asscalar() / user.shape[0])
                loss.backward()
            trainer.step(user.shape[0])
            
        metric_name = 'RMSE'
        if type == 'classification':
            metric_name = 'Accuracy'
        print("Epoch {}: \n  {} training set: {} dev set: {}, test set: {}".format(ep, metric_name,
            evaluate_network(model, dataiter_train, type=type),
            evaluate_network(model, dataiter_dev, type=type),
            evaluate_network(model, dataiter_test, type=type)))
    plt.figure()
    plt.plot(np.arange(len(loss_trace))  * 100, loss_trace)
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    user_cnt=943
    item_cnt=1682
    rank = 64
    batch_size = 50

    uid_train, uid_dev, sid_train, sid_dev, score_train, score_dev = train_test_split(
        *util.load_dataset_raw('../data/ml-100k/ua.base'),
        test_size=0.1)
    uid_test, sid_test, score_test = util.load_dataset_raw('../data/ml-100k/ua.test')

    dataset_train = gluon.data.ArrayDataset(uid_train, sid_train, score_train)
    dataset_dev = gluon.data.ArrayDataset(uid_dev, sid_dev, score_dev)
    dataset_test = gluon.data.ArrayDataset(uid_test, sid_test, score_test)
    
    dataiter_train = gluon.data.DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    dataiter_dev = gluon.data.DataLoader(dataset_dev, shuffle=True, batch_size=batch_size)
    dataiter_test = gluon.data.DataLoader(dataset_test, shuffle=True, batch_size=batch_size)
    
    model = FactorizationBlock(user_cnt, item_cnt, rank, type='regression')
    model.collect_params().initialize(init=mx.init.Xavier(), ctx=mx.cpu(), force_reinit=True)
    
    train(model, dataiter_train, dataiter_dev, dataiter_test, epoches=5)
    