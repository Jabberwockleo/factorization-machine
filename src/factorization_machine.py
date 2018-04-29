#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: factorization_machine.py
Author: leowan(leowan)
Date: 2018/04/29 15:06:30
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from fastFM import sgd
from fastFM import als

def create_model(alg="als", type="regression", rank=5,
    n_iter=100, init_stdev=0.1, l2_reg_w=0.1, l2_reg_V=0.1,
    step_size=0.01):
    model = None
    if alg == "als" and type == "regression":
        model = als.FMRegression(n_iter=n_iter, init_stdev=init_stdev, rank=rank, l2_reg_w=l2_reg_w, l2_reg_V=l2_reg_V)
    elif alg == "als" and type == "classification":
        model = als.FMClassification(n_iter=n_iter, init_stdev=init_stdev, rank=rank, l2_reg_w=l2_reg_w, l2_reg_V=l2_reg_V)
    elif alg == "sgd" and type == "regression":
        model = sgd.FMRegression(n_iter=n_iter, init_stdev=init_stdev, rank=rank, l2_reg_w=l2_reg_w, l2_reg_V=l2_reg_V, step_size=step_size)
    elif alg == "sgd" and type == "classification":
        model = sgd.FMClassification(n_iter=n_iter, init_stdev=init_stdev, rank=rank, l2_reg_w=l2_reg_w, l2_reg_V=l2_reg_V, step_size=step_size)
    return model

def train(model, X_train, y_train, X_test, y_test, alg="als", type="regression", epoch=1000, step_size=1, trace_graph=False):
    model.fit(X_train, y_train)
    if trace_graph == True:
        if alg == "sgd":
            pass
        elif alg == "als":
            acc_train = []
            acc_test = []
            auc_train = []
            auc_test = []
            rmse_train = []
            rmse_test = []
            r2_score_train = []
            r2_score_test = []
            for _ in range(epoch):
                model.fit(X_train, y_train, n_more_iter=step_size)
                if type == "regression":
                    rmse_train.append(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
                    rmse_test.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
                    r2_score_train.append(r2_score(y_train, model.predict(X_train)))
                    r2_score_test.append(r2_score(y_test, model.predict(X_test)))

                if type == "classification":
                    acc_train.append(accuracy_score(y_train, model.predict(X_train)))
                    acc_test.append(accuracy_score(y_test, model.predict(X_test)))
                    auc_train.append(roc_auc_score(y_train, model.predict_proba(X_train)))
                    auc_test.append(roc_auc_score(y_test, model.predict_proba(X_test)))

            if type == "regression":
                fig, axes = plt.subplots(ncols=2, figsize=(9, 4))
                x = np.arange(epoch) * step_size
                axes[0].plot(x, rmse_train, label='RMSE-train', color='b', ls="--")
                axes[0].plot(x, rmse_test, label='RMSE-test', color='b')
                axes[1].plot(x, r2_score_train, label='R^2-train', color='b', ls="--")
                axes[1].plot(x, r2_score_test, label='R^2-test', color='b')
                axes[0].set_ylabel('RMSE', color='b')
                axes[1].set_ylabel('R^2', color='b')
                axes[0].legend()
                axes[1].legend()
                plt.show()

            if type == "classification":
                fig, axes = plt.subplots(ncols=2, figsize=(9, 4))
                x = np.arange(epoch) * step_size
                axes[0].plot(x, acc_train, label='Accuracy-train', color='b', ls="--")
                axes[0].plot(x, acc_test, label='Accuracy-test', color='b')
                axes[1].plot(x, auc_train, label='AUC-train', color='b', ls="--")
                axes[1].plot(x, auc_test, label='AUC-test', color='b')
                axes[0].set_ylabel('Accuracy', color='b')
                axes[1].set_ylabel('AUC', color='b')
                axes[0].legend()
                axes[1].legend()
                plt.show()
