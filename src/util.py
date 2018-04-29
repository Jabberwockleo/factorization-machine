#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: util.py
Author: leowan(leowan)
Date: 2018/04/29 10:25:12
"""
import numpy as np
import csv
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy import sparse

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('mse:', mean_squared_error(y_test, y_pred))
    print('r2 :', r2_score(y_test, y_pred))
    return y_pred

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print('acc:', accuracy_score(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_pred_proba))
    return y_pred_proba

def load_dataset_mf(fname, user_cnt, item_cnt, entry_cnt):
    """
        Create MF dataset
        File format:
            uid sid rating timestamp
        Note:
            uid and sid starts from 1
    """
    X = sparse.lil_matrix((entry_cnt, user_cnt + item_cnt)).astype('float32')
    Y = []
    ent_idx = 0
    with open(fname, 'r') as f:
        elems = csv.reader(f, delimiter='\t')
        for uid, sid, rating, timestamp in elems:
            X[ent_idx, int(uid) - 1] = 1
            X[ent_idx, user_cnt + int(sid) - 1] = 1
            Y.append(int(rating))
            ent_idx += 1
    Y = np.array(Y).astype('float32')
    return X, Y

def load_dataset_svd(fname, user_cnt, item_cnt, entry_cnt):
    """
        Create SVD++ dataset
        File format:
            uid sid rating timestamp
        Note:
            uid and sid starts from 1
    """
    X = sparse.lil_matrix((entry_cnt, user_cnt + item_cnt*2)).astype('float32')
    Y = []

    item_mat = sparse.eye(item_cnt).tolil()
    item_vec = []
    for idx in range(item_cnt):
        item_vec.append(item_mat.getrow(idx))
    user_dict = {} # uid: rating vector
    with open(fname, 'r') as f:
        elems = csv.reader(f, delimiter='\t')
        for uid, sid, rating, timestamp in elems:
            ratvec = item_vec[int(sid) - 1] * int(rating)
            if uid not in user_dict:
                user_dict[uid] = ratvec
            else:
                uservec = user_dict[uid]
                rows, cols = ratvec.nonzero()
                uservec[rows[0], cols[0]] = 0
                uservec += ratvec
                user_dict[uid] = uservec
    for uid, ratvec in user_dict.items():
        ratvec /= np.sqrt(ratvec.count_nonzero())
    ent_idx = 0
    with open(fname, 'r') as f:
        elems = csv.reader(f, delimiter='\t')
        for uid, sid, rating, timestamp in elems:
            ratvec = user_dict[uid]
            X[ent_idx, int(uid) - 1] = 1
            X[ent_idx, user_cnt + int(sid) - 1] = 1
            for idx in ratvec.nonzero()[1]:
                X[ent_idx, user_cnt + item_cnt + idx] = ratvec[0, idx]
            Y.append(int(rating))
            ent_idx += 1
    Y = np.array(Y).astype('float32')
    return X, Y
