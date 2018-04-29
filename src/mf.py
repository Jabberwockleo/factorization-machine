#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: mf.py
Author: leowan(leowan)
Date: 2018/04/29 17:08:56
"""
from sklearn.model_selection import train_test_split
import factorization_machine as fm
import util

if __name__ == "__main__":
    ### Matrix factorization
    X_base, Y_base = util.load_dataset_mf('../data/ml-100k/ua.base', user_cnt=943, item_cnt=1682, entry_cnt=90570)
    X_test, Y_test = util.load_dataset_mf('../data/ml-100k/ua.test', user_cnt=943, item_cnt=1682, entry_cnt=9430)
    X_train, X_dev, y_train, y_dev = train_test_split(X_base, Y_base)
    y_test = Y_test
    model = fm.create_model(alg="als", type="regression", n_iter=0, init_stdev=0.1, l2_reg_w=2.1,
                            l2_reg_V=20.1, rank=5, step_size=0.01)
    fm.train(model, X_train, y_train, X_dev, y_dev, alg="als", type="regression",
             epoch=20, step_size=1, trace_graph=True)

    ytr = util.evaluate_regression(model, X_train, y_train)
    print(list(zip(ytr[:5], y_train[:5])))
    yde = util.evaluate_regression(model, X_dev, y_dev)
    print(list(zip(yde[:5], y_dev[:5])))
    yte = util.evaluate_regression(model, X_test, Y_test)
    print(list(zip(yte[:5], y_test[:5])))

