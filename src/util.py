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

def load_dataset_raw(fname):
    """
        Create MF dataset
        File format:
            uid sid rating timestamp
        Note:
            uid and sid starts from 1
    """
    user = []
    item = []
    score = []
    with open(fname, 'r') as f:
        elems = csv.reader(f, delimiter='\t')
        for uid, sid, rating, timestamp in elems:
            user.append(int(uid) - 1)
            item.append(int(sid) - 1)
            score.append(int(rating))
    return np.array(user), np.array(item), np.array(score).astype('float32')

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
                if len(rows) > 0:
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


def yield_uid_sids_from_file(fn):
    """
        Generator
    """
    fd = open(fn, "r")
    for line in fd:
        elems = line.rstrip().split(',')
        uid = elems[0]
        sids = set()
        for sid in elems[1].split('|'):
            if sid != '':
                sids.add(sid)
        yield uid, sids
    fd.close()


def make_sid_count_dict(fn):
    """
        Count
    """
    from collections import defaultdict
    sid_cnt_dict = defaultdict(int)
    for uid, sids in yield_uid_sids_from_file(fn):
        for sid in sids:
            sid_cnt_dict[sid] += 1
    return sid_cnt_dict


def make_sid_index(sid_cnt_dict):
    """
        Sort
    """
    import operator
    return sorted(sid_cnt_dict.items(), key=operator.itemgetter(1), reverse=True)


def make_index(fn):
    """
        Index
    """
    idx2uid = {}
    uid2idx = {}
    idx2sid = {}
    sid2idx = {}
    idx = 1
    for uid, _ in yield_uid_sids_from_file(fn):
        idx2uid[idx] = uid
        uid2idx[uid] = idx
        idx += 1
    sid_index_arr = make_sid_index(make_sid_count_dict(fn))
    idx = 1
    for sid, _ in sid_index_arr:
        idx2sid[idx] = sid
        sid2idx[sid] = idx
        idx += 1
    return tuple([idx2uid, uid2idx, idx2sid, sid2idx])


def make_uid_sids_dict(fn):
    """
        uid -> sids
    """
    uid_sids_dict = {}
    for uid, sids in yield_uid_sids_from_file(fn):
        uid_sids_dict[uid] = sids
    return uid_sids_dict


def negative_sample(uid, uid_sids_dict, idx2sid):
    """
        Negative sampling
    """
    import random
    return random.sample(list(idx2sid.values()), len(uid_sids_dict[uid]))


def make_train_dev_files(idx2uid, uid2idx, idx2sid, sid2idx, uid_sids_dict):
    """
        Create train dev files from uid sids info
    """
    from sklearn.model_selection import train_test_split
    import numpy as np
    fdt = open('fm_train.txt', 'w')
    fdd = open('fm_dev.txt', 'w')
    for uid, sids in uid_sids_dict.items():
        pos_sids = list(sids)
        pos_ratings = np.ones(len(pos_sids), dtype=int)
        neg_sids = negative_sample(uid, uid_sids_dict, idx2sid)
        neg_ratings = np.zeros(len(neg_sids), dtype=int)
        X = pos_sids + neg_sids
        y = pos_ratings.tolist() + neg_ratings.tolist()
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2)
        for xt, yt in zip(X_train, y_train):
            fdt.write('{}\t{}\t{}\t{}\n'.format(uid2idx[uid], sid2idx[xt], str(yt), '0'))
        for xd, yd in zip(X_dev, y_dev):
            fdd.write('{}\t{}\t{}\t{}\n'.format(uid2idx[uid], sid2idx[xd], str(yd), '0'))
    fdt.close()
    fdd.close()


def count_user_item_entry(fn):
    """
        Count user item entry in file
    """
    import csv
    entry_cnt = 0
    max_user_idx = 0
    max_item_idx = 0
    with open(fn, 'r') as f:
        elems = csv.reader(f, delimiter='\t')
        for uididx, sididx, rating, timestamp in elems:
            entry_cnt += 1
            max_user_idx = max(max_user_idx, int(uididx))
            max_item_idx = max(max_item_idx, int(sididx))
    return max_user_idx, max_item_idx, entry_cnt


def predict_uid_sid(model, uid, sids, idx2uid, uid2idx, idx2sid, sid2idx):
    """
        Predict
    """
    from scipy import sparse
    uididx = int(uid2idx[uid])
    user_cnt = len(idx2uid.values())
    X = sparse.lil_matrix((len(sids),
        len(idx2uid.values()) + len(idx2sid.values()))).astype('float32')
    idx = 0
    for sid in sids:
        sididx = int(sid2idx[sid])
        X[idx, uididx - 1] = 1
        X[idx, user_cnt + sididx - 1] = 1
        idx += 1
    return model.predict(X)


def get_uid_vec(model, uid, uid2idx, sid2idx):
    """
        Uid vec getter
    """
    return model.V_.T[int(uid2idx[uid] - 1)]


def get_sid_vec(model, sid, uid2idx, sid2idx):
    """
        Sid vec getter
    """
    user_cnt = len(uid2idx.values())
    return model.V_.T[user_cnt + int(sid2idx[sid] - 1)]


def predict_uid_sid_analytically(model, uid, sid, idx2uid, uid2idx, idx2sid, sid2idx):
    """
        Predict Analytically
    """
    uv = get_uid_vec(model, uid, uid2idx, sid2idx)
    sv = get_sid_vec(model, sid, uid2idx, sid2idx)
    return np.sum(uv * sv) + model.w0_ + model.w_[int(uid2idx[uid] - 1)]\
        + model.w_[len(uid2idx.values()) + int(sid2idx[sid] - 1)]


def get_top_n(model, idx2uid, uid2idx, idx2sid, sid2idx, top_n=20):
    """
        Get topn in each latent dims
    """
    user_cnt = len(uid2idx.values())
    item_cnt = len(sid2idx.values())
    item_mat = model.V_.T[user_cnt:user_cnt + item_cnt][:].T
    grouped_top_n = []
    for group_idx in range(np.shape(item_mat)[0]):
        dim_view = item_mat[group_idx][:]
        uididx_pairs = zip(dim_view.tolist(), range(1, item_cnt + 1))
        top_list = sorted(uididx_pairs, key=lambda x:x[0], reverse=True)[:top_n]
        grouped_top_n.append(top_list)
    return grouped_top_n


def describe_sid_iterable(iterable):
    """
        Tag description
    """
    import sid.resource_tag_readonly as tag
    for sid in iterable:
        yield sid, tag.get(sid)


def describe_grouped_top_n(grouped_top_n, idx2sid):
    """
        Group topn description
    """
    description_arr = []
    for group_id in range(len(grouped_top_n)):
        group = grouped_top_n[group_id]
        scores, sididxs = zip(*group)
        sids = map(lambda x:idx2sid[x], sididxs)
        description_arr.append(zip(scores, list(describe_sid_iterable(sids))))
    return description_arr


def recommend_mf(model, uid, idx2uid, uid2idx, idx2sid, sid2idx, uid_sids_dict, top_n=50):
    """
        Recommend using matrix factorization
    """
    from scipy import sparse
    recom_list = []
    sids_set = uid_sids_dict[uid]
    user_cnt = len(uid2idx.values())
    item_cnt = len(sid2idx.values())
    uididx = int(uid2idx[uid])
    X = sparse.lil_matrix((item_cnt, user_cnt + item_cnt)).astype('float32')
    for sididx in range(1, item_cnt + 1):
        X[sididx - 1, uididx - 1] = 1
        X[sididx - 1, user_cnt + sididx - 1] = 1
    y = model.predict(X)
    user_item_affinity = zip(y, list(range(1, item_cnt + 1)))
    user_item_affinity = sorted(user_item_affinity, key=lambda x:x[0], reverse=True)
    for score, sididx in user_item_affinity:
        sid = idx2sid[sididx]
        if sid not in sids_set:
            recom_list.append(tuple([score, sid]))
            if len(recom_list) == top_n:
                break
    return recom_list


def recommend_svdpp(model, uid, idx2uid, uid2idx, idx2sid, sid2idx, uid_sids_dict, top_n=50):
    """
        Recommend using SVDpp
    """
    from scipy import sparse
    recom_list = []
    sids_set = uid_sids_dict[uid]
    user_cnt = len(uid2idx.values())
    item_cnt = len(sid2idx.values())
    uididx = int(uid2idx[uid])
    X = sparse.lil_matrix((item_cnt, user_cnt + item_cnt*2)).astype('float32')

    # get user history factor slice, assuming ratings are 1
    user_history_factor = sparse.lil_matrix((1, user_cnt + item_cnt*2)).astype('float32')
    for sid in uid_sids_dict[uid]:
        sididx = sid2idx[sid]
        user_history_factor[0, user_cnt + item_cnt + sididx - 1] = 1
    if user_history_factor.count_nonzero() > 0:
        user_history_factor /= np.sqrt(user_history_factor.count_nonzero())
    
    for sididx in range(1, item_cnt + 1):
        X[sididx - 1, uididx - 1] = 1
        X[sididx - 1, user_cnt + sididx - 1] = 1
        X[sididx - 1] += user_history_factor
    y = model.predict(X)
    user_item_affinity = zip(y, list(range(1, item_cnt + 1)))
    user_item_affinity = sorted(user_item_affinity, key=lambda x:x[0], reverse=True)
    for score, sididx in user_item_affinity:
        sid = idx2sid[sididx]
        if sid not in sids_set:
            recom_list.append(tuple([score, sid]))
            if len(recom_list) == top_n:
                break
    return recom_list


def predict_uid_sid_svdpp(model, uid, sids, idx2uid, uid2idx, idx2sid, sid2idx, uid_sids_dict):
    """
        Prediction using svdpp
    """
    from scipy import sparse
    uididx = int(uid2idx[uid])
    user_cnt = len(uid2idx.values())
    item_cnt = len(sid2idx.values())
    X = sparse.lil_matrix((len(sids), user_cnt + item_cnt*2)).astype('float32')
    
    # get user history factor slice, assuming ratings are 1
    user_history_factor = sparse.lil_matrix((1, user_cnt + item_cnt*2)).astype('float32')
    for sid in uid_sids_dict[uid]:
        sididx = sid2idx[sid]
        user_history_factor[0, user_cnt + item_cnt + sididx - 1] = 1
    if user_history_factor.count_nonzero() > 0:
        user_history_factor /= np.sqrt(user_history_factor.count_nonzero())
    
    idx = 0
    for sid in sids:
        sididx = int(sid2idx[sid])
        X[idx, uididx - 1] = 1
        X[idx, user_cnt + sididx - 1] = 1
        X[idx - 1] += user_history_factor
        idx += 1
    return model.predict(X)