# coding: utf-8
import networkx as nx

import argparse, re, os
from urllib import parse
import scipy.sparse as smat
import numpy as np
from rf_util import *
from tqdm import tqdm
import pickle as pkl
import scipy as sp
# from xbert.rf_util import smat_util
import csv


def parse_mlc2seq_format(data_path):
    assert(os.path.isfile(data_path))
    with open(data_path, newline='') as csvfile:
        labels, corpus = [], []
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            if len(row) != 4:
                continue
            id, title, dscp, tag = row
            labels.append(tag)
            corpus.append(title + " " + dscp)
    return labels, corpus


def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-t", "--threshold", default=100, type=int, required=False)
    args = parser.parse_args()

    ds_path = './dataset'
    threshold = args.threshold

    head_instances = []
    head_Y = []

    heads = []
    tag2id = {}

    trn_labels, trn_corpus = parse_mlc2seq_format(ds_path + '/mlc2seq/programweb-data.csv')
    for idx, labels in tqdm(enumerate(trn_labels)):
        labels = np.array(list(labels.strip().split('###')))
        labels = [t for t in labels if t != '']
        lbs = []
        for label in list(set(labels)):
            if label not in tag2id:
                tag_id = len(tag2id)
                tag2id[label] = tag_id
                heads.append(label)
            lbs.append(tag2id[label])

        if len(lbs):
            head_instances.append(trn_corpus[idx])
            head_Y.append(lbs)

    with open(ds_path+'/mlc2seq/heads-'+str(threshold), 'wb') as g:
        pkl.dump(heads, g)

    head_instances = np.array(head_instances)
    head_Y = np.array(head_Y)

    ind = np.random.RandomState(seed=10).permutation(len(head_instances))
    split = int(len(head_instances) * 0.9)

    train_head_instances = head_instances[ind[:split]].tolist()
    train_head_Y = head_Y[ind[:split]].tolist()
    test_head_instances = head_instances[ind[split:]].tolist()
    test_head_Y = head_Y[ind[split:]].tolist()

    with open(ds_path+'/mlc2seq/train_heads_X-'+str(threshold), 'wb') as g:
        pkl.dump(train_head_instances, g)
    with open(ds_path+'/mlc2seq/train_heads_Y-'+str(threshold), 'wb') as g:
        pkl.dump(train_head_Y, g)

    with open(ds_path + '/mlc2seq/test_heads_X-' + str(threshold), 'wb') as g:
        pkl.dump(test_head_instances, g)
    with open(ds_path + '/mlc2seq/test_heads_Y-' + str(threshold), 'wb') as g:
        pkl.dump(test_head_Y, g)


if __name__ == '__main__':
    main()
