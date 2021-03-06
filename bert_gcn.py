import argparse
import os, sys, time
import pickle as pkl
import scipy.sparse as smat
#from xbert.rf_util import smat_util
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
#from gensim.models import KeyedVectors as KV
from sklearn.neighbors import NearestNeighbors as NN
import random
# from mather.bert import *
# import xbert.data_utils as data_utils
# import xbert.rf_linear as rf_linear
# import xbert.rf_util as rf_util
from Hyperparameters import Hyperparameters
from GCN import GraphConvolution, gen_A, gen_adj

from word_embedding import *

import math
import sklearn.metrics as sm
from output_measure import AveragePrecisionMeter


class GraphUtil():
    def __init__(self, Y, num_labels):
        self.Y = Y
        self.nums = np.zeros(num_labels)
        self.adj = np.zeros([num_labels, num_labels])

    def gen_graph(self):
        for label_list in tqdm(self.Y):
            for i in range(len(label_list)):
                for j in range(len(label_list)):
                    self.adj[label_list[i]][label_list[j]] += 1
                    self.adj[label_list[j]][label_list[i]] += 1

    def cal_degree(self):
        self.nums = np.sum(self.adj, axis=1)

class BertGCN(BertModel):
    def __init__(self, config, ft, num_labels, res, H, device_num):
        super(BertGCN, self).__init__(config)
        self.device = torch.device('cuda:' + device_num)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.H = get_tensor(H, self.device)
        self.ft = ft
        self.num_labels = num_labels
        self.FCN = nn.Linear(768, num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.apply(self.init_bert_weights)
        self.gcn_weight1 = Parameter(torch.Tensor(H.shape[1], 1500))

        stdv = 1. / math.sqrt(self.gcn_weight1.size(1))
        self.gcn_weight1.data.uniform_(-stdv, stdv)

        self.gcn_weight2 = Parameter(torch.Tensor(1500, 768))

        stdv = 1. / math.sqrt(self.gcn_weight2.size(1))
        self.gcn_weight2.data.uniform_(-stdv, stdv)

        self.lkrelu = nn.LeakyReLU(0.2)
        self.A = torch.tensor(gen_A(num_labels, res)).float().to(self.device)
        # self.A = Parameter(torch.from_numpy(gen_A(num_labels, res)).float()).to(self.device)
        self.adj = gen_adj(self.A).detach()

        # self.FCN2 = nn.Linear(num_labels, num_labels)

    def forward(self, input_ids, gcn_limit=False, token_type_ids=None, attention_mask=None):
        if self.ft:
            with torch.no_grad():
                _, pooled_output = self.bert(input_ids, output_all_encoded_layers=False)
        else:
            _, pooled_output = self.bert(input_ids, output_all_encoded_layers=False)

        bert_logits = pooled_output#self.dropout(pooled_output)
        skip = self.lkrelu(self.FCN(bert_logits))
        adj = self.adj
        # adj = gen_adj(self.A).detach()

        x = torch.matmul(adj, torch.matmul(self.H, self.gcn_weight1))#self.dropout(torch.matmul(self.H, self.gcn_weight1)))
        x = self.lkrelu(x)
        x = torch.matmul(adj, torch.matmul(x, self.gcn_weight2))
        x = self.lkrelu(x)
        x = x.transpose(1, 0)
        x = torch.matmul(bert_logits, x)
        #logits = x + skip

        logits = x
        # logits = self.FCN2(logits)
        return self.softmax(logits)
        # return logits
    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.bert.parameters(), 'lr': lr * lrp},
            {'params': self.FCN.parameters(), 'lr': lr},
            {'params': self.gcn_weight1, 'lr': lr},
            {'params': self.gcn_weight2, 'lr': lr},
        ]

    def get_bertout(self, input_ids):
        with torch.no_grad():
            _, pooled_output = self.bert(input_ids, output_all_encoded_layers=False)
        return pooled_output

def get_binary_vec(label_list, output_dim, divide=False):
    if divide:
        bs = int(len(label_list)/10)
        res = smat.lil_matrix(np.zeros([len(label_list[:bs]), output_dim]))
        for step in range(1, int(len(label_list)/bs+1)):
            t = smat.lil_matrix(np.zeros([len(label_list[step*bs:(step+1)*bs]), output_dim]))
            res = smat.vstack([res, t])
        res = smat.lil_matrix(res)
    else:
        res = smat.lil_matrix(np.zeros([len(label_list), output_dim]))
    # print(res.shape)
    for i, label in enumerate(label_list):
        res[i, label]=1
    return res

def get_tensor(M, dv):
    return torch.tensor(M).float().to(dv)

class BertGCNClassifier():
    def __init__(self, hypes, heads, t, device_num, ft, epochs, gutil, label_space, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.ds_path = '../datasets/' + hypes.dataset
        self.hypes = hypes
        self.epochs = epochs
        self.ft = ft
        self.t = t
        self.H = label_space
        self.heads = heads
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.criterion =  nn.MultiLabelSoftMarginLoss()
        # self.criterion =  nn.BCELoss()
        self.device = torch.device('cuda:' + device_num)
        self.model = BertGCN.from_pretrained('bert-base-uncased', ft, len(heads), gutil, self.H, device_num)
        self.ap_meter = AveragePrecisionMeter()

    def update_label_feature(self, X, Y, ep, output_dir):
        feature_path = output_dir + 'L.BERT-head_'+str(self.t)+'-ep_'+str(ep)+'.npz'
        self.model.eval()
        self.model.to(self.device)
        print('updating label fetures...')
        all_input_ids = torch.tensor(X)

        Y = get_binary_vec(Y, self.H.shape[0], divide=False).transpose() # m * n
        outputs = np.zeros([Y.shape[0], 768])

        sample_size = 20
        for i in trange(Y.shape[0]):
            y = Y[i].todense()
            inds = np.where(y)[0]
            inds = np.random.choice(inds, sample_size)
            input_ids = all_input_ids[inds].to(self.device)
            with torch.no_grad():
                output = self.model.get_bertout(input_ids).cpu().detach().numpy()
            outputs[i] = np.mean(output, axis=0)

        return outputs

    def train(self, X, Y, val_X, val_Y, model_path=None, ft_from=0):
        if model_path: # fine tuning
            self.model.load_state_dict(torch.load(model_path))
            epohcs_range = range(ft_from+1, ft_from+self.epochs+1)
        else:
            epohcs_range = range(1, self.epochs+1)

        all_input_ids = torch.tensor(X)
        bs = 50
        self.model.train()
        self.model.to(self.device)
        total_run_time = 0.0

        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = BertAdam(optimizer_grouped_parameters, #self.model.get_config_optim(self.hypes.learning_rate, lrp=0.1),
                             lr=self.hypes.learning_rate,
                             warmup=self.hypes.warmup_rate)

        nb_tr_examples = 0
        for epoch in tqdm(epohcs_range):
            eval_t = 0
            tr_loss = 0
            nb_tr_steps = 0
            num_corrects = 0
            total = 0
            start_time = time.time()

            for step in range(int(len(X)/bs)-1):
                # if step % self.hypes.log_interval != 0:
                #     continue
                input_ids = all_input_ids[step*bs:(step+1)*bs].to(self.device)
                labels = get_binary_vec(Y[step*bs:(step+1)*bs], self.H.shape[0])

                labels = get_tensor(labels.toarray(), self.device)
                c_pred = self.model(input_ids)

                loss = self.criterion(c_pred, labels)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if step % self.hypes.log_interval == 0:
                #     map = get_score(c_pred, labels)

            # if epoch % 20 == 0:
            output_dir = '../save_models/gcn_classifier/'+self.hypes.dataset+'/t-'+str(self.t)+'_ep-'+str(epoch)+'/'
            val_inputs = np.array(val_X)
            val_labels = np.array(val_Y)
            acc = self.evaluate(val_inputs, val_labels)

            self.model.train()
            # if not self.ft:
            #     self.model.H = get_tensor(self.update_label_feature(X, Y, epoch, output_dir), self.device)
            #self.save(output_dir)


    def evaluate(self, X, Y, model_path=''):
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        all_input_ids = torch.tensor(X)
        all_Ys = get_binary_vec(Y, len(self.heads))
        bs = self.hypes.eval_batch_size
        self.model.eval()
        self.model.to(self.device)

        num_corrects = 0
        start_time = time.time()
        self.ap_meter.reset()

        print('Inferencing...')
        for step in trange(int(len(X)/bs)-1):
            input_ids = all_input_ids[step*bs:(step+1)*bs].to(self.device)
            labels = all_Ys[step*bs:(step+1)*bs]
            labels = get_tensor(labels.toarray(), self.device)
            with torch.no_grad():
                c_pred = self.model(input_ids)

            self.ap_meter.add(c_pred.data.cpu(), labels.cpu())

        map = 100 * np.nanmean(self.ap_meter.value().numpy())
        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        print('mAP {map:.3f}'.format(map=map))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))

    def get_bert_token(self, trn_text, only_CLS=False):
        X = []
        # self.model.cuda(1)
        print('========================================================')
        print('getting sentence embedding...')
        for text in tqdm(trn_text):
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(marked_text)
            if len(tokenized_text) > self.max_seq_len:
                tokenized_text = tokenized_text[:self.max_seq_len-1] + ['[SEP]']

            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            if len(indexed_tokens) < self.max_seq_len:
                padding = [0] * (self.max_seq_len - len(indexed_tokens))
                indexed_tokens += padding
            else:
                indexed_tokens = indexed_tokens[:self.max_seq_len]

            X.append(indexed_tokens)
        print('========================================================')
        return X

    def save(self, output_dir):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

def load_data(X_path, head_X, bert):
    if os.path.isfile(X_path):
        with open(X_path, 'rb') as g:
            X = pkl.load(g)
    else:
        X = bert.get_bert_token(head_X)
        with open(X_path, 'wb') as g:
            pkl.dump(X, g)

    return X

def load_label(ds_path):
    # label_space = smat.load_npz(ds_path+'/L.elmo.npz')
    # label_space = smat.lil_matrix(label_space)

    label_path = os.path.join(ds_path, 'word_embedding_model', 'glove_word2vec_programwebTag.pkl') #ds_path+'/L.elmo_768.npy'

    if os.path.exists(label_path):
        with open(label_path, 'rb') as fp:
            label_space = pkl.load(fp)
    else:
        print('reducing dimensions in label space with t-SVD...')
        # tsvd = TruncatedSVD(768)
        # label_space = tsvd.fit_transform(label_space)
        # np.save(label_path, label_space)
    return label_space


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-ds", "--dataset", default="ProgrammerWeb-12K", type=str, required=False)
    parser.add_argument("-t", "--head_threshold", default=100, type=int)
    parser.add_argument("-gpu", "--device_num", default='1', type=str)
    parser.add_argument("-train", "--is_train", default=1, type=int)
    parser.add_argument("-ep", "--epochs", default=20, type=int)
    parser.add_argument("-ft", "--fine_tune", default=0, type=int)
    parser.add_argument("-from", "--ft_from", default=0, type=int)
    args = parser.parse_args()

    ft = (args.fine_tune == 1)
    ft_from = args.ft_from
    hypes = Hyperparameters(args.dataset)
    ds_path = './dataset'
    device_num = args.device_num
    head_threshold = args.head_threshold
    with open(ds_path+'/mlc2seq/train_heads_X-'+str(head_threshold), 'rb') as g:
        trn_head_X = pkl.load(g)
    with open(ds_path+'/mlc2seq/train_heads_Y-'+str(head_threshold), 'rb') as g:
        trn_head_Y = pkl.load(g)
    with open(ds_path+'/mlc2seq/test_heads_X-'+str(head_threshold), 'rb') as g:
        test_head_X = pkl.load(g)
    with open(ds_path+'/mlc2seq/test_heads_Y-'+str(head_threshold), 'rb') as g:
        test_head_Y = pkl.load(g)

    with open(ds_path+'/mlc2seq/heads-'+str(head_threshold), 'rb') as g:
        heads = pkl.load(g)

    label_space = load_label(ds_path)
    label_space = label_space[:len(heads)]

    output_dim = len(heads)
    gutil = GraphUtil(trn_head_Y, output_dim)
    gutil.gen_graph()
    gutil.cal_degree()
    bert = BertGCNClassifier(hypes, heads, head_threshold, device_num, ft, args.epochs, gutil, label_space, max_seq_len=256)
    trn_X_path = ds_path+'/head_data/trn_X-' + str(head_threshold)
    test_X_path = ds_path+'/head_data/test_X-' + str(head_threshold)
    trn_X = load_data(trn_X_path, trn_head_X, bert)
    test_X = load_data(test_X_path, test_head_X, bert)
    print('Number of labels:', len(heads))
    print('Number of trn instances:', len(trn_X))

    if args.is_train:
        if ft:
            print('======================Start Fine-Tuning======================')
            model_path = '../save_models/gcn_classifier/'+hypes.dataset+'/t-'+str(head_threshold)+'_ep-' + str(ft_from)+'/pytorch_model.bin'
            bert.train(trn_X, trn_head_Y, test_X, test_head_Y, model_path, ft_from)
        else:
            print('======================Start Training======================')
            bert.train(trn_X, trn_head_Y, test_X, test_head_Y)
        bert.evaluate(test_X, test_head_Y)
        output_dir = '../save_models/gcn_classifier/'+hypes.dataset+'/t-'+str(head_threshold)+'_ep-' + str(args.epochs + ft_from)+'/'
        bert.save(output_dir)

    else:
        model_path = '../save_models/gcn_classifier/'+args.dataset+'/t-'+str(head_threshold)+'_ep-'+str(ft_from)+'/pytorch_model.bin'
        print('======================Start Testing======================')
        bert.evaluate(test_X, test_head_Y, model_path)


if __name__ == '__main__':
    main()

#
