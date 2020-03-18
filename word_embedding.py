import re
import os
import json
import numpy as np
import pickle as pkl
import argparse

"""
for extracting word embedding yourself, please download pretrained model from one of the following links.
"""

url = {'glove': 'http://nlp.stanford.edu/data/glove.6B.zip',
       'google': 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing',
       'fasttext': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip'}

feat_len = 300


def embed_text_file(text_list, word_vectors, save_file):

    all_feats = []

    has = 0
    cnt_missed = 0
    missed_list = []
    for i in range(len(text_list)):
        class_name = text_list[i].lower()
        if i % 500 == 0:
            print('%d / %d : %s' % (i, len(text_list), class_name))
        feat = np.zeros(feat_len)

        options = class_name.split()
        cnt_word = 0
        for j in range(len(options)):
            now_feat = get_embedding(options[j].strip(), word_vectors)
            if np.abs(now_feat.sum()) > 0:
                cnt_word += 1
                feat += now_feat
        if cnt_word > 0:
            feat = feat / cnt_word

        if np.abs(feat.sum()) == 0:
            print('cannot find word ' + class_name)
            cnt_missed = cnt_missed + 1
            missed_list.append(class_name)
        else:
            has += 1
            feat = feat / (np.linalg.norm(feat) + 1e-6)

        all_feats.append(feat)

    all_feats = np.array(all_feats)

    for each in missed_list:
        print(each)
    print('does not have semantic embedding: ', cnt_missed, 'has: ', has)

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
        print('## Make Directory: %s' % save_file)
    with open(save_file, 'wb') as fp:
        pkl.dump(all_feats, fp)
    print('save to : %s' % save_file)


def get_embedding(entity_str, word_vectors):
    try:
        feat = word_vectors[entity_str]
        return feat
    except:
        feat = np.zeros(feat_len)

    str_set = list(filter(None, re.split("[ \-_]+", entity_str)))

    cnt_word = 0
    for i in range(len(str_set)):
        temp_str = str_set[i]
        try:
            now_feat = word_vectors[temp_str]
            feat = feat + now_feat
            cnt_word = cnt_word + 1
        except:
            continue

    if cnt_word > 0:
        feat = feat / cnt_word
    return feat


def get_glove_dict(txt_dir):
    print('load glove word embedding')
    txt_file = os.path.join(txt_dir, 'glove.6B.300d.txt')
    word_dict = {}
    feat = np.zeros(feat_len)
    with open(txt_file) as fp:
        for line in fp:
            words = line.split()
            assert len(words) - 1 == feat_len
            for i in range(feat_len):
                feat[i] = float(words[i+1])
            feat = np.array(feat)
            word_dict[words[0]] = feat
    print('loaded to dict!')
    return word_dict

def obtain_tag_embedding(ds_path='dataset', wv='glove', model_path='dataset'):

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-t", "--threshold", default=100, type=int, required=False)
    args = parser.parse_args()

    threshold = args.threshold

    if wv == 'glove':
        save_file = os.path.join(ds_path, 'word_embedding_model', 'glove_word2vec_programwebTag.pkl')
        if not os.path.exists(save_file):
            word_vectors = get_glove_dict(model_path)
    else:
        raise NotImplementedError

    if not os.path.exists(save_file):
        with open(ds_path + '/mlc2seq/heads-' + str(threshold), 'rb') as g:
            heads = pkl.load(g)

        print('obtain semantic word embedding', save_file)
        embed_text_file(heads, word_vectors, save_file)
    else:
        print('Embedding existed :', save_file, 'Skip!!!')


if __name__ == '__main__':
    obtain_tag_embedding()


