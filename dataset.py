# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    
"""
import json
import logging
import pathlib
import pickle
import pprint
from collections import defaultdict

import batch
import gensim
import grequests
import numpy as np
import requests
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

__author__ = "freemso"

ABB2ENT_FILENAME = "data/abb2ent_valid.txt"

CONCEPT2IDX_FILENAME = "data/concept2idx.json"
TAG2IDX_FILENAME = "data/tag2idx.json"
WORD2CONCEPT_FILENAME = "data/word2concept.json"

TRAIN_DATA_FILENAME = "data/abb2ent_train.pickle"
EVAL_DATA_FILENAME = "data/abb2ent_eval.pickle"

CHAR_EMB_MODEL_FILENAME = "model/char2vec.200.key.model"
WORD_EMB_MODEL_FILENAME = "model/word2vec.200.key.model"

concept_url = "http://knowledgeworks.cn:20314/probaseplus/pbapi/getconcepts"

VALIDATE_SPLIT_RATE = 0.1

MAX_SEQ_LEN = 15

# Set logger
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

char2vec = gensim.models.KeyedVectors.load(CHAR_EMB_MODEL_FILENAME)
word2vec = gensim.models.KeyedVectors.load(WORD_EMB_MODEL_FILENAME)

# Load char2idx
char2idx = dict((char, idx + 2) for idx, char in enumerate(char2vec.index2word))
char2idx["PAD"] = 0
char2idx["UNK"] = 1

# Load word2idx
word2idx = dict((word, idx + 2) for idx, word in enumerate(word2vec.index2word))
word2idx["PAD"] = 0
word2idx["UNK"] = 1

char_vocab_size = len(char2idx)
char_vec_dim = 200
word_vocab_size = len(word2idx)
word_vec_dim = 200


# Load the data
def load_data():
    if pathlib.Path(TRAIN_DATA_FILENAME).is_file() and \
            pathlib.Path(EVAL_DATA_FILENAME).is_file() and \
            pathlib.Path(CONCEPT2IDX_FILENAME) and \
            pathlib.Path(TAG2IDX_FILENAME) and \
            pathlib.Path(WORD2CONCEPT_FILENAME):
        data_train = pickle.load(open(TRAIN_DATA_FILENAME, "rb"))
        data_eval = pickle.load(open(EVAL_DATA_FILENAME, "rb"))
        concept2idx = json.load(open(CONCEPT2IDX_FILENAME))
        tag2idx = json.load(open(TAG2IDX_FILENAME))
        word2concept = json.load(open(WORD2CONCEPT_FILENAME))

    else:
        abb2ent = [line.strip().split("\t") for line in open(ABB2ENT_FILENAME)]
        abb2ent = set((abb, ent) for abb, ent in abb2ent)

        ent_abbs_map = defaultdict(list)
        abb_ents_map = defaultdict(list)
        for abb, ent in abb2ent:
            ent_abbs_map[ent].append(abb)
            abb_ents_map[abb].append(ent)
        # Remove those ent-abb pair of which abb appears more than one time
        for abb, ents in abb_ents_map.items():
            if len(ents) > 1:
                for ent in ents:
                    ent_abbs_map[ent].remove(abb)

        # Training and validation set only contains those 1-1 pairs
        data_train = [(ent, abb_list[0]) for ent, abb_list in ent_abbs_map.items() if len(abb_list) == 1]
        ent_list = [ent for ent, _ in data_train]
        abb_list = [abb for _, abb in data_train]
        x_char, x_word, x_tag, x_concept, concept2idx, tag2idx, word2concept = get_x_for_train(ent_list)
        labels = get_y(data_train)

        # Those 1-n pairs serve as test set
        data_eval = [(ent, abb_list) for ent, abb_list in ent_abbs_map.items() if len(abb_list) > 1]
        ent_list_eval = [ent for ent, _ in data_eval]
        abbs_list_eval = [abbs for _, abbs in data_eval]
        x_char_eval, x_word_eval, x_tag_eval, x_concept_eval = get_x_for_eval(
            ent_list_eval, concept2idx, tag2idx, word2concept)

        data_train = {
            "ent": ent_list,
            "abb": abb_list,
            "x_char": x_char,
            "x_word": x_word,
            "x_concept": x_concept,
            "x_tag": x_tag,
            "label": labels
        }
        pickle.dump(
            data_train,
            open(TRAIN_DATA_FILENAME, "wb")
        )

        data_eval = {
            "ent": ent_list_eval,
            "abbs": abbs_list_eval,
            "x_char": x_char_eval,
            "x_word": x_word_eval,
            "x_concept": x_concept_eval,
            "x_tag": x_tag_eval
        }
        pickle.dump(
            data_eval,
            open(EVAL_DATA_FILENAME, "wb")
        )
        json.dump(
            concept2idx,
            open(CONCEPT2IDX_FILENAME, "w"),
            ensure_ascii=False
        )
        json.dump(
            tag2idx,
            open(TAG2IDX_FILENAME, "w"),
            ensure_ascii=True
        )
        json.dump(
            word2concept,
            open(WORD2CONCEPT_FILENAME, "w"),
            ensure_ascii=True
        )

    return data_train, data_eval, concept2idx, tag2idx, word2concept


def get_x_for_eval(ent_list, concept2idx, tag2idx, word2concept):
    boson_tagger_client = batch.Segmentor()
    concepts_idx_list = []
    chars_idx_list = []
    words_idx_list = []
    tags_idx_list = []

    tag_results = boson_tagger_client.analysis(ent_list)
    for tag_result in tag_results:
        words_idx, chars_idx, tags_idx, concepts_idx = [], [], [], []
        for word, tag in zip(tag_result['word'], tag_result['tag']):
            for char in word:
                words_idx.append(word2idx[word] if word in word2idx else 1)
                chars_idx.append(char2idx[char] if char in char2idx else 1)
                tags_idx.append(tag2idx[tag] if tag in tag2idx else 1)
                concepts_idx.append(concept2idx[word2concept[word]] if word in word2concept else 1)
        words_idx_list.append(words_idx)
        tags_idx_list.append(tags_idx)
        chars_idx_list.append(chars_idx)
        concepts_idx_list.append(concepts_idx)

    x_char = pad(chars_idx_list)
    x_word = pad(words_idx_list)
    x_tag = pad(tags_idx_list)
    x_concept = pad(concepts_idx_list)

    return x_char, x_word, x_tag, x_concept


def get_char_emb_layer():
    emb_weights = np.zeros((char_vocab_size, char_vec_dim))

    for char, idx in char2idx.items():
        if char in char2vec:
            emb_weights[idx, :] = char2vec[char]
        else:
            emb_weights[idx, :] = np.random.uniform(-0.25, 0.25, char_vec_dim)

    emb_layer = Embedding(
        char_vocab_size,
        char_vec_dim,
        weights=[emb_weights],
        input_length=MAX_SEQ_LEN,
        mask_zero=True,
        name="embedding",
    )

    return emb_layer


def get_word_emb_layer():
    emb_weights = np.zeros((word_vocab_size, word_vec_dim))

    for word, idx in word2idx.items():
        if word in word2vec:
            emb_weights[idx, :] = word2vec[word]
        else:
            emb_weights[idx, :] = np.random.uniform(-0.25, 0.25, word_vec_dim)

    emb_layer = Embedding(
        word_vocab_size,
        word_vec_dim,
        weights=[emb_weights],
        input_length=MAX_SEQ_LEN,
        mask_zero=True
    )

    return emb_layer


def get_labels(ent, abb):
    label = []
    i = 0
    for c in ent:
        t = abb.find(c, i)
        if t == -1:
            label.append(0)
        else:
            i = t + 1
            label.append(1)
    return label


def get_y(ent_abb_list):
    labels_list = [get_labels(ent, abb) for ent, abb in ent_abb_list]
    y = pad_sequences(
        labels_list,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post",
        value=0
    )
    y = y.reshape(list(y.shape) + [1])
    return y


def pad(idx_list):
    return pad_sequences(
        idx_list,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post",
        value=0
    )


def get_x_for_train(ent_list):
    boson_tagger_client = batch.Segmentor()
    chars_list = []
    words_list = []
    tags_list = []

    word_vocab = set()
    tag_vocab = set()
    concept_vocab = set()

    tag_results = boson_tagger_client.analysis(ent_list)
    for tag_result in tag_results:
        words, chars, tags = [], [], []
        for word, tag in zip(tag_result['word'], tag_result['tag']):
            word_vocab.add(word)
            tag_vocab.add(tag)
            for char in word:
                words.append(word)
                chars.append(char)
                tags.append(tag)
        words_list.append(words)
        tags_list.append(tags)
        chars_list.append(chars)

    # Get concepts
    word2concept = {}
    word_vocab_list = list(word_vocab)
    iter_times = 5
    old_len = len(word_vocab_list)
    while len(word_vocab_list) > 0 and iter_times > 0:
        iter_times -= 1
        max_conn_size = 256
        for x in range(0, len(word_vocab_list), max_conn_size):
            request_list = (
                grequests.get(concept_url, params={"kw": word, "start": 0})
                for word in word_vocab_list[x:x + max_conn_size]
            )
            response_list = grequests.map(request_list)
            for idx, response in enumerate(response_list):
                if response:
                    if response.status_code == requests.codes.ok:
                        try:
                            response_json = response.json()
                            if response_json["numcon"] > 0:
                                first_concept = response_json["concept"][0][0]
                                concept_vocab.add(first_concept)
                                word2concept[word_vocab_list[x + idx]] = first_concept
                        except Exception:
                            logging.info("Error! Word: {}".format(word_vocab_list[x + idx]))
                    response.close()
        err_list = [word for word in word_vocab_list if word not in word2concept]

        if len(err_list) > 0:
            logging.info("Still {} words not process. Go another round.".format(len(err_list)))

        word_vocab_list = err_list

        if len(word_vocab_list) == old_len:
            break
        else:
            old_len = len(word_vocab_list)

    logging.info("All: {}, have concept: {}".format(
        len(word_vocab), len(word2concept)
    ))

    concept2idx = {c: i + 2 for i, c in enumerate(concept_vocab)}
    tag2idx = {t: i + 2 for i, t in enumerate(tag_vocab)}

    concepts_idx_list = [
        [concept2idx[word2concept[word]] if word in word2concept else 1 for word in words]
        for words in words_list
    ]
    chars_idx_list = [
        [char2idx[c] if c in char2idx else 1 for c in chars]
        for chars in chars_list
    ]
    words_idx_list = [
        [word2idx[w] if w in word2idx else 1 for w in words]
        for words in words_list
    ]
    tags_idx_list = [
        [tag2idx[t] if t in tag2idx else 1 for t in tags]
        for tags in tags_list
    ]

    x_char = pad(chars_idx_list)
    x_word = pad(words_idx_list)
    x_tag = pad(tags_idx_list)
    x_concept = pad(concepts_idx_list)

    return x_char, x_word, x_tag, x_concept, concept2idx, tag2idx, word2concept
