# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import pickle
from pathlib import Path

import numpy as np
from keras.engine import Input, Model
from keras.layers import Embedding, Dense, add, Activation, TimeDistributed, LSTM, K

import embedding
import radd
from util.ProgressBar import ProgressBar

__author__ = "freemso"

ABB_DATA_FILE = "data/abb2ent/valid.data"

MODEL_WEIGHT_FILE = "model/rnn-radd.weight"
TRAIN_DATA_FILE = "temp/rnn-radd.dat"

MAX_ABB_WORD_LEN = 20

NUM_FEATURE = 9

RNN_DIM = 50
NUM_EPOCH = 100
BATCH_SIZE = 128
TEST_SPLIT = 0.0
VALIDATION_SPLIT = 0.2


class AbbRNN:
    def __init__(self):
        self.cv = embedding.CharVector()
        self.radd = radd.RADD()

        self.model = self._build()

        model_weight_file = Path(MODEL_WEIGHT_FILE)
        if model_weight_file.is_file():
            logging.info("loading AbbRNN model weight...")
            self.model.load_weights(MODEL_WEIGHT_FILE)
        else:
            logging.info("training AbbRNN model...")
            # check whether there is preprocessed data
            train_data_file = Path(TRAIN_DATA_FILE)
            if not train_data_file.is_file():
                # pre-process the data
                # preprocessing and model-building must be separated
                # or there will be ran-out-of-memory error
                data, labels, fv = self._preprocess()
                logging.info("storing training data to disk...")
                with open(TRAIN_DATA_FILE, "wb") as out_file:
                    pickle.dump((data, labels, fv), out_file)
            else:
                # load pre-processed data and label
                logging.info("loading training data from disk...")
                with open(TRAIN_DATA_FILE, "rb") as in_file:
                    data, labels, fv = pickle.load(in_file)

            # build&train the model
            self._train(data, labels, fv)

    def _preprocess(self):
        # load data
        word2abb = load_abb_data(self.cv)

        data = np.zeros((len(word2abb), MAX_ABB_WORD_LEN), dtype="int32")
        labels = np.zeros((len(word2abb), MAX_ABB_WORD_LEN), dtype="int32")
        feature_grams = np.zeros(
            (len(word2abb), MAX_ABB_WORD_LEN, NUM_FEATURE, radd.MAX_RADD_WORD_LEN), dtype="int32")

        bar = ProgressBar(total=len(word2abb))
        for idx, item in enumerate(word2abb):
            bar.log()
            word = item[0]
            abb = item[1]

            word_seq = []
            abb_seq = []

            i = 0
            for char in word:
                word_seq.append(self.cv.char2idx[char] + 1)
                t = abb.find(char, i)
                if t == -1:
                    abb_seq.append(0)
                else:
                    i = t + 1
                    abb_seq.append(1)
            word_trunc = word_seq[:MAX_ABB_WORD_LEN]
            data[idx, :len(word_trunc)] = word_trunc
            abb_trunc = abb_seq[:MAX_ABB_WORD_LEN]
            labels[idx, :len(abb_trunc)] = abb_trunc
            feature_grams[idx, :len(word_trunc)] = generate_feature_grams(word_trunc)

        flattened_feature_grams = feature_grams.reshape((len(word2abb) * MAX_ABB_WORD_LEN * NUM_FEATURE,
                                                         radd.MAX_RADD_WORD_LEN))
        fv = self.radd.predict(flattened_feature_grams).reshape(
            (len(word2abb), MAX_ABB_WORD_LEN, NUM_FEATURE))

        logging.info('Shape of data tensor: {}'.format(data.shape))
        logging.info('Shape of label tensor: {}'.format(labels.shape))
        logging.info('Shape of feature tensor: {}'.format(fv.shape))

        return data, labels, fv

    def _build(self):
        # embedding layer weight setting
        embedding_weights = np.zeros((self.cv.vocab_size + 1, self.cv.vector_size))
        embedding_weights[0, :] = np.zeros((self.cv.vector_size,))
        for index, char in self.cv.idx2char.items():
            embedding_weights[index + 1, :] = self.cv.wv[char]

        # build model
        logging.info('building model...')
        i = Input(shape=(MAX_ABB_WORD_LEN,), name='char_input')

        c = Embedding(self.cv.vocab_size + 1, self.cv.vector_size,
                      weights=[embedding_weights], input_length=MAX_ABB_WORD_LEN,
                      trainable=False, name="emb")(i)

        m = Input(shape=(MAX_ABB_WORD_LEN, NUM_FEATURE), name='feature_input')

        s = LSTM(RNN_DIM, activation='sigmoid',
                 use_bias=False, return_sequences=True, name="lstm")(add([c, m]))

        v_s = TimeDistributed(Dense(1, use_bias=False, name="v_s"), name="v_s_time")(s)
        fy_m = TimeDistributed(Dense(1, use_bias=False, name="fy_m"), name="fy_m_time")(m)

        y = Activation(activation='sigmoid', name='model_output')(add([v_s, fy_m]))

        model = Model(inputs=[i, m], outputs=y)

        # compile model
        model.compile(loss="binary_crossentropy",
                      optimizer="sgd")

        return model

    def _train(self, data, labels, fv):
        assert data.shape[0] == labels.shape[0] == fv.shape[0]
        num_data = data.shape[0]

        # shuffle the data
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices].reshape(list(labels.shape) + [1])
        fv = fv[indices][:]

        # split the data into a training set and a test set
        logging.info("splitting data...")
        num_test_data = int(VALIDATION_SPLIT * num_data)
        x_train = data[:-num_test_data]
        f_train = fv[:-num_test_data][:]
        y_train = labels[:-num_test_data][:]
        x_test = data[-num_test_data:]
        f_test = fv[-num_test_data:][:]
        y_test = labels[-num_test_data:][:]

        # to reduce memory cost
        del data, labels, fv

        # train model
        logging.info("training model...")
        self.model.fit({"char_input": x_train, "feature_input": f_train},
                       {"model_output": y_train},
                       validation_data=({"char_input": x_test, "feature_input": f_test},
                       {"model_output": y_test}),
                       epochs=NUM_EPOCH, batch_size=BATCH_SIZE)

        # save model weights
        self.model.save_weights(MODEL_WEIGHT_FILE, overwrite=True)

    def predict(self, words):
        num_data = len(words)
        data = np.zeros((num_data, MAX_ABB_WORD_LEN), dtype="int32")
        feature_grams = np.zeros(
            (num_data, MAX_ABB_WORD_LEN, NUM_FEATURE, radd.MAX_RADD_WORD_LEN), dtype="int32")
        for idx, word in enumerate(words):
            word_seq = []
            for char in word:
                if char in self.cv.char2idx.keys():
                    word_seq.append(self.cv.char2idx[char] + 1)
                else:
                    word_seq.append(0)
            word_trunc = word_seq[:MAX_ABB_WORD_LEN]
            data[idx, :len(word_trunc)] = word_trunc
            feature_grams[idx, :len(word_trunc)] = generate_feature_grams(word_trunc)
        flattened_feature_grams = feature_grams.reshape((num_data * MAX_ABB_WORD_LEN * NUM_FEATURE,
                                                         radd.MAX_RADD_WORD_LEN))
        fv = self.radd.predict(flattened_feature_grams).reshape(
            (num_data, MAX_ABB_WORD_LEN, NUM_FEATURE))

        results = self.model.predict({"char_input": data, "feature_input": fv})

        abb_list = []
        for idx, result in enumerate(results):
            abb = ""
            for i, label in enumerate(result):
                if label[0] > 0.5:
                    abb += self.cv.idx2char[data[idx][i] - 1]
            abb_list.append(abb)
        return abb_list


def load_abb_data(cv):
    logging.info("loading abbreviation data...")
    word2abb = []
    # count = 0
    with open(ABB_DATA_FILE) as in_file:
        for line in in_file:
            # count += 1
            # if count > 1000:
            #     break
            abb, word = line.strip().split("\t")

            if len(word) < MAX_ABB_WORD_LEN and len(abb) > MIN_ABB_WORD_LEN and all([c in cv.wv for c in word]):
                word2abb.append((word, abb))

    return word2abb


def generate_feature_grams(word_vector):
    fg = np.zeros(shape=(len(word_vector), NUM_FEATURE, radd.MAX_RADD_WORD_LEN))
    for idx, char_idx in enumerate(word_vector):
        # 2-gram
        if idx >= 2:
            fg[idx][0][:2] = [word_vector[idx - 2], word_vector[idx - 1]]
        if idx >= 1:
            fg[idx][1][:2] = [word_vector[idx - 1], char_idx]
        if idx < len(word_vector) - 1:
            fg[idx][2][:2] = [char_idx, word_vector[idx + 1]]
        if idx < len(word_vector) - 2:
            fg[idx][3][:2] = [word_vector[idx + 1], word_vector[idx + 2]]
        # 3-gram
        if idx >= 2:
            fg[idx][4][:3] = [word_vector[idx - 2], word_vector[idx - 1], char_idx]
        if 1 <= idx < len(word_vector) - 1:
            fg[idx][5][:3] = [word_vector[idx - 1], char_idx, word_vector[idx + 1]]
        if idx < len(word_vector) - 2:
            fg[idx][6][:3] = [char_idx, word_vector[idx + 1], word_vector[idx + 2]]
        # 4-gram
        if 2 <= idx < len(word_vector) - 1:
            fg[idx][7][:4] = [word_vector[idx - 2], word_vector[idx - 1], char_idx, word_vector[idx + 1]]
        if 1 <= idx < len(word_vector) - 2:
            fg[idx][8][:4] = [word_vector[idx - 1], char_idx, word_vector[idx + 1], word_vector[idx + 2]]
    return fg