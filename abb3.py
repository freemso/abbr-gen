# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import pickle
from pathlib import Path

import jieba
import numpy as np
from keras.engine import Input, Model
from keras.layers import Embedding, LSTM, K, TimeDistributed, Dense, concatenate
from keras.preprocessing import sequence

import embedding

__author__ = "freemso"

RAW_DATA_FILE = "data/abb2ent/men2ent_v2_clean.data"

MODEL_WEIGHT_FILE = "model/abb3.2.model"
TEMP_DATA_FILE = "temp/abb.data"

MAX_ABB_CHAR_LEN = 15
MIN_ABB_CHAR_LEN = 1

RNN_DIM = 1000

NUM_EPOCH = 32
BATCH_SIZE = 32

TEST_SPLIT = 0.2
DEV_SPLIT = 0.1


def loss_function(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true))


class AbbRNN:
    def __init__(self):
        self.c2v = embedding.CharVector()
        self.w2v = embedding.WordVector()

        processed_data_path = Path(TEMP_DATA_FILE)
        if not processed_data_path.is_file():
            logging.info("Preprocessing data...")
            self._preprocess()
            logging.info("Storing preprocessed data...")
            with open(TEMP_DATA_FILE, "wb") as out_file:
                pickle.dump((self.c_test, self.c_dev, self.c_train,
                             self.w_test, self.w_dev, self.w_train,
                             self.y_test, self.y_dev, self.y_train), out_file)
        else:
            logging.info("Loading preprocessed data...")
            with open(TEMP_DATA_FILE, "rb") as in_file:
                self.c_test, self.c_dev, self.c_train, self.w_test, self.w_dev, self.w_train, \
                    self.y_test, self.y_dev, self.y_train = pickle.load(in_file)

        logging.info("Building model...")
        self.model = self._build()

        model_weight_path = Path(MODEL_WEIGHT_FILE)
        if model_weight_path.is_file():
            logging.info("Loading model weight...")
            self.model.load_weights(MODEL_WEIGHT_FILE)
        else:
            # Train model
            logging.info("Training model...")
            self._train(self.c_train, self.w_train, self.y_train)

        # Evaluate the model
        logging.info("Evaluating...")
        accuracy = self.evaluate(self.c_dev, self.w_dev, self.y_dev)
        logging.info("Accuracy of the model is {}".format(accuracy))

    def _preprocess(self):
        # Load data
        logging.info("Loading raw data...")
        ent_as_char_list = []
        ent_as_word_list = []
        abb_as_bin_list = []
        # count = 0
        with open(RAW_DATA_FILE) as in_file:
            for line in in_file:
                # count += 1
                # if count > 1000:
                #     break
                ent_as_char, ent_as_word, abb_as_bin = [], [], []

                abb, ent = line.strip().split("\t")
                ent_seg = jieba.cut(ent)

                # Set up abb binary
                i = 0
                for char in ent:
                    t = abb.find(char, i)
                    if t == -1:
                        abb_as_bin.append(0)
                    else:
                        i = t + 1
                        abb_as_bin.append(1)

                # Set up ent chars and words
                for word in ent_seg:
                    if word in self.w2v.wv:
                        word_idx = self.w2v.word2idx[word] + 1
                    else:
                        word_idx = 0
                    for char in word:
                        if char in self.c2v.wv:
                            char_idx = self.c2v.char2idx[char] + 1
                        else:
                            char_idx = 0
                        ent_as_char.append(char_idx)
                        ent_as_word.append(word_idx)

                assert len(ent_as_word) == len(abb_as_bin)

                ent_as_char_list.append(ent_as_char)
                ent_as_word_list.append(ent_as_word)
                abb_as_bin_list.append(abb_as_bin)

        char_data = sequence.pad_sequences(ent_as_char_list, maxlen=MAX_ABB_CHAR_LEN,
                                           padding="post", truncating="post", value=0)
        word_data = sequence.pad_sequences(ent_as_word_list, maxlen=MAX_ABB_CHAR_LEN,
                                           padding="post", truncating="post", value=0)
        abb_label = sequence.pad_sequences(abb_as_bin_list, maxlen=MAX_ABB_CHAR_LEN,
                                           padding="post", truncating="post", value=0)

        abb_label = abb_label.reshape(list(abb_label.shape) + [1])

        logging.info('Shape of char data tensor: {}'.format(char_data.shape))
        logging.info('Shape of word data tensor: {}'.format(word_data.shape))
        logging.info('Shape of abb label tensor: {}'.format(abb_label.shape))

        num_data = char_data.shape[0]

        # Shuffle the data
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        char_data = char_data[indices]
        word_data = word_data[indices]
        abb_label = abb_label[indices]
        # labels = labels[indices].reshape(list(labels.shape) + [1])

        # Split the data into a training set, a dev set and a test set
        logging.info("Splitting data...")
        num_dev_data = int(DEV_SPLIT * num_data)
        num_test_data = int(TEST_SPLIT * num_data)

        self.c_test = char_data[:num_test_data]
        self.c_dev = char_data[-num_dev_data:]
        self.c_train = char_data[num_test_data:-num_dev_data]

        self.w_test = word_data[:num_test_data]
        self.w_dev = word_data[-num_dev_data:]
        self.w_train = word_data[num_test_data:-num_dev_data]

        self.y_test = abb_label[:num_test_data]
        self.y_dev = abb_label[-num_dev_data:]
        self.y_train = abb_label[num_test_data:-num_dev_data]

    def _build(self):
        # Char embedding layer weight setting
        char_emb_weights = np.zeros((self.c2v.vocab_size + 1, self.c2v.vector_size))
        char_emb_weights[0, :] = np.zeros((self.c2v.vector_size,))
        for index, char in self.c2v.idx2char.items():
            char_emb_weights[index + 1, :] = self.c2v.wv[char]

        # Word embedding layer weight setting
        word_emb_weights = np.zeros((self.w2v.vocab_size + 1, self.w2v.vector_size))
        word_emb_weights[0, :] = np.zeros((self.w2v.vector_size,))
        for index, word in self.w2v.idx2word.items():
            word_emb_weights[index + 1, :] = self.w2v.wv[word]

        char_input_layer = Input(shape=(MAX_ABB_CHAR_LEN,), name="char_input")
        word_input_layer = Input(shape=(MAX_ABB_CHAR_LEN,), name="word_input")

        char_emb_layer = Embedding(self.c2v.vocab_size + 1, self.c2v.vector_size,
                                   weights=[char_emb_weights], input_length=MAX_ABB_CHAR_LEN,
                                   trainable=False, name="char_emb", mask_zero=True)
        word_emb_layer = Embedding(self.w2v.vocab_size + 1, self.w2v.vector_size,
                                   weights=[word_emb_weights], input_length=MAX_ABB_CHAR_LEN,
                                   trainable=False, name="word_emb", mask_zero=True)

        char_rnn_layer = LSTM(RNN_DIM, use_bias=False,
                              return_sequences=True, name="char_rnn_layer")

        out_layer = TimeDistributed(Dense(1, activation="sigmoid", use_bias=False, name="out_layer"))

        char_emb = char_emb_layer(char_input_layer)
        word_emb = word_emb_layer(word_input_layer)
        rnn_out = char_rnn_layer(concatenate([char_emb, word_emb]))
        out = out_layer(rnn_out)

        model = Model(inputs=[char_input_layer, word_input_layer], outputs=out)

        # compile model
        model.compile(loss=loss_function,
                      optimizer="sgd")

        return model

    def _train(self, char_data, word_data, abb_label):
        logging.info('Number of training data: {}'.format(char_data.shape[0]))
        self.model.fit({"char_input": char_data, "word_input": word_data},
                       abb_label, epochs=NUM_EPOCH, batch_size=BATCH_SIZE)

        logging.info("Saving model...")
        self.model.save_weights(MODEL_WEIGHT_FILE, overwrite=True)

    def evaluate(self, char_data, word_data, abb_label):
        results = self.model.predict({"char_input": char_data, "word_input": word_data},
                                     batch_size=BATCH_SIZE)
        correct_count = 0
        for idx, predict_probs in enumerate(results):
            true_label = abb_label[idx]
            correct = True
            for i, prob in enumerate(predict_probs):
                if (true_label[i] == 1 and prob < 0.5) or \
                        (true_label[i] == 0 and prob >= 0.5):
                    correct = False
                    break
            if correct:
                correct_count += 1
        return correct_count / len(results)

    def predict(self, ents):
        ent_as_char_list, ent_as_word_list = [], []

        for ent in ents:
            ent_as_char, ent_as_word = [], []

            ent_seg = jieba.cut(ent)

            # Set up ent chars and words
            for word in ent_seg:
                word_idx = self.w2v.word2idx[word] + 1
                for char in word:
                    char_idx = self.c2v.char2idx[char] + 1
                    ent_as_char.append(char_idx)
                    ent_as_word.append(word_idx)

            ent_as_char_list.append(ent_as_char)
            ent_as_word_list.append(ent_as_word)

        char_data = sequence.pad_sequences(ent_as_char_list, maxlen=MAX_ABB_CHAR_LEN,
                                           padding="post", truncating="post", value=0)
        word_data = sequence.pad_sequences(ent_as_word_list, maxlen=MAX_ABB_CHAR_LEN,
                                           padding="post", truncating="post", value=0)

        results = self.model.predict({"char_input": char_data, "word_input": word_data})

        abb_list = []
        for idx, predict_probs in enumerate(results):
            abb = ""
            for i, prob in enumerate(predict_probs):
                if prob > 0.5:
                    abb += ents[idx][i]
            abb_list.append(abb)
        return abb_list


def main():
    model = AbbRNN()

    # # Test the model
    # logging.info("Test...")
    # accuracy = model.evaluate(mode.c_test, model.w_test, model.y_test)
    # logging.info("Accuracy of the model is {}".format(accuracy))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    main()
