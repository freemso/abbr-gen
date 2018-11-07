# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import pickle
from pathlib import Path

import numpy as np
from keras.engine import Input, Model
from keras.layers import Embedding, LSTM, K, TimeDistributed, Dense, add
from keras.preprocessing import sequence

import embedding
import radd

__author__ = "freemso"

RAW_DATA_FILE = "data/abb2ent/men2ent_v2_clean.data"

MODEL_WEIGHT_FILE = "model/abb4.1.model"
TEMP_DATA_FILE = "temp/abb4.data"

MAX_ABB_CHAR_LEN = 15
MIN_ABB_CHAR_LEN = 1

NUM_FEATURE = 9

RNN_DIM = 50

NUM_EPOCH = 10
BATCH_SIZE = 32

TEST_SPLIT = 0.2
DEV_SPLIT = 0.1


def loss_function(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true))


class AbbRNN:
    def __init__(self):
        self.c2v = embedding.CharVector()
        self.radd = radd.RADD()

        processed_data_path = Path(TEMP_DATA_FILE)
        if not processed_data_path.is_file():
            logging.info("Preprocessing data...")
            self._preprocess()
            logging.info("Storing preprocessed data...")
            with open(TEMP_DATA_FILE, "wb") as out_file:
                pickle.dump((self.c_test, self.c_dev, self.c_train,
                             self.f_test, self.f_dev, self.f_train,
                             self.y_test, self.y_dev, self.y_train), out_file)
        else:
            logging.info("Loading preprocessed data...")
            with open(TEMP_DATA_FILE, "rb") as in_file:
                self.c_test, self.c_dev, self.c_train, self.f_test, self.f_dev, self.f_train, \
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
            self._train()

        # Evaluate the model
        logging.info("Evaluating...")
        accuracy = self.evaluate(self.c_dev, self.f_dev, self.y_dev)
        logging.info("Accuracy of the model is {}".format(accuracy))

    def _preprocess(self):
        # Load data
        logging.info("Loading raw data...")
        ent_as_char_list = []
        features_list = []
        abb_as_bin_list = []
        # count = 0
        with open(RAW_DATA_FILE) as in_file:
            for line in in_file:
                ent_as_char, features, abb_as_bin = [], [], []

                abb, ent = line.strip().split("\t")

                i = 0
                for char in ent:
                    # Set up ent chars
                    if char in self.c2v.wv:
                        char_idx = self.c2v.char2idx[char] + 1
                    else:
                        char_idx = 0
                    ent_as_char.append(char_idx)
                    # Set up abb binary
                    t = abb.find(char, i)
                    if t == -1:
                        abb_as_bin.append(0)
                    else:
                        i = t + 1
                        abb_as_bin.append(1)

                # Set up feature vector
                features = self.generate_feature_grams(ent)

                ent_as_char_list.append(ent_as_char)
                features_list.append(features)
                abb_as_bin_list.append(abb_as_bin)

        char_data = sequence.pad_sequences(ent_as_char_list, maxlen=MAX_ABB_CHAR_LEN,
                                           padding="post", truncating="post", value=0)
        feature_data = sequence.pad_sequences(features_list, maxlen=MAX_ABB_CHAR_LEN,
                                              padding="post", truncating="post", value=0)
        abb_label = sequence.pad_sequences(abb_as_bin_list, maxlen=MAX_ABB_CHAR_LEN,
                                           padding="post", truncating="post", value=0)

        abb_label = abb_label.reshape(list(abb_label.shape) + [1])

        logging.info('Shape of char data tensor: {}'.format(char_data.shape))
        logging.info('Shape of word data tensor: {}'.format(feature_data.shape))
        logging.info('Shape of abb label tensor: {}'.format(abb_label.shape))

        num_data = char_data.shape[0]

        # Shuffle the data
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        char_data = char_data[indices]
        feature_data = feature_data[indices]
        abb_label = abb_label[indices]
        # labels = labels[indices].reshape(list(labels.shape) + [1])

        # Split the data into a training set, a dev set and a test set
        logging.info("Splitting data...")
        num_dev_data = int(DEV_SPLIT * num_data)
        num_test_data = int(TEST_SPLIT * num_data)

        self.c_test = char_data[:num_test_data]
        self.c_dev = char_data[-num_dev_data:]
        self.c_train = char_data[num_test_data:-num_dev_data]

        self.f_test = feature_data[:num_test_data]
        self.f_dev = feature_data[-num_dev_data:]
        self.f_train = feature_data[num_test_data:-num_dev_data]

        self.y_test = abb_label[:num_test_data]
        self.y_dev = abb_label[-num_dev_data:]
        self.y_train = abb_label[num_test_data:-num_dev_data]

    def generate_feature_grams(self, ent):
        fg = np.zeros([len(ent), NUM_FEATURE, 1])
        for idx, char in enumerate(ent):
            # 2-gram
            if idx >= 2:
                fg[idx][0] = self.radd.predict(ent[idx-2:idx])
            if idx >= 1:
                fg[idx][1] = self.radd.predict(ent[idx-1:idx+1])
            if idx < len(ent) - 1:
                fg[idx][2] = self.radd.predict(ent[idx:idx+2])
            if idx < len(ent) - 2:
                fg[idx][3] = self.radd.predict(ent[idx+1:idx+3])
            # 3-gram
            if idx >= 2:
                fg[idx][4] = self.radd.predict(ent[idx-2:idx+1])
            if 1 <= idx < len(ent) - 1:
                fg[idx][5] = self.radd.predict(ent[idx-1:idx+2])
            if idx < len(ent) - 2:
                fg[idx][6] = self.radd.predict(ent[idx:idx+3])
            # 4-gram
            if 2 <= idx < len(ent) - 1:
                fg[idx][7] = self.radd.predict(ent[idx-2:idx+2])
            if 1 <= idx < len(ent) - 2:
                fg[idx][8] = self.radd.predict(ent[idx-1:idx+3])

        return fg

    def _build(self):
        # Char embedding layer weight setting
        char_emb_weights = np.zeros((self.c2v.vocab_size + 1, self.c2v.vector_size))
        char_emb_weights[0, :] = np.zeros((self.c2v.vector_size,))
        for index, char in self.c2v.idx2char.items():
            char_emb_weights[index + 1, :] = self.c2v.wv[char]

        char_input_layer = Input(shape=(MAX_ABB_CHAR_LEN,), name="char_input")
        feature_input_layer = Input(shape=(MAX_ABB_CHAR_LEN,), name="feature_input")

        char_emb_layer = Embedding(self.c2v.vocab_size + 1, self.c2v.vector_size,
                                   weights=[char_emb_weights], input_length=MAX_ABB_CHAR_LEN,
                                   trainable=False, name="char_emb", mask_zero=True)

        char_rnn_layer = LSTM(RNN_DIM, use_bias=False,
                              return_sequences=True, name="char_rnn_layer")

        out_layer = TimeDistributed(Dense(1, activation="sigmoid", use_bias=False, name="out_layer"))

        char_emb = char_emb_layer(char_input_layer)
        rnn_out = char_rnn_layer(add([char_emb, feature_input_layer]))
        out = out_layer(add([rnn_out, feature_input_layer]))

        model = Model(inputs=[char_input_layer, feature_input_layer], outputs=out)

        # compile model
        model.compile(loss=loss_function,
                      optimizer="sgd")

        return model

    def _train(self):
        logging.info('Number of training data: {}'.format(self.c_train.shape[0]))
        self.model.fit({"char_input": self.c_train, "word_input": self.f_train},
                       self.y_train, epochs=NUM_EPOCH, batch_size=BATCH_SIZE)

        logging.info("Saving model...")
        self.model.save_weights(MODEL_WEIGHT_FILE, overwrite=True)

    def evaluate(self, c, f, y):
        results = self.model.predict({"char_input": c, "word_input": f},
                                     batch_size=BATCH_SIZE)
        correct_count = 0
        for idx, predict_probs in enumerate(results):
            true_label = y[idx]
            correct = True
            for i, prob in enumerate(predict_probs):
                if (true_label[i] == 1 and prob < 0.5) or \
                        (true_label[i] == 0 and prob >= 0.5):
                    correct = False
                    break
            if correct:
                correct_count += 1
        return correct_count / len(results)

    # def predict(self, ents):
    #     ent_as_char_list, ent_as_word_list = [], []
    #
    #     for ent in ents:
    #         ent_as_char, features = [], []
    #
    #         # Set up ent chars and words
    #         for char in ent:
    #             char_idx = self.c2v.char2idx[char] + 1
    #             ent_as_char.append(char_idx)
    #
    #         ent_as_char_list.append(ent_as_char)
    #         ent_as_word_list.append(ent_as_word)
    #
    #     char_data = sequence.pad_sequences(ent_as_char_list, maxlen=MAX_ABB_CHAR_LEN,
    #                                        padding="post", truncating="post", value=0)
    #     word_data = sequence.pad_sequences(ent_as_word_list, maxlen=MAX_ABB_CHAR_LEN,
    #                                        padding="post", truncating="post", value=0)
    #
    #     results = self.model.predict({"char_input": char_data, "word_input": word_data})
    #
    #     abb_list = []
    #     for idx, predict_probs in enumerate(results):
    #         abb = ""
    #         for i, prob in enumerate(predict_probs):
    #             if prob > 0.5:
    #                 abb += ents[idx][i]
    #         abb_list.append(abb)
    #     return abb_list


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
