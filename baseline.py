# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Character-level LSTM model.
    Serve as baseline
"""

import logging
import pathlib

import keras.preprocessing.sequence
import keras.engine
import keras.layers
import numpy

import embedding
import util.decorator
import util.terminal

__author__ = "freemso"

RAW_DATA_FILE = "data/abb2ent/valid.data"
MODEL_FILE = "model/baseline.model"

CHAR_LEN = 20

TEST_SPLIT_RATE = 0.1
VALIDATION_RATE = 0.05

UNK_IDX = 1
PAD_IDX = 0


def preprocess():
    logging.info("Loading raw data...")
    ent_idx_lists = []
    abb_bin_lists = []

    with open(RAW_DATA_FILE, "r") as in_file:
        for line in in_file:
            ent_idx_list, abb_bin_list = parse_ent_abb(line)
            ent_idx_lists.append(ent_idx_list)
            abb_bin_lists.append(abb_bin_list)

    x = keras.preprocessing.sequence.pad_sequences(ent_idx_lists, maxlen=CHAR_LEN,
                                                   padding="post", truncating="post", value=PAD_IDX)
    y = keras.preprocessing.sequence.pad_sequences(abb_bin_lists, maxlen=CHAR_LEN,
                                                   padding="post", truncating="post", value=0)

    return x, y


def train(model):
    x, y = preprocess()

    y = y.reshape(list(y.shape) + [1])

    logging.info("Shape of char x tensor: {}".format(x.shape))
    logging.info("Shape of abb y tensor: {}".format(y.shape))

    num_samples = x.shape[0]

    # Shuffle the data
    indices = numpy.arange(num_samples)
    numpy.random.shuffle(indices)
    x, y = x[indices], y[indices]

    # Split into train set and test set
    num_test_samples = int(TEST_SPLIT_RATE * num_samples)

    x_test, x_train = x[:num_test_samples], x[num_test_samples:]
    y_test, y_train = y[:num_test_samples], y[num_test_samples:]

    logging.info("Develop on {} samples, test on {} samples".format(x_train.shape[0], x_test.shape[0]))

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=0, verbose=1, mode="auto")

    model.fit(
        x_train,
        y_train,
        epochs=EPOCH,
        batch_size=BATCH,
        validation_split=VALIDATION_RATE,
        # callbacks=[early_stopping]
    )

    logging.info("Saving model...")
    model.save_weights(MODEL_FILE, overwrite=True)

    # Evaluate model
    predicts = model.predict(x_test, batch_size=BATCH)
    correct_count = 0
    for idx, y_pred in enumerate(predicts):
        y_true = y_test[idx]
        correct = True
        for ci, cp in enumerate(y_pred):
            if (y_true[ci] == 1 and cp < 0.5) \
                    or (y_true[ci] == 0 and cp > 0.5):
                correct = False
                break
        if correct:
            correct_count += 1

    accuracy = correct_count / x_test.shape[0]

    logging.info("Accuracy: {}".format(accuracy))


def predict(model, words):
    abbs = []

    char_idx_lists = []
    for word in words:
        char_idx_list = []
        for char in word:
            if char in char_emb.wv:
                idx = char_emb.char2idx[char] + 2
            else:
                idx = UNK_IDX
            char_idx_list.append(idx)
        char_idx_lists.append(char_idx_list)

    x = keras.preprocessing.sequence.pad_sequences(char_idx_lists,
                                                   CHAR_LEN,
                                                   padding="post",
                                                   truncating="post",
                                                   value=PAD_IDX)

    predicts = model.predict(x, batch_size=BATCH)

    for idx, y_pred in enumerate(predicts):
        word = words[idx]
        abb = ""
        for ci, char in enumerate(word):
            if y_pred[ci] > 0.5:
                abb += char
        abbs.append(abb)
    return abbs


def main():
    model = build_model()
    train(model)

    with open("data/result.txt", "w") as out_fd:
        ents = [line.strip().split("\t")[1] for line in open(RAW_DATA_FILE)]
        abbs = [line.strip().split("\t")[0] for line in open(RAW_DATA_FILE)]
        preds = predict(model, ents)
        for ent, abb, pred in zip(ents, abbs, preds):
            correct = abb == pred
            out_fd.write("{}\t{}\t{}\t{}\n".format(ent, abb, pred, correct))


if __name__ == "__main__":
    util.terminal.setup_logging_config(debug=False)
    main()
