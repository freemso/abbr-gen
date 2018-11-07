# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
from keras.engine import Input, Model
from keras.layers import Embedding, Dense, LSTM
from keras.preprocessing import sequence
from nltk import ngrams

import embedding
import util.preprocessing
from util.ProgressBar import ProgressBar

__author__ = "freemso"

MIN_RADD_WORD_LEN = 1
MAX_RADD_WORD_LEN = 10

RNN_DIM = 200
NUM_EPOCH = 10
BATCH_SIZE = 32

DEV_SPLIT = 0.2

MODEL_WEIGHT_FILE = "model/radd.model"
TEMP_DATA_FILE = "temp/radd.data"

SOUGO_DATA_FILE = "data/sougoCA/sougoCA_txt.data"
RAW_DATA_PATH = "data/sougoCA/sougoCA_xml.data"


class RADD:
    def __init__(self):
        # Char2Vec model
        self.cv = embedding.CharVector()

        self.model = self._build()

        model_weight_path = Path(MODEL_WEIGHT_FILE)
        if model_weight_path.is_file():
            logging.info("Loading RADD model weight...")
            self.model.load_weights(MODEL_WEIGHT_FILE)
        else:
            logging.info("Training RADD model...")

            train_data_path = Path(TEMP_DATA_FILE)
            if not train_data_path.is_file():
                logging.info("Preprocessing...")
                self._preprocess()
                logging.info("Storing preprocessed data...")
                with open(TEMP_DATA_FILE, "wb") as out_file:
                    pickle.dump((self.x_train, self.y_train, self.x_dev, self.y_dev), out_file)
            else:
                logging.info("Loading preprocessed data...")
                with open(TEMP_DATA_FILE, "rb") as in_file:
                    self.x_train, self.y_train, self.x_dev, self.y_dev = pickle.load(in_file)

            self._train()

            # Evaluate the model
            logging.info("Evaluating...")
            accuracy = self._evaluate()
            logging.info("Accuracy of the model is {}".format(accuracy))

    def _preprocess(self):

        word_set = self._load_data()
        logging.info("Loaded {} words".format(len(word_set)))

        logging.info("Generating negative examples...")
        non_word_set = generate_non_words(word_set)
        logging.info("Generated {} negative examples".format(len(non_word_set)))

        words = list(word_set) + list(non_word_set)
        labels = [1] * len(word_set) + [0] * len(non_word_set)

        # padding the words to the same length
        # and make them a ndarray
        num_data = len(words)
        data = np.zeros((num_data, MAX_RADD_WORD_LEN), dtype="float32")
        bar = ProgressBar(total=num_data)
        logging.info("Padding data...")
        for idx, word in enumerate(words):
            bar.log()
            seq = []
            for char in word:
                seq.append(self.cv.char2idx[char] + 1)
            trunc = seq[:MAX_RADD_WORD_LEN]
            data[idx, :len(trunc)] = trunc

        labels = np.array(labels)

        logging.info('Shape of data tensor: {}'.format(data.shape))
        logging.info('Shape of label tensor: {}'.format(labels.shape))

        # Shuffle the data
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        # Split the data into a training set, a dev set and a test set
        logging.info("Splitting data...")
        num_dev_data = int(DEV_SPLIT * num_data)

        self.x_train = data[:-num_dev_data]
        self.x_dev = data[-num_dev_data:]

        self.y_train = labels[:-num_dev_data]
        self.y_dev = labels[-num_dev_data:]

    def _load_data(self):
        """ load chinese word data
        merge data from different sources
        convert them into utf-8 & simplified chinese
        """
        word_set = set()

        logging.info("loading common words...")
        with open("data/chinese_words/common_words.data") as in_file:
            for line in in_file:
                word = line.split("\t")[0].strip()
                select_to_set(word_set, word, self.cv)
        # logging.info("loading company names...")
        # with open("data/chinese_words/company_names.data") as in_file:
        #     for line in in_file:
        #         word = line.strip()
        #         select_to_set(word_set, word, self.cv)
        # logging.info("loading people names...")
        # with open("data/chinese_words/people_names.data") as in_file:
        #     for line in in_file:
        #         word = line.strip()
        #         select_to_set(word_set, word, self.cv)
        logging.info("loading souhu dictionary...")
        with open("data/chinese_words/souhu_dict.data") as in_file:
            for line in in_file:
                word = line.strip()
                select_to_set(word_set, word, self.cv)
        logging.info("loading idiom dictionary...")
        with open("data/chinese_words/idiom_dict.data") as in_file:
            for line in in_file:
                word = line.split(" ")[0].strip()
                select_to_set(word_set, word, self.cv)
        logging.info("loading word dictionary...")
        with open("data/chinese_words/word_dict.data") as in_file:
            for line in in_file:
                word = line.split(" ")[0].strip()
                select_to_set(word_set, word, self.cv)

        return word_set

    def _build(self):
        # embedding layer weight setting
        embedding_weights = np.zeros((self.cv.vocab_size + 1, self.cv.vector_size))
        embedding_weights[0, :] = np.zeros((self.cv.vector_size,))
        for index, char in self.cv.idx2char.items():
            embedding_weights[index + 1, :] = self.cv.wv[char]

        # build model
        logging.info('building model...')
        i = Input(shape=(MAX_RADD_WORD_LEN,), name='char_input')

        c = Embedding(self.cv.vocab_size + 1, self.cv.vector_size,
                      weights=[embedding_weights], input_length=MAX_RADD_WORD_LEN,
                      trainable=False, name="emb", mask_zero=True)(i)

        s = LSTM(RNN_DIM, use_bias=False, return_sequences=False, name="lstm")(c)

        y = Dense(1, activation='sigmoid', use_bias=False, name="model_output")(s)

        model = Model(inputs=i, outputs=y)

        model.compile(loss="binary_crossentropy",
                      optimizer="sgd")

        return model

    def predict(self, word):
        word_vector = []
        for char in word:
            if char in self.cv.wv:
                word_vector.append(self.cv.char2idx[char] + 1)
            else:
                word_vector.append(0)
        data = sequence.pad_sequences(word_vector, maxlen=MAX_RADD_WORD_LEN,
                                      padding="post", truncating="post", value=0)
        pred = self.model.predict(data, batch_size=BATCH_SIZE)
        return int(pred[0] >= 0.5)

    def _train(self):
        self.model.fit(self.x_train, self.y_train, epochs=NUM_EPOCH, batch_size=BATCH_SIZE)

        # Save model weights
        self.model.save_weights(MODEL_WEIGHT_FILE, overwrite=True)

    def _evaluate(self):
        results = self.model.predict(self.x_dev, batch_size=BATCH_SIZE)
        correct_count = 0
        for idx, prob in enumerate(results):
            true_label = self.y_dev[idx]
            if (true_label == 0 and prob < 0.5) or \
                    (true_label == 1 and prob >= 0.5):
                correct_count += 1
        return correct_count / len(results)


def get_common_char(word_set):
    """ to compute a list of chinese char that most commonly char rank first"""
    count = Counter()
    for word in word_set:
        for char in word:
            count[char] += 1
    return count.most_common()


def generate_non_words(word_set):
    non_word_set = set()
    num_line = 0
    sougo_data_path = Path(SOUGO_DATA_FILE)
    if not sougo_data_path.is_file():
        load_sougo_data()
    with open(SOUGO_DATA_FILE) as in_file:
        for _ in in_file:
            num_line += 1
    bar = ProgressBar(total=num_line)
    with open(SOUGO_DATA_FILE) as in_file:
        for sentence in in_file:
            bar.log()
            sentence = sentence.strip().split(" ")
            if len(sentence) < 2:
                continue
            bigrams = list(ngrams(sentence, 2))
            trigrams = list(ngrams(sentence, 3))
            fourgrams = list(ngrams(sentence, 4))
            fivegrams = list(ngrams(sentence, 5))

            buf = bigrams + trigrams + fourgrams + fivegrams
            for gram in random.sample(buf, 1):
                gram = "".join(gram).strip()
                if gram not in word_set:
                    non_word_set.add(gram)
            buf.clear()
    return non_word_set


def select_to_set(word_set, word, cv):
    word = util.preprocessing.rm_paren_content(word)
    word = util.preprocessing.rm_non_chinese_char(word)
    if MIN_RADD_WORD_LEN < len(word) < MAX_RADD_WORD_LEN and all([c in cv.wv for c in word]):
        word_set.add(word.strip())


def load_sougo_data():
    logging.info("Loading sougo data...")
    # count = 0
    with open(RAW_DATA_PATH) as in_file, open(SOUGO_DATA_FILE, "w") as out_file:
        for line in in_file:
            if line.strip().startswith("<content>"):
                txt = line.strip("</content>")
                if txt is not None:
                    for sentence in util.preprocessing.to_sentences(txt):
                        sentence = " ".join([c for c in util.preprocessing.rm_non_chinese_char(sentence)])
                        out_file.write(sentence + "\n")


def main():
    model = RADD()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    main()
