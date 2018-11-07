# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import multiprocessing
import pathlib

import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import util.preprocessing

__author__ = "freemso"

SOUGO_DATA_FILE = "data/sougoCA/sougoCA_txt.data"
RAW_DATA_PATH = "data/sougoCA/sougoCA_xml.data"

CHAR_MODEL_PATH = "model/char2vec.model"
WORD_MODEL_PATH = "model/word2vec.model"

CHAR_VEC_DIM = 200
WORD_VEC_DIM = 200


class CharVector:
    def __init__(self):
        model_file_path = pathlib.Path(CHAR_MODEL_PATH)
        if model_file_path.is_file():
            # load model from disk file
            logging.info("Loading Char2Vec model from disk...")
            self._model = Word2Vec.load(CHAR_MODEL_PATH)
        else:
            logging.info("Training Char2Vec model...")
            sougo_file = pathlib.Path(SOUGO_DATA_FILE)
            if not sougo_file.is_file():
                self.load_sougo_data()
            self._model = Word2Vec(LineSentence(SOUGO_DATA_FILE), sg=1, size=CHAR_VEC_DIM,
                                   workers=multiprocessing.cpu_count(), min_count=0)
            # save model to disk
            self._model.save(CHAR_MODEL_PATH)

        self.wv = self._model.wv
        self.char2idx = dict((c, i) for i, c in enumerate(self._model.wv.index2word))
        self.idx2char = dict((i, c) for i, c in enumerate(self._model.wv.index2word))
        self.vocab_size = len(self._model.wv.index2word)
        self.vector_size = self._model.vector_size

    def load_sougo_data(self):
        logging.info("Loading sougo data...")
        # count = 0
        with open(RAW_DATA_PATH) as in_file, open(SOUGO_DATA_FILE, "w") as out_file:
            for line in in_file:
                # count += 1
                # if count > 1000:
                #     break
                if line.strip().startswith("<content>"):
                    txt = line.strip("</content>")
                    if txt is not None:
                        for sentence in util.preprocessing.to_sentences(txt):
                            sentence = " ".join([c for c in util.preprocessing.rm_non_chinese_char(sentence)])
                            out_file.write(sentence + "\n")


class WordVector:
    def __init__(self):
        model_file_path = pathlib.Path(WORD_MODEL_PATH)
        if model_file_path.is_file():
            # load model from disk file
            logging.info("Loading Word2Vec model from disk...")
            self._model = Word2Vec.load(WORD_MODEL_PATH)
        else:
            logging.info("Training Word2Vec model...")
            sougo_file = pathlib.Path(SOUGO_DATA_FILE)
            if not sougo_file.is_file():
                self.load_sougo_data()
            self._model = Word2Vec(LineSentence(SOUGO_DATA_FILE),
                                   sg=1, size=WORD_VEC_DIM, min_count=5,
                                   workers=multiprocessing.cpu_count())
            # save model to disk
            self._model.save(WORD_MODEL_PATH)

        self.wv = self._model.wv
        self.word2idx = dict((w, i) for i, w in enumerate(self._model.wv.index2word))
        self.idx2word = dict((i, w) for i, w in enumerate(self._model.wv.index2word))
        self.vocab_size = len(self._model.wv.index2word)
        self.vector_size = self._model.vector_size

    def load_sougo_data(self):
        logging.info("Loading sougo data...")
        # count = 0
        with open(RAW_DATA_PATH) as in_file, open(RAW_DATA_PATH, "w") as out_file:
            for line in in_file:
                # count += 1
                # if count > 1000:
                #     break
                if line.strip().startswith("<content>"):
                    txt = line.strip("</content>")
                    if txt is not None:
                        sent_seg = jieba.cut(txt)
                        line = " ".join([word for word in sent_seg])
                        out_file.write(line + "\n")
