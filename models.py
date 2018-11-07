# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    
"""
import numpy as np
from keras import Input, losses
from keras.callbacks import EarlyStopping
from keras.engine import Model
from keras.layers import LSTM, Bidirectional, TimeDistributed, Dense, K, concatenate, Lambda

import dataset

__author__ = "freemso"


class ConceptTagWord(object):
    def __init__(self,
                 concept2idx,
                 tag2idx,
                 word2concept,
                 lstm_dim=50,
                 hidden_dim=50):
        self.concept2idx = concept2idx,
        self.tag2idx = tag2idx,
        self.word2concept = word2concept

        c_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="char_input")
        con_i = Input(shape=(dataset.MAX_SEQ_LEN,), dtype="int32")
        t_i = Input(shape=(dataset.MAX_SEQ_LEN,), dtype="int32")
        w_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="word_input")

        c_emb = dataset.get_char_emb_layer()(c_i)
        con_emb = Lambda(
            K.one_hot,
            arguments={'num_classes': len(concept2idx) + 2},
            output_shape=(dataset.MAX_SEQ_LEN, len(concept2idx) + 2)
        )(con_i)
        t_emb = Lambda(
            K.one_hot,
            arguments={'num_classes': len(tag2idx) + 2},
            output_shape=(dataset.MAX_SEQ_LEN, len(tag2idx) + 2)
        )(t_i)
        w_emb = dataset.get_word_emb_layer()(w_i)

        x = Bidirectional(LSTM(lstm_dim, return_sequences=True, name="lstm"))(
            concatenate([c_emb, con_emb, t_emb, w_emb])
        )
        x = TimeDistributed(Dense(hidden_dim, activation="sigmoid"))(x)

        o = TimeDistributed(Dense(1, activation="sigmoid", name="output"))(x)

        self.model = Model(inputs=[c_i, con_i, t_i, w_i], outputs=o)
        self.model.compile(
            loss=losses.binary_crossentropy,
            optimizer="adam",
            metrics=[seq_accuracy, "accuracy"]
        )

    def train(self, data_train, epoch, batch, early_stop=True, patience=5):
        x_char = data_train["x_char"]
        x_word = data_train["x_word"]
        x_concept = data_train["x_concept"]
        x_tag = data_train["x_tag"]
        y = data_train["label"]

        early_stopping_callback = EarlyStopping(
            monitor="val_seq_accuracy",
            patience=patience,
            verbose=1,
            mode="auto"
        )

        self.model.fit(
            [x_char, x_concept, x_tag, x_word],
            y,
            epochs=epoch,
            batch_size=batch,
            validation_split=dataset.VALIDATE_SPLIT_RATE,
            callbacks=[early_stopping_callback] if early_stop else None
        )

    def predict(self, entities):
        x_char, x_word, x_tag, x_concept = dataset.get_x_for_eval(
            entities, self.concept2idx, self.tag2idx, self.word2concept)
        y_preds = self.model.predict([x_char, x_concept, x_tag, x_word])
        return [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]

    def evaluate(self, data_eval):
        entities = data_eval["ent"]
        abbs_true_list = data_eval["abbs"]
        y_preds = self.model.predict(
            [data_eval["x_char"], data_eval["x_concept"], data_eval["x_tag"], data_eval["x_word"]]
        )
        abbs_pred = [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]
        correct = sum([
            1 if abb_pred in abb_true_list else 0
            for abb_pred, abb_true_list
            in zip(abbs_pred, abbs_true_list)
        ])
        return correct / len(entities), abbs_pred


class ConceptTag(object):
    def __init__(self,
                 concept2idx,
                 tag2idx,
                 word2concept,
                 lstm_dim=50,
                 hidden_dim=50):
        self.concept2idx = concept2idx,
        self.tag2idx = tag2idx,
        self.word2concept = word2concept

        c_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="char_input")
        con_i = Input(shape=(dataset.MAX_SEQ_LEN,), dtype="int32")
        t_i = Input(shape=(dataset.MAX_SEQ_LEN,), dtype="int32")

        c_emb = dataset.get_char_emb_layer()(c_i)
        con_emb = Lambda(
            K.one_hot,
            arguments={'num_classes': len(concept2idx) + 2},
            output_shape=(dataset.MAX_SEQ_LEN, len(concept2idx) + 2)
        )(con_i)
        t_emb = Lambda(
            K.one_hot,
            arguments={'num_classes': len(tag2idx) + 2},
            output_shape=(dataset.MAX_SEQ_LEN, len(tag2idx) + 2)
        )(t_i)

        x = Bidirectional(LSTM(lstm_dim, return_sequences=True, name="lstm"))(
            concatenate([c_emb, con_emb, t_emb])
        )
        x = TimeDistributed(Dense(hidden_dim, activation="sigmoid"))(x)

        o = TimeDistributed(Dense(1, activation="sigmoid", name="output"))(x)

        self.model = Model(inputs=[c_i, con_i, t_i], outputs=o)
        self.model.compile(
            loss=losses.binary_crossentropy,
            optimizer="adam",
            metrics=[seq_accuracy, "accuracy"]
        )

    def train(self, data_train, epoch, batch, early_stop=True, patience=5):
        x_char = data_train["x_char"]
        x_concept = data_train["x_concept"]
        x_tag = data_train["x_tag"]
        y = data_train["label"]

        early_stopping_callback = EarlyStopping(
            monitor="val_seq_accuracy",
            patience=patience,
            verbose=1,
            mode="auto"
        )

        self.model.fit(
            [x_char, x_concept, x_tag],
            y,
            epochs=epoch,
            batch_size=batch,
            validation_split=dataset.VALIDATE_SPLIT_RATE,
            callbacks=[early_stopping_callback] if early_stop else None
        )

    def predict(self, entities):
        x_char, _, x_tag, x_concept = dataset.get_x_for_eval(
            entities, self.concept2idx, self.tag2idx, self.word2concept)
        y_preds = self.model.predict([x_char, x_concept, x_tag])
        return [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]

    def evaluate(self, data_eval):
        entities = data_eval["ent"]
        abbs_true_list = data_eval["abbs"]
        y_preds = self.model.predict(
            [data_eval["x_char"], data_eval["x_concept"], data_eval["x_tag"]]
        )
        abbs_pred = [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]
        correct = sum([
            1 if abb_pred in abb_true_list else 0
            for abb_pred, abb_true_list
            in zip(abbs_pred, abbs_true_list)
        ])
        return correct / len(entities), abbs_pred


class ConceptWord(object):
    def __init__(self,
                 concept2idx,
                 tag2idx,
                 word2concept,
                 lstm_dim=50,
                 hidden_dim=50):
        self.concept2idx = concept2idx,
        self.tag2idx = tag2idx,
        self.word2concept = word2concept

        c_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="char_input")
        con_i = Input(shape=(dataset.MAX_SEQ_LEN,), dtype="int32")
        w_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="word_input")

        c_emb = dataset.get_char_emb_layer()(c_i)
        con_emb = Lambda(
            K.one_hot,
            arguments={'num_classes': len(concept2idx) + 2},
            output_shape=(dataset.MAX_SEQ_LEN, len(concept2idx) + 2)
        )(con_i)
        w_emb = dataset.get_word_emb_layer()(w_i)

        x = Bidirectional(LSTM(lstm_dim, return_sequences=True, name="lstm"))(
            concatenate([c_emb, con_emb, w_emb])
        )
        x = TimeDistributed(Dense(hidden_dim, activation="sigmoid"))(x)

        o = TimeDistributed(Dense(1, activation="sigmoid", name="output"))(x)

        self.model = Model(inputs=[c_i, con_i, w_i], outputs=o)
        self.model.compile(
            loss=losses.binary_crossentropy,
            optimizer="adam",
            metrics=[seq_accuracy, "accuracy"]
        )

    def train(self, data_train, epoch, batch, early_stop=True, patience=5):
        x_char = data_train["x_char"]
        x_word = data_train["x_word"]
        x_concept = data_train["x_concept"]
        y = data_train["label"]

        early_stopping_callback = EarlyStopping(
            monitor="val_seq_accuracy",
            patience=patience,
            verbose=1,
            mode="auto"
        )

        self.model.fit(
            [x_char, x_concept, x_word],
            y,
            epochs=epoch,
            batch_size=batch,
            validation_split=dataset.VALIDATE_SPLIT_RATE,
            callbacks=[early_stopping_callback] if early_stop else None
        )

    def predict(self, entities):
        x_char, x_word, _, x_concept = dataset.get_x_for_eval(
            entities, self.concept2idx, self.tag2idx, self.word2concept)
        y_preds = self.model.predict([x_char, x_concept, x_word])
        return [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]

    def evaluate(self, data_eval):
        entities = data_eval["ent"]
        abbs_true_list = data_eval["abbs"]
        y_preds = self.model.predict(
            [data_eval["x_char"], data_eval["x_concept"], data_eval["x_word"]]
        )
        abbs_pred = [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]
        correct = sum([
            1 if abb_pred in abb_true_list else 0
            for abb_pred, abb_true_list
            in zip(abbs_pred, abbs_true_list)
        ])
        return correct / len(entities), abbs_pred


class TagWord(object):
    def __init__(self,
                 concept2idx,
                 tag2idx,
                 word2concept,
                 lstm_dim=50,
                 hidden_dim=50):
        self.concept2idx = concept2idx,
        self.tag2idx = tag2idx,
        self.word2concept = word2concept

        c_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="char_input")
        t_i = Input(shape=(dataset.MAX_SEQ_LEN,), dtype="int32")
        w_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="word_input")

        c_emb = dataset.get_char_emb_layer()(c_i)
        t_emb = Lambda(
            K.one_hot,
            arguments={'num_classes': len(tag2idx) + 2},
            output_shape=(dataset.MAX_SEQ_LEN, len(tag2idx) + 2)
        )(t_i)
        w_emb = dataset.get_word_emb_layer()(w_i)

        x = Bidirectional(LSTM(lstm_dim, return_sequences=True, name="lstm"))(
            concatenate([c_emb, t_emb, w_emb])
        )
        x = TimeDistributed(Dense(hidden_dim, activation="sigmoid"))(x)

        o = TimeDistributed(Dense(1, activation="sigmoid", name="output"))(x)

        self.model = Model(inputs=[c_i, t_i, w_i], outputs=o)
        self.model.compile(
            loss=losses.binary_crossentropy,
            optimizer="adam",
            metrics=[seq_accuracy, "accuracy"]
        )

    def train(self, data_train, epoch, batch, early_stop=True, patience=5):
        x_char = data_train["x_char"]
        x_word = data_train["x_word"]
        x_tag = data_train["x_tag"]
        y = data_train["label"]

        early_stopping_callback = EarlyStopping(
            monitor="val_seq_accuracy",
            patience=patience,
            verbose=1,
            mode="auto"
        )

        self.model.fit(
            [x_char, x_tag, x_word],
            y,
            epochs=epoch,
            batch_size=batch,
            validation_split=dataset.VALIDATE_SPLIT_RATE,
            callbacks=[early_stopping_callback] if early_stop else None
        )

    def predict(self, entities):
        x_char, x_word, x_tag, _ = dataset.get_x_for_eval(
            entities, self.concept2idx, self.tag2idx, self.word2concept)
        y_preds = self.model.predict([x_char, x_tag, x_word])
        return [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]

    def evaluate(self, data_eval):
        entities = data_eval["ent"]
        abbs_true_list = data_eval["abbs"]
        y_preds = self.model.predict(
            [data_eval["x_char"], data_eval["x_tag"], data_eval["x_word"]]
        )
        abbs_pred = [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]
        correct = sum([
            1 if abb_pred in abb_true_list else 0
            for abb_pred, abb_true_list
            in zip(abbs_pred, abbs_true_list)
        ])
        return correct / len(entities), abbs_pred


class Concept(object):
    def __init__(self,
                 concept2idx,
                 tag2idx,
                 word2concept,
                 lstm_dim=50,
                 hidden_dim=50):
        self.concept2idx = concept2idx,
        self.tag2idx = tag2idx,
        self.word2concept = word2concept

        c_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="char_input")
        con_i = Input(shape=(dataset.MAX_SEQ_LEN,), dtype="int32")

        c_emb = dataset.get_char_emb_layer()(c_i)
        con_emb = Lambda(
            K.one_hot,
            arguments={'num_classes': len(concept2idx) + 2},
            output_shape=(dataset.MAX_SEQ_LEN, len(concept2idx) + 2)
        )(con_i)

        x = Bidirectional(LSTM(lstm_dim, return_sequences=True, name="lstm"))(
            concatenate([c_emb, con_emb])
        )
        x = TimeDistributed(Dense(hidden_dim, activation="sigmoid"))(x)

        o = TimeDistributed(Dense(1, activation="sigmoid", name="output"))(x)

        self.model = Model(inputs=[c_i, con_i], outputs=o)
        self.model.compile(
            loss=losses.binary_crossentropy,
            optimizer="adam",
            metrics=[seq_accuracy, "accuracy"]
        )

    def train(self, data_train, epoch, batch, early_stop=True, patience=5):
        x_char = data_train["x_char"]
        x_concept = data_train["x_concept"]
        y = data_train["label"]

        early_stopping_callback = EarlyStopping(
            monitor="val_seq_accuracy",
            patience=patience,
            verbose=1,
            mode="auto"
        )

        self.model.fit(
            [x_char, x_concept],
            y,
            epochs=epoch,
            batch_size=batch,
            validation_split=dataset.VALIDATE_SPLIT_RATE,
            callbacks=[early_stopping_callback] if early_stop else None
        )

    def predict(self, entities):
        x_char, _, _, x_concept = dataset.get_x_for_eval(
            entities, self.concept2idx, self.tag2idx, self.word2concept)
        y_preds = self.model.predict([x_char, x_concept])
        return [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]

    def evaluate(self, data_eval):
        entities = data_eval["ent"]
        abbs_true_list = data_eval["abbs"]
        y_preds = self.model.predict(
            [data_eval["x_char"], data_eval["x_concept"]]
        )
        abbs_pred = [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]
        correct = sum([
            1 if abb_pred in abb_true_list else 0
            for abb_pred, abb_true_list
            in zip(abbs_pred, abbs_true_list)
        ])
        return correct / len(entities), abbs_pred


class Tag(object):
    def __init__(self,
                 concept2idx,
                 tag2idx,
                 word2concept,
                 lstm_dim=50,
                 hidden_dim=50):
        self.concept2idx = concept2idx,
        self.tag2idx = tag2idx,
        self.word2concept = word2concept

        c_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="char_input")
        t_i = Input(shape=(dataset.MAX_SEQ_LEN,), dtype="int32")

        c_emb = dataset.get_char_emb_layer()(c_i)
        t_emb = Lambda(
            K.one_hot,
            arguments={'num_classes': len(tag2idx) + 2},
            output_shape=(dataset.MAX_SEQ_LEN, len(tag2idx) + 2)
        )(t_i)

        x = Bidirectional(LSTM(lstm_dim, return_sequences=True, name="lstm"))(
            concatenate([c_emb, t_emb])
        )
        x = TimeDistributed(Dense(hidden_dim, activation="sigmoid"))(x)

        o = TimeDistributed(Dense(1, activation="sigmoid", name="output"))(x)

        self.model = Model(inputs=[c_i, t_i], outputs=o)
        self.model.compile(
            loss=losses.binary_crossentropy,
            optimizer="adam",
            metrics=[seq_accuracy, "accuracy"]
        )

    def train(self, data_train, epoch, batch, early_stop=True, patience=5):
        x_char = data_train["x_char"]
        x_tag = data_train["x_tag"]
        y = data_train["label"]

        early_stopping_callback = EarlyStopping(
            monitor="val_seq_accuracy",
            patience=patience,
            verbose=1,
            mode="auto"
        )

        self.model.fit(
            [x_char, x_tag],
            y,
            epochs=epoch,
            batch_size=batch,
            validation_split=dataset.VALIDATE_SPLIT_RATE,
            callbacks=[early_stopping_callback] if early_stop else None
        )

    def predict(self, entities):
        x_char, _, x_tag, _ = dataset.get_x_for_eval(
            entities, self.concept2idx, self.tag2idx, self.word2concept)
        y_preds = self.model.predict([x_char, x_tag])
        return [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]

    def evaluate(self, data_eval):
        entities = data_eval["ent"]
        abbs_true_list = data_eval["abbs"]
        y_preds = self.model.predict(
            [data_eval["x_char"], data_eval["x_tag"]]
        )
        abbs_pred = [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]
        correct = sum([
            1 if abb_pred in abb_true_list else 0
            for abb_pred, abb_true_list
            in zip(abbs_pred, abbs_true_list)
        ])
        return correct / len(entities), abbs_pred


class Word(object):
    def __init__(self,
                 concept2idx,
                 tag2idx,
                 word2concept,
                 lstm_dim=50,
                 hidden_dim=50):
        self.concept2idx = concept2idx,
        self.tag2idx = tag2idx,
        self.word2concept = word2concept

        c_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="char_input")
        w_i = Input(shape=(dataset.MAX_SEQ_LEN,), name="word_input")

        c_emb = dataset.get_char_emb_layer()(c_i)
        w_emb = dataset.get_word_emb_layer()(w_i)

        x = Bidirectional(LSTM(lstm_dim, return_sequences=True, name="lstm"))(
            concatenate([c_emb, w_emb])
        )
        x = TimeDistributed(Dense(hidden_dim, activation="sigmoid"))(x)

        o = TimeDistributed(Dense(1, activation="sigmoid", name="output"))(x)

        self.model = Model(inputs=[c_i, w_i], outputs=o)
        self.model.compile(
            loss=losses.binary_crossentropy,
            optimizer="adam",
            metrics=[seq_accuracy, "accuracy"]
        )

    def train(self, data_train, epoch, batch, early_stop=True, patience=5):
        x_char = data_train["x_char"]
        x_word = data_train["x_word"]
        y = data_train["label"]

        early_stopping_callback = EarlyStopping(
            monitor="val_seq_accuracy",
            patience=patience,
            verbose=1,
            mode="auto"
        )

        self.model.fit(
            [x_char, x_word],
            y,
            epochs=epoch,
            batch_size=batch,
            validation_split=dataset.VALIDATE_SPLIT_RATE,
            callbacks=[early_stopping_callback] if early_stop else None
        )

    def predict(self, entities):
        x_char, x_word, _, _ = dataset.get_x_for_eval(
            entities, self.concept2idx, self.tag2idx, self.word2concept)
        y_preds = self.model.predict([x_char, x_word])
        return [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]

    def evaluate(self, data_eval):
        entities = data_eval["ent"]
        abbs_true_list = data_eval["abbs"]
        y_preds = self.model.predict(
            [data_eval["x_char"], data_eval["x_word"]]
        )
        abbs_pred = [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]
        correct = sum([
            1 if abb_pred in abb_true_list else 0
            for abb_pred, abb_true_list
            in zip(abbs_pred, abbs_true_list)
        ])
        return correct / len(entities), abbs_pred


class Baseline(object):
    """
    Simple RNN(Bi-LSTM) model that take a sequence of character as input.
    Serve as baseline
    """

    def __init__(self,
                 concept2idx,
                 tag2idx,
                 word2concept,
                 lstm_dim=50,
                 hidden_dim=50):
        self.concept2idx = concept2idx,
        self.tag2idx = tag2idx,
        self.word2concept = word2concept

        i = Input(shape=(dataset.MAX_SEQ_LEN,), name="input")
        x = dataset.get_char_emb_layer()(i)
        x = Bidirectional(LSTM(lstm_dim, return_sequences=True, name="lstm"))(x)
        x = TimeDistributed(Dense(hidden_dim, activation="sigmoid"))(x)
        o = TimeDistributed(Dense(1, activation="sigmoid", name="output"))(x)
        self.model = Model(inputs=i, outputs=o)
        self.model.compile(
            loss=losses.binary_crossentropy,
            optimizer="adam",
            metrics=[seq_accuracy, "accuracy"]
        )

    def train(self, data_train, epoch, batch, early_stop=True, patience=5):
        x_char = data_train["x_char"]
        y = data_train["label"]

        early_stopping_callback = EarlyStopping(
            monitor="val_seq_accuracy",
            patience=patience,
            verbose=1,
            mode="auto"
        )

        self.model.fit(
            x_char,
            y,
            epochs=epoch,
            batch_size=batch,
            validation_split=dataset.VALIDATE_SPLIT_RATE,
            callbacks=[early_stopping_callback] if early_stop else None
        )

    def predict(self, entities):
        x_char, _, _, _ = dataset.get_x_for_eval(entities, self.concept2idx, self.tag2idx, self.word2concept)
        y_preds = self.model.predict(x_char)
        return [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]

    def evaluate(self, data_eval):
        entities = data_eval["ent"]
        abbs_true_list = data_eval["abbs"]
        y_preds = self.model.predict(data_eval["x_char"])
        abbs_pred = [
            "".join([c if l == 1 else "" for c, l in zip(ent, np.round(y_pred))])
            for ent, y_pred in zip(entities, y_preds)
        ]
        correct = sum([
            1 if abb_pred in abb_true_list else 0
            for abb_pred, abb_true_list
            in zip(abbs_pred, abbs_true_list)
        ])
        return correct / len(entities), abbs_pred


def seq_accuracy(y_true, y_pred):
    return K.cast(K.all(K.equal(y_true, K.round(y_pred)), axis=-2), 'float32')
