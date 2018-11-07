# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    
"""
import logging

import sys

import models
import dataset

__author__ = "freemso"


def log_result(data, preds, out_filename):
    ents = data["ent"]
    abbs_true_list = data["abbs"]
    with open(out_filename, "w") as out_fd:
        for ent, abb_true_list, abb_pred in zip(ents, abbs_true_list, preds):
            out_fd.write("{}\t{}\t{}\n".format(ent, abb_true_list, abb_pred))


def main(args):
    data_train, data_eval, concept2idx, tag2idx, word2concept = dataset.load_data()
    m = args[0]
    model_map = {
        "baseline": models.Baseline,
        "word": models.Word,
        "tag": models.Tag,
        "concept": models.Concept,
        "tag_word": models.TagWord,
        "concept_tag": models.ConceptTag,
        "concept_word": models.ConceptWord,
        "concept_tag_word": models.ConceptTagWord
    }

    if m in model_map:
        lstm_dim, hidden_dim, epoch, batch, early_stop, patience = args[1:]
        model = model_map[m](
            concept2idx=concept2idx,
            tag2idx=tag2idx,
            word2concept=word2concept,
            lstm_dim=int(lstm_dim),
            hidden_dim=int(hidden_dim)
        )
        model.train(data_train, int(epoch), int(batch), early_stop == "1", int(patience))
        accuracy, preds = model.evaluate(data_eval)
        logging.info("Accuracy: {}".format(accuracy))
        log_result(data_eval, preds, "out/{}_{}_{}_{}.txt".format(m, lstm_dim, hidden_dim, accuracy))

if __name__ == '__main__':
    main(sys.argv[1:])
