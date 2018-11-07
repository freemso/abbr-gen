#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" to clean the men2ent_all.dat data set """


__author__ = "freemso"

import logging

from util import preprocessing

from util.terminal import ProgressBar


def is_substring(s1, s2):
    """
    check whether s1 is the substring of s2
        which means every character of s1 appears in
        s2 in the same order as in s1
        e.g. 'abc' is the substring of 'adbec'
        while 'bdc' is not
    """

    if len(s1) > len(s2):
        return False

    i = 0
    for c in s1:
        j = s2.find(c, i)
        if j != -1:
            i = j + 1
            continue
        else:
            return False
    return True


INPUT_DATA_PATH = "data/abb2ent/men2ent_v1.dat"
OUTPUT_PATH = "data/abb2ent/men2ent_v1_clean.data"


def filtering():
    line_count = sum(1 for _ in open(INPUT_DATA_PATH))
    logging.info("before line #: %s" % line_count)
    count = 0
    bar = ProgressBar(total=line_count)
    out_list = []
    with open(INPUT_DATA_PATH) as in_file:
        for line in in_file:
            try:
                bar.log()

                men, ent = line.strip().split("\t")
                abb = men

                if "||" in ent:
                    break

                # remove contents in parentheses
                abb = preprocessing.rm_paren_content(abb)
                ent = preprocessing.rm_paren_content(ent)

                # remove non-chinese characters of mention
                abb = preprocessing.rm_non_chinese_char(abb)
                ent = preprocessing.rm_non_chinese_char(ent)
                if 1 < len(abb) < len(ent) < 15 and is_substring(abb, ent):
                    count += 1
                    out_list.append(abb + "\t" + ent + "\n")
            except Exception:
                logging.error("error occur in line: {0}\nline text: {1}".format(count, line))
                continue

    with open(OUTPUT_PATH, 'w') as out_file:
        for e in out_list:
            out_file.write(e)

    logging.info("after line #: %s" % count)
    logging.info("unique #: %s" % len(out_list))


# def merge():
#     out_set = set()
#     with open("data/abb2ent_v1.dat") as in_file:
#         for line in in_file:
#             spl = line.split("\t")
#             abb = spl[0].strip()
#             ent = spl[1].strip()
#             if 1 < len(abb) < len(ent) < 20 and is_substring(abb, ent):
#                 out_set.add(ent + "\t" + abb + "\n")
#     with open("data/ent2abb_v2.dat") as in_file:
#         for line in in_file:
#             spl = line.split("\t")
#             abb = spl[1]
#             ent = spl[0]
#             if 1 < len(abb) < len(ent) < 20 and is_substring(abb, ent):
#                 out_set.add(ent + "\t" + abb + "\n")
#
#     with open("data/ent2abb_all.dat", 'w') as out_file:
#         for e in out_set:
#             out_file.write(e)
#
#     logging.info("all #: %s" % len(out_set))
#
#
# def test_entity():
#     word_set = set()
#     count = 0
#     out_set = set()
#     with open("data/chinese_word_dict.dat") as in_file:
#         for line in in_file:
#             word = line.split("\t")[0].strip()
#             word_set.add(word)
#
#     with open("data/ent2abb_all.dat") as in_file:
#         for line in in_file:
#             ent = line.split("\t")[0].strip()
#             if ent in word_set:
#                 count += 1
#                 out_set.add(line.strip() + "\n")
#
#     with open("data/ent2abb_clean.dat", "w") as out_file:
#         for e in out_set:
#             out_file.write(e)
#     logging.info("count: %s" % count)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    filtering()

