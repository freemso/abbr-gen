#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pre-processing helper functions for chinese NLP tasks
"""

import unicodedata

__author__ = 'freemso'

import re

PUNC_NORM_MAP = {
    "·": re.compile("[•]"),
    ",": re.compile("[，]"),
    ".": re.compile("[。]"),
    "(": re.compile("[（]"),
    ")": re.compile("[）]"),
    "!": re.compile("[！]"),
    "\"": re.compile("[“”]"),
    ":": re.compile("[：]"),
    ";": re.compile("[；]"),
    "~": re.compile("[～]]"),
    "?": re.compile("[？]"),
    "[": re.compile("[【]"),
    "]": re.compile("[】]"),
    "+": re.compile("[＋]"),
    "=": re.compile("[＝]"),
    "&": re.compile("＆")
}

WHITE_SPACES = ("\u0020\u00A0\u1680\u180E"
                "\u2000\u2001\u2002\u2003\u2004\u2005"
                "\u2006\u2007\u2008\u2009\u200A\u200B"
                "\u202F\u205F\u3000\uFEFF\n\r\t")


# Regular expression that matches white spaces
REGEX_WHITE_SPACES = re.compile('[%s]' % WHITE_SPACES)

REGEX_SENTENCE_SEPARATOR = re.compile("[，。！？；;,.?!•◆]")

REGEX_PUNC = re.compile(
    """[“”＋＝＆:|,，.。\]\[}{：;；'"\\\\、()（）`·《》<>?？/+\-=—_~～!#$%^&*！＠@￥…×\n\t\r\b　．＊【】]""")

REGEX_TAGS = re.compile(r"<.*?>")

REGEX_PAREN_CONTENT = re.compile("[(|（].*?[)|）]")


def is_punc_mark(c):
    """
    Check whether a character is a punctuation mark
    """
    return bool(REGEX_PUNC.match(c))


def is_chinese_char(c):
    """
    Check whether a character is a chinese character
    """
    return u'\u4e00' <= c <= u'\u9fff'


def rm_punc_marks(s):
    """
    Remove all the punctuation marks within a string
    :param s: the string to operate
    :return: the string after removal
    """
    return REGEX_PUNC.sub('', s)


def rm_paren_content(s):
    """
    Remove the parentheses and the contents within
    """
    return REGEX_PAREN_CONTENT.sub('', s)


def has_non_chinese_char(s):
    """
    Check whether a string contains non chinese characters
    """
    for c in s:
        if not is_chinese_char(c):
            return True
    return False


def rm_non_chinese_char(s):
    """
    Remove all non chinese characters
    """
    return "".join([c for c in s if is_chinese_char(c)])


def cut_to_sentences(doc):
    """
    Cut the chinese document into list of sentences
    """
    return REGEX_SENTENCE_SEPARATOR.split(doc)


def normalize_unicode(text):
    """
    Normalize unicode text
    """
    return unicodedata.normalize("NFKC", text)


def normalize_punc(text):
    """
    Merge different types of punctuations into one using predefined map
    """
    for norm, regex in PUNC_NORM_MAP.items():
        text = regex.sub(norm, text)
    return text


def rm_tags(text):
    """
    Remove tags(e.g. <html></html>) in the text
    """
    return REGEX_TAGS.sub('', text)


def rm_whitespaces(text):
    return REGEX_WHITE_SPACES.sub('', text)


def split_with_whitespace(s):
    return REGEX_WHITE_SPACES.split(s)

