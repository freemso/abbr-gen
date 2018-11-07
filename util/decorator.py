# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    All kinds of decorator functions
"""
import functools
import pathlib

import pickle

__author__ = "freemso"


def cache(file_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            file_path = pathlib.Path(file_name)
            if file_path.is_file():
                with open(file_name, "rb") as in_file:
                    return pickle.load(in_file)
            else:
                data = func(*args, **kw)
                with open(file_name, "wb") as out_file:
                    pickle.dump(data, out_file)
            return data
        return wrapper
    return decorator


if __name__ == '__main__':
    pass
