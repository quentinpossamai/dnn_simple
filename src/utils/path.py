import os
import pathlib

WD = f'{os.getcwd()}/'
ANALYSIS = f'{WD}analysis/'
DATA = f'{WD}data/'


def mkdir(paths: list):
    assert isinstance(paths, list), "Path given must be a list."
    assert WD in ["/Users/quentin/learning_tests/dnn_simple/", ]
    for path in paths:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)