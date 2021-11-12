import os
from collections import defaultdict
import re

import numpy as np
from pypinyin import pinyin, Style

from utils import CHINESE_RE

DIMSIM_MATRIX_PATH = os.path.join(os.path.dirname(__file__), 'dimsim_dist_matrix.npz')
MAX_DIMSIM_DISTANCE = 140000


def phonize(char):
    def convert_pypinyin_phoneme(bopomofo):
        last = bopomofo[-1]
        mapping = {
            'ˊ': 2,
            'ˇ': 3,
            'ˋ': 4,
            '˙': 5,
        }
        if last in mapping:
            return bopomofo[:-1] + str(mapping[last])
        else:
            return bopomofo + '1'
    bopomofo = pinyin(char, style=Style.BOPOMOFO)[0][0]

    bopomofo = convert_pypinyin_phoneme(bopomofo)
    return bopomofo


def prepare_dimsim(vocab_dict):
    # load dimsim data
    dimsim_matrix_npz = np.load(DIMSIM_MATRIX_PATH)
    bopomofo_dict = {symbol: i for i, symbol in enumerate(dimsim_matrix_npz['bopomofo'])}
    dimsim_matrix = dimsim_matrix_npz['dist_matrix']  # matrix of dimsim distance, shape: [num_bopomofo, num_bopomofo]
    num_bopomofo = len(bopomofo_dict)
    num_vocab = len(vocab_dict)

    # build the matrix which converts bopomofo_id to token_id
    bopomofo_to_token = np.zeros((num_bopomofo + 1, num_vocab))  # +1 for unknown bopomofo
    unknown_bopomofo_id = len(bopomofo_dict)
    for token, token_id in vocab_dict.items():
        if re.match(CHINESE_RE, token):
            bopomofo_id = bopomofo_dict.get(phonize(token), unknown_bopomofo_id)
        else:
            bopomofo_id = unknown_bopomofo_id
        bopomofo_to_token[bopomofo_id, token_id] = 1

    # set MAX_DIMSIM_DISTANCE on dimsim_matrix
    dimsim_matrix[dimsim_matrix > MAX_DIMSIM_DISTANCE] = MAX_DIMSIM_DISTANCE
    # augment dimsim_matrix for unknown bopomofo
    dist_to_unknown = np.full((num_bopomofo, 1), MAX_DIMSIM_DISTANCE)
    dimsim_matrix = np.hstack((dimsim_matrix, dist_to_unknown))  # shape: [num_bopomofo, num_bopomofo + 1]

    bopomofo_to_dist = np.matmul(dimsim_matrix, bopomofo_to_token)  # shape: [num_bopomofo, num_vocab]
    # augment bopomofo_to_dist for unknown bopomofo
    unknown_bopomofo_dist_vector = np.full((1, num_vocab), MAX_DIMSIM_DISTANCE)
    bopomofo_to_dist = np.vstack((bopomofo_to_dist, unknown_bopomofo_dist_vector))  # shape: [num_bopomofo + 1, num_vocab]

    # update bopomofo_dict
    bopomofo_dict['UNK'] = len(bopomofo_dict)

    assert (bopomofo_to_dist >= 0.).all()
    return bopomofo_dict, bopomofo_to_dist
