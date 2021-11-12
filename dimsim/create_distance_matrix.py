import sys
import math

import numpy as np

from dimsim_data import convert_table, consonantMap_TwoDCode, vowelMap_TwoDCode, hardcodeMap


def _get_distance_2d_code(X, Y):
    x1, x2 = X
    y1, y2 = Y
    x1d = abs(x1 - y1)
    x2d = abs(x2 - y2)
    return math.sqrt(x1d**2 + x2d**2)


def _get_sim_dis_from_hardcod_map(consonant_1, vowel_1, consonant_2, vowel_2):
    without_tone_1 = f'{consonant_1}{vowel_1}'
    without_tone_2 = f'{consonant_2}{vowel_2}'

    simPy = hardcodeMap.get(without_tone_1)
    if simPy is not None:
        if simPy == without_tone_2:
            return 2.0
    else:
        simPy = hardcodeMap.get(without_tone_2)
        if simPy is not None and simPy == without_tone_1:
            return 2.0
    return sys.float_info.max


def get_dimsim_distance(bopomofo_1, bopomofo_2):
    tone_1 = int(bopomofo_1[-1])
    tone_2 = int(bopomofo_2[-1])
    consonant_1, vowel_1 = convert_table[bopomofo_1[:-1]]
    consonant_2, vowel_2 = convert_table[bopomofo_2[:-1]]

    twoDcode_consonant_1 = consonantMap_TwoDCode[consonant_1]
    twoDcode_consonant_2 = consonantMap_TwoDCode[consonant_2]
    consonant_dist = abs(_get_distance_2d_code(twoDcode_consonant_1, twoDcode_consonant_2))

    twoDcode_vowel_1 = vowelMap_TwoDCode[vowel_1]
    twoDcode_vowel_2 = vowelMap_TwoDCode[vowel_2]
    vowel_dist = abs(_get_distance_2d_code(twoDcode_vowel_1, twoDcode_vowel_2))

    hardcod_dist = _get_sim_dis_from_hardcod_map(consonant_1, vowel_1, consonant_2, vowel_2)

    return min((consonant_dist + vowel_dist), hardcod_dist) + abs(tone_1 - tone_2) / 10.0


def main():
    output_path = 'dimsim_dist_matrix.npz'

    bopomofos = [f'{without_tone}{tone}' for tone in ['1', '2', '3', '4', '5'] for without_tone in convert_table]
    print(f'number of bopomofos: {len(bopomofos)}')

    dist_matrix = []
    for bopomofo_1 in bopomofos:
        dist_row = []
        for bopomofo_2 in bopomofos:
            dist = get_dimsim_distance(bopomofo_1, bopomofo_2)
            dist_row.append(dist)
        dist_matrix.append(dist_row)

    np.savez(output_path, bopomofo=np.array(bopomofos), dist_matrix=np.array(dist_matrix))


if __name__ == '__main__':
    main()
