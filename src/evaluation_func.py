import re
from copy import deepcopy

import numpy as np

from utils import wordize_and_map


def _get_zero_confusion_matrix():
    return {
        'sentence_level': {
            'detect': {
                'TP': 0, 'FP': 0, 'FN': 0
            },
            'correct': {
                'TP': 0, 'FP': 0, 'FN': 0
            },
        },
        'character_level': {
            'detect': {
                'TP': 0, 'FP': 0, 'FN': 0
            },
            'correct': {
                'TP': 0, 'FP': 0, 'FN': 0
            },
        },
    }


def _accumlate_confusion_matrix(confusion_matrix, true_position, pred_position):
    true_indexes = set([i for i, char in true_position])
    pred_indexes = set([i for i, char in pred_position])
    true_pairs = set(true_position)
    pred_pairs = set(pred_position)

    # sentence level detect
    if true_indexes == pred_indexes:
        if len(pred_indexes) > 0:
            confusion_matrix['sentence_level']['detect']['TP'] += 1
    else:
        if len(pred_indexes) > 0:
            confusion_matrix['sentence_level']['detect']['FP'] += 1
        if len(true_indexes) > 0:
            confusion_matrix['sentence_level']['detect']['FN'] += 1

    # sentence level correct
    if true_pairs == pred_pairs:
        if len(pred_pairs) > 0:
            confusion_matrix['sentence_level']['correct']['TP'] += 1
    else:
        if len(pred_pairs) > 0:
            confusion_matrix['sentence_level']['correct']['FP'] += 1
        if len(true_pairs) > 0:
            confusion_matrix['sentence_level']['correct']['FN'] += 1

    # character level detect
    true_positive_positions = true_indexes & pred_indexes
    confusion_matrix['character_level']['detect']['TP'] += len(true_positive_positions)
    confusion_matrix['character_level']['detect']['FP'] += len(pred_indexes - true_positive_positions)
    confusion_matrix['character_level']['detect']['FN'] += len(true_indexes - true_positive_positions)

    # character level correct
    pred_pairs_with_only_detect_true = set()
    for position, char in pred_pairs:
        if position in true_positive_positions:
            pred_pairs_with_only_detect_true.add((position, char))
    true_positive_pairs = true_pairs & pred_pairs_with_only_detect_true
    confusion_matrix['character_level']['correct']['TP'] += len(true_positive_pairs)
    confusion_matrix['character_level']['correct']['FP'] += len(pred_pairs_with_only_detect_true - true_positive_pairs)
    confusion_matrix['character_level']['correct']['FN'] += len(true_pairs - true_positive_pairs)


def _update_csc_confusion_matrix(gt_position, pred_position, init_confusion_matrix=None):
    if init_confusion_matrix is None:
        confusion_matrix = _get_zero_confusion_matrix()
    else:
        confusion_matrix = init_confusion_matrix

    _accumlate_confusion_matrix(confusion_matrix, gt_position, pred_position)
    return confusion_matrix


def get_csc_metric(gt_positions, pred_positions):
    # build confusion matrix
    confusion_matrix = None
    for gt_position, pred_position in zip(gt_positions, pred_positions):
        confusion_matrix = _update_csc_confusion_matrix(gt_position, pred_position, init_confusion_matrix=confusion_matrix)

    # get metric
    evaluation = {}
    for level, level_values in confusion_matrix.items():
        for stage, values in level_values.items():
            metric_prefix = '{}_{}_'.format(
                'sent' if level == 'sentence_level' else 'char',
                stage,
            )
            try:
                precision = values['TP'] / (values['TP'] + values['FP'])
            except ZeroDivisionError:
                precision = None

            try:
                recall = values['TP'] / (values['TP'] + values['FN'])
            except ZeroDivisionError:
                recall = None

            if precision is None or recall is None:
                f1 = None
            else:
                try:
                    f1 = 2 / (1 / precision + 1 / recall)
                except ZeroDivisionError:
                    f1 = None

            evaluation[metric_prefix + 'precision'] = precision
            evaluation[metric_prefix + 'recall'] = recall
            evaluation[metric_prefix + 'f1'] = f1
    return evaluation


def _remove_mark(word_lst):
    ch_en_num_rule = r'[a-zA-Z0-9\u4e00-\u9fa5\u3105-\u3129\u02CA\u02C7\u02CB\u02D9\s]'
    removed_word_list = []
    for word in word_lst:
        if re.match(ch_en_num_rule, word):
            removed_word_list.append(word)
    return removed_word_list


def _english_upper(word_lst):
    return [word.upper() for word in word_lst]


def _create_opt_and_cost_matrix(ref_lst, hyp_lst):
    len_ref = len(ref_lst)
    len_hyp = len(hyp_lst)
    cost_matrix = np.zeros((len_hyp+1, len_ref+1), dtype=np.int16)
    operation_matrix = np.zeros((len_hyp, len_ref), dtype=np.int8)
    for ref_idx in range(len_ref+1):
        cost_matrix[0][ref_idx] = ref_idx
    for hyp_idx in range(len_hyp+1):
        cost_matrix[hyp_idx][0] = hyp_idx
    for hyp_idx in range(len_hyp):
        for ref_idx in range(len_ref):
            if ref_lst[ref_idx] == hyp_lst[hyp_idx]:
                cost_matrix[hyp_idx+1][ref_idx + 1] = cost_matrix[hyp_idx][ref_idx]
            else:
                substitution = cost_matrix[hyp_idx][ref_idx] + 1
                insertion = cost_matrix[hyp_idx][ref_idx+1] + 1
                deletion = cost_matrix[hyp_idx+1][ref_idx] + 1
                operation_lst = [substitution, insertion, deletion]
                min_cost_val = min(operation_lst)
                operation_idx = operation_lst.index(min_cost_val) + 1
                cost_matrix[hyp_idx+1][ref_idx+1] = min_cost_val
                operation_matrix[hyp_idx][ref_idx] = operation_idx
    return operation_matrix, cost_matrix


def _trans_trace_from_start(executions):
    adj_num = 0
    adj_executions = []
    for idx, exe, ref, hyp in executions[::-1]:
        idx += adj_num
        if exe == 'insertion':
            adj_num += 1
        elif exe == 'deletion':
            adj_num -= 1
        adj_executions.append((idx, exe, ref, hyp))
    return adj_executions


def editops(ref_lst, hyp_lst):
    ref_lst, hyp_lst = deepcopy(ref_lst), deepcopy(hyp_lst)
    operation_matrix, _ = _create_opt_and_cost_matrix(ref_lst, hyp_lst)
    executions = []
    ref_idx, hyp_idx = len(ref_lst)-1, len(hyp_lst)-1
    # ------------ decode from back ------------
    while (ref_idx >= 0) | (hyp_idx >= 0):
        if (ref_idx >= 0) & (hyp_idx >= 0):
            if operation_matrix[hyp_idx][ref_idx] == 0:
                ref_idx -= 1
                hyp_idx -= 1
            elif operation_matrix[hyp_idx][ref_idx] == 1:
                executions.append(
                    (ref_idx, 'subsitute', ref_lst[ref_idx], hyp_lst[hyp_idx]))
                ref_idx -= 1
                hyp_idx -= 1
            elif operation_matrix[hyp_idx][ref_idx] == 2:
                executions.append(
                    (ref_idx+1, 'insertion', ref_lst[ref_idx], hyp_lst[hyp_idx]))
                hyp_idx -= 1
            elif operation_matrix[hyp_idx][ref_idx] == 3:
                executions.append(
                    (ref_idx, 'deletion', ref_lst[ref_idx], hyp_lst[hyp_idx]))
                ref_idx -= 1
        # ------------ 邊界處理 ------------
        elif (ref_idx < 0):
            executions.append((0, 'insertion', None, hyp_lst[hyp_idx]))
            hyp_idx -= 1
        elif (hyp_idx < 0):
            executions.append((ref_idx, 'deletion', ref_lst[ref_idx], None))
            ref_idx -= 1
    executions = _trans_trace_from_start(executions)
    return executions


def calculate_cer(gt_str, pred_str, exec_detail=True):
    gt_words, _, _ = wordize_and_map(gt_str)
    pred_words, _, _ = wordize_and_map(pred_str)

    gt_words = _remove_mark(gt_words)
    pred_words = _remove_mark(pred_words)
    gt_words = _english_upper(gt_words)
    pred_words = _english_upper(pred_words)

    if exec_detail:
        executions = editops(ref_lst=gt_words, hyp_lst=pred_words)
        cer = len(executions) / len(gt_words)
        return cer, executions
    else:
        _, cost_matrix = _create_opt_and_cost_matrix(
            ref_lst=gt_words, hyp_lst=pred_words)
        cer = cost_matrix[-1][-1] / len(gt_words)
        return cer
