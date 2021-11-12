from unittest.mock import Mock

import pytest

from evaluation_func import get_csc_metric, _remove_mark, _english_upper, editops, calculate_cer


def test_get_csc_confusion_matrix_at_e2e():
    gt_positions = [
        [],
        [(0, '也'), (1, '是')],
        [(0, '玉'), (2, '銀')],
        [],
        [(0, '葉')]
    ]
    pred_positions = [
        [(2, '我')],
        [(0, '不'), (1, '是')],
        [(0, '玉'), (2, '銀')],
        [],
        [(1, '有')]
    ]
    evaluation = get_csc_metric(gt_positions, pred_positions)

    expected_confusion_matrix = {
        'sentence_level': {'detect': {'FN': 1, 'FP': 2, 'TP': 2}, 'correct': {'FN': 2, 'FP': 3, 'TP': 1}},
        'character_level': {'detect': {'FN': 1, 'FP': 2, 'TP': 4}, 'correct': {'FN': 2, 'FP': 1, 'TP': 3}},
    }

    def get_precision(values):
        return values['TP'] / (values['TP'] + values['FP'])

    def get_recall(values):
        return values['TP'] / (values['TP'] + values['FN'])

    def get_f1(values):
        return 2 / (1 / get_precision(values) + 1 / get_recall(values))

    expected_evaluation = {
        'char_correct_f1': get_f1(expected_confusion_matrix['character_level']['correct']),
        'char_correct_precision': get_precision(expected_confusion_matrix['character_level']['correct']),
        'char_correct_recall': get_recall(expected_confusion_matrix['character_level']['correct']),
        'char_detect_f1': get_f1(expected_confusion_matrix['character_level']['detect']),
        'char_detect_precision': get_precision(expected_confusion_matrix['character_level']['detect']),
        'char_detect_recall': get_recall(expected_confusion_matrix['character_level']['detect']),
        'sent_correct_f1': get_f1(expected_confusion_matrix['sentence_level']['correct']),
        'sent_correct_precision': get_precision(expected_confusion_matrix['sentence_level']['correct']),
        'sent_correct_recall': get_recall(expected_confusion_matrix['sentence_level']['correct']),
        'sent_detect_f1': get_f1(expected_confusion_matrix['sentence_level']['detect']),
        'sent_detect_precision': get_precision(expected_confusion_matrix['sentence_level']['detect']),
        'sent_detect_recall': get_recall(expected_confusion_matrix['sentence_level']['detect'])
    }
    assert evaluation == expected_evaluation


@pytest.mark.parametrize('inputs, expect_output',
                         [(['好', '個', 'abcd'], ['好', '個', 'ABCD']),
                          (['好', 'AbCd', '啊'], ['好', 'ABCD', '啊']),
                          (['讚', 'aBcD'], ['讚', 'ABCD'])
                          ])
def test_english_upper(inputs, expect_output):
    outputs = _english_upper(inputs)
    assert outputs == expect_output


@pytest.mark.parametrize('inputs, expect_output',
                         [(['好', '個', 'abcd', '，'], ['好', '個', 'abcd']),
                          (['好', 'AbCd', '啊', '!!!'], ['好', 'AbCd', '啊']),
                          (['讚', 'aBcD', '^_^'], ['讚', 'aBcD'])
                          ])
def test_remove_mark(inputs, expect_output):
    outputs = _remove_mark(inputs)
    assert outputs == expect_output


@pytest.mark.parametrize('reference, hypothesis, expected_exec_from_start',
                         [(['中', '美', '貿', '易', '摩', '擦', '不', '斷', 'BY', 'ESUNBANK'], 
                           ['真', '棒', '種', '美', '貿', '易', '摩', '擦', '不', '斷', 'BY', 'ESUN1313'],
                           [(0, 'insertion', None, '真'), (1, 'insertion', None, '棒'), (2, 'subsitute', '中', '種'), (11, 'subsitute', 'ESUNBANK', 'ESUN1313')]
                           ),
                          (['BY', 'ESUNBANK', '中', '美', '貿', '易', '摩', '擦', '不', '斷'],
                           ['BY', 'ESUN1313', '真', '棒', '種', '美', '貿', '易', '摩', '擦', '不', '斷'],
                           [(1, 'insertion', 'BY', 'ESUN1313'), (2, 'insertion', 'BY', '真'), (3, 'subsitute', 'ESUNBANK', '棒'), (4, 'subsitute', '中', '種')]
                           ),
                          (['中', '美', '貿', '易', '摩', '擦', '不', '斷'],
                           ['鐘', '美', '貿', '易', '摩', '擦', '不', '斷'],
                           [(0, 'subsitute', '中', '鐘')]
                           )
                          ])
def test_editops(reference, hypothesis, expected_exec_from_start):
    exec_from_top = editops(reference, hypothesis)
    assert exec_from_top == expected_exec_from_start


@pytest.mark.parametrize('reference, hypothesis, expected_cer',
                         [('中美貿易摩擦不斷 by esunbank',
                           '真棒 ，種美 貿易 摩擦 不斷 by esun1313',
                           0.4),
                          ('by esunbank: 中美貿易摩擦不斷',
                           'by esun1313: 真棒 ，種美 貿易 摩擦 不斷',
                           0.4),
                          ('by esunbank: 中美貿易摩擦不斷',
                           'by esun1313: !@#$%^&*()_+真棒 ，種美 貿易 摩擦 不斷',
                           0.4),
                          ('by esunbank: 中美貿易摩擦不斷',
                           '!@#$%^&*()_+真棒 ，種美 貿易 摩擦 不斷 by esunbank',
                           0.5),
                          ('中美貿易摩擦不斷',
                           '鐘美 貿易 摩擦 不斷',
                           0.125)])
def test_calculate_cer(reference, hypothesis, expected_cer):
    cer = calculate_cer(reference, hypothesis, exec_detail=False)
    assert cer == expected_cer
