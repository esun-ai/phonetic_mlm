from unittest.mock import Mock

import torch

from dataset import TypoDataset


class TestTypoDataset:
    def test_getitem_when_detect(self):
        def convert_tokens_to_ids(tokens):
            mapping = {
                '[PAD]': 0,
                '[CLS]': 101,
                '[SEP]': 102,
                '可': 1,
                '能': 2,
                '有': 3,
                '臭': 4,
                '過': 5,
                '就': 6,
                '不': 7,
                '再': 8,
                '來': 9,
                '了': 10,
                'apple': 11,
                '走': 12,
                '路': 13,
                '要': 14,
            }
            return [mapping[token] for token in tokens]
        mocked_tokenizer = Mock()
        mocked_tokenizer.tokenize = lambda word: [word]
        mocked_tokenizer.convert_tokens_to_ids = convert_tokens_to_ids

        texts = [
            '可能有臭',
            'apple臭過就不再來了',
            '走過路過不要臭過',
            '臭臭臭'
        ]
        typos = [
            [['臭', '錯', 3, 4]],
            [['臭', '錯', 5, 6]],
            [['臭', '錯', 6, 7]],
            [['臭', '錯', 0, 1], ['臭', '錯', 1, 2], ['臭', '錯', 2, 3]],
        ]
        max_len = 7

        dataset = TypoDataset(mocked_tokenizer, texts, typos, max_length=max_len, for_train=True, for_detect=True)

        input_ids, token_type_ids, attention_mask, labels, info = dataset[0]
        expected_input_ids = torch.tensor([101, 1, 2, 3, 4, 102])
        expected_labels = torch.tensor([0, 0, 0, 0, 1, 0])
        assert torch.all(torch.eq(input_ids, expected_input_ids))
        assert torch.all(torch.eq(labels, expected_labels))

        input_ids, token_type_ids, attention_mask, labels, info = dataset[1]
        expected_input_ids = torch.tensor([101, 11, 4, 5, 6, 7, 102])
        expected_labels = torch.tensor([0, 0, 1, 0, 0, 0, 0])
        assert torch.all(torch.eq(input_ids, expected_input_ids))
        assert torch.all(torch.eq(labels, expected_labels))

        input_ids, token_type_ids, attention_mask, labels, info = dataset[2]
        expected_input_ids = torch.tensor([101, 12, 5, 13, 5, 7, 102])
        expected_labels = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        assert torch.all(torch.eq(input_ids, expected_input_ids))
        assert torch.all(torch.eq(labels, expected_labels))

        input_ids, token_type_ids, attention_mask, labels, info = dataset[3]
        expected_input_ids = torch.tensor([101, 4, 4, 4, 102])
        expected_labels = torch.tensor([0, 1, 1, 1, 0])
        assert torch.all(torch.eq(input_ids, expected_input_ids))
        assert torch.all(torch.eq(labels, expected_labels))
