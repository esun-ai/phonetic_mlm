import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import tokenize_and_map


def prepare_data(path):
    texts, typos, bopomofos = [], [], []
    for line in open(path).readlines():
        line = line.strip()
        content = json.loads(line)
        text = content['sentence']

        typo = []
        for (wrong_char, corrected_char, start) in content['typo']:
            typo.append([wrong_char, corrected_char, start, start + 1])
        texts.append(text)
        typos.append(typo)

        if 'bopomofos' in content:
            bopomofo = content['bopomofos']
            bopomofos.append(bopomofo)
    if bopomofos:
        return texts, typos, bopomofos
    else:
        return texts, typos, None


class TypoDataset(Dataset):
    def __init__(self, tokenizer, texts, typos=None, max_length=512, for_train=True, for_detect=True):
        self.tokenizer = tokenizer
        self.texts = texts
        self.typos = typos
        self.max_length = max_length
        self.for_train = for_train
        self.for_detect = for_detect

    def _truncate(self, max_len, text, tokens, text2token, token2text):
        truncate_len = max_len - 2
        if len(tokens) <= truncate_len:
            return (text, tokens, text2token, token2text)

        cut_index = truncate_len
        cut_text_index = text2token.index(cut_index)

        text = text[:cut_text_index]
        tokens = tokens[:cut_index]
        text2token = text2token[:cut_text_index]
        token2text = token2text[:cut_index]
        return (text, tokens, text2token, token2text)

    def _obtain_typo_flags(self, typo, text2token):
        typo_flags = []
        for char_position, token_index in enumerate(text2token):
            if token_index is None:
                continue

            typo_flag = 0
            for typo_word, correct_word, start, end in typo:
                if start <= char_position < end:
                    typo_flag = 1
                    break

            if token_index >= len(typo_flags):  # append是參照token level而非char level
                typo_flags.append(typo_flag)
        return typo_flags

    def _obtain_correct_tokens(self, typo, text2token):
        correct_tokens = []
        for char_position, token_index in enumerate(text2token):
            if token_index is None:
                continue

            correct_token = tokens[token_index]
            for typo_word, correct_word, start, end in typo:
                if start <= char_position < end:
                    correct_token = correct_word
                    break

            if token_index >= len(correct_tokens):  # append是參照token level而非char level
                correct_tokens.append(correct_token)
        return correct_tokens

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens, text2token, token2text = tokenize_and_map(self.tokenizer, text)
        text, tokens, text2token, token2text = \
            self._truncate(self.max_length, text, tokens, text2token, token2text)

        processed_tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(processed_tokens))
        token_type_ids = torch.tensor([0] * len(processed_tokens))
        attention_mask = torch.tensor([1] * len(processed_tokens))

        outputs = (input_ids, token_type_ids, attention_mask, )

        if self.for_train:
            typo = self.typos[idx]
            if self.for_detect:
                typo_flags = self._obtain_typo_flags(typo, text2token)
                labels = torch.tensor(
                    [0] + typo_flags + [0]
                    # for [CLS] and [SEP]
                )
            else:
                correct_tokens = self._obtain_correct_tokens(typo, text2token)
                labels = torch.tensor(
                    [self.tokenizer.cls_token_id]
                    + self.tokenizer.convert_tokens_to_ids(correct_tokens)
                    + [self.tokenizer.sep_token_id]
                )

            if input_ids.size(0) != labels.size(0) or labels.size(0) > self.max_length:
                logging.warn(f'Drop sentence "{text}"')
                return self[(idx + 1) % len(self)]
            outputs += (labels, )

        info = {
            'text': text,
            'tokens': tokens,
            'text2token': text2token,
            'token2text': token2text,
        }
        outputs += (info, )
        return outputs

    def __len__(self):
        return len(self.texts)

    def create_mini_batch(self, samples):
        outputs = list(zip(*samples))

        # zero pad 到同一序列長度
        input_ids = pad_sequence(outputs[0], batch_first=True)
        token_type_ids = pad_sequence(outputs[1], batch_first=True)
        attention_mask = pad_sequence(outputs[2], batch_first=True)

        batch_output = (input_ids, token_type_ids, attention_mask)

        if self.for_train:
            labels = pad_sequence(outputs[3], batch_first=True)
            batch_output += (labels, )
        else:
            infos = outputs[3]
            batch_output += (infos, )

        return batch_output
