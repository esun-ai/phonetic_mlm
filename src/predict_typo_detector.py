import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from tqdm import tqdm

from dataset import prepare_data, TypoDataset
from detector_utils import obtain_valid_detection_preds
from utils import load_config


def predict(model, dataloader, device, tokenizer, threshold=0.5):
    model.eval()

    detected_char_positions_collect = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='predict'):
            input_ids, token_type_ids, attention_mask = [d.to(device) for d in data[:3]]
            infos = data[-1]

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )

            preds = obtain_valid_detection_preds(outputs.logits, input_ids, tokenizer, threshold=threshold)

            for pred_seq, info in zip(preds, infos):
                token2text = info['token2text']
                detected_char_positions = []
                for i, is_detected in enumerate(pred_seq):
                    if is_detected == 1:
                        token_position = i - 1  # remove [CLS]
                        start, end = token2text[token_position]
                        detected_char_positions.append(start)
                detected_char_positions_collect.append(detected_char_positions)

    return detected_char_positions_collect


def main(config, checkpoint, text_path, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(config.model_source)

    texts = [line.rstrip() for line in open(text_path).readlines()]

    dataset = TypoDataset(tokenizer, texts, max_length=config.max_len, for_train=False, for_detect=True)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_mini_batch,
        num_workers=config.num_workers
    )

    model = BertForTokenClassification.from_pretrained(config.model_source, return_dict=True, num_labels=2)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    detected_char_positions_collect = predict(model, dataloader, device, tokenizer, threshold=threshold)

    for text, detected_char_position in zip(texts, detected_char_positions_collect):
        for i in sorted(detected_char_position, reverse=True):
            text = text[:i] + '×' + text[i:]
        print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint')
    parser.add_argument('--text_path', required=True, help='text path')
    parser.add_argument('--threshold', type=float, default=0.5, help='probability threshold')
    opt = parser.parse_args()

    config = load_config(opt.config)

    main(config, opt.checkpoint, opt.text_path, opt.threshold)
