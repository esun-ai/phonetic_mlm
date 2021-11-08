import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification

from dataset import prepare_data, TypoDataset
from utils import load_config
from train_typo_detector import evaluate


def main(config, test_json, checkpoint, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(config.model_source)

    texts, typos, _ = prepare_data(test_json)

    dataset = TypoDataset(tokenizer, texts, typos, max_length=config.max_len, for_train=True, for_detect=True)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_mini_batch,
        num_workers=config.num_workers
    )

    model = BertForTokenClassification.from_pretrained(config.model_source, return_dict=True, num_labels=2)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)

    evaluation = evaluate(model, dataloader, device, tokenizer, threshold=threshold)
    precision, recall, f1 = evaluation['averge_precision'], evaluation['averge_recall'], evaluation['averge_f1']
    print(f'test: precision={precision}, recall={recall}, f1={f1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint')
    parser.add_argument('--json', required=True, help='json path')
    parser.add_argument('--threshold', type=float, default=0.5, help='probability threshold')

    opt = parser.parse_args()

    config = load_config(opt.config)

    main(config, opt.json, opt.checkpoint, opt.threshold)
