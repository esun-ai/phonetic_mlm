import os
import argparse
from datetime import datetime
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertForTokenClassification

from dataset import prepare_data, TypoDataset
from utils import load_config, RunningAverage, get_logger
from detector_utils import obtain_valid_detection_preds


def train_batch(model, data, optimizer, device):
    model.train()
    input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in data[:4]]

    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def _get_metric(input_ids, labels, preds):
    precisions, recalls, f1s = [], [], []
    for token_id_list, label_list, pred_list in zip(input_ids, labels, preds):
        true_pos, false_pos, false_neg = 0, 0, 0
        for token_id, label, pred in zip(token_id_list, label_list, pred_list):
            true_pos += 1 if label == 1 and pred == 1 else 0
            false_pos += 1 if label == 0 and pred == 1 else 0
            false_neg += 1 if label == 1 and pred == 0 else 0

        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else None
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else None
        f1 = 2 / (1 / precision + 1 / recall) if precision and recall else None

        if precision is not None:
            precisions.append(precision)
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1s.append(f1)
    return precisions, recalls, f1s


def log_examples(logger, tokenizer, examples):
    hline = '--------------------------------------------------------------------------------'
    logger.info(hline)
    logger.info(f'{"Ground Truth":25s} | {"Prediction":25s} | T/F')
    logger.info(hline)
    skip_tokens = [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
    for input_id_list, pred_list, label_list in examples:
        pred_sentence = ''
        label_sentence = ''
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)
        for token, pred, label in zip(tokens, pred_list, label_list):
            if token in skip_tokens:
                continue
            pred_sentence += f'×{token}' if pred == 1 else token
            label_sentence += f'×{token}' if label == 1 else token
        is_correct = pred_sentence == label_sentence

        logger.info(f'{label_sentence:25s} | {pred_sentence:25s} | {str(is_correct)}')
    logger.info(hline)


def evaluate(model, valid_loader, device, tokenizer, threshold=0.5):
    model.eval()

    loss_averger = RunningAverage()
    precision_averger = RunningAverage()
    recall_averger = RunningAverage()
    f1_averger = RunningAverage()

    input_ids_collection = []
    preds_collection = []
    labels_collection = []

    with torch.no_grad():
        for data in tqdm(valid_loader, desc='evaluate'):
            input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in data[:-1]]

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss_averger.add(outputs.loss.item())

            preds = obtain_valid_detection_preds(outputs.logits, input_ids, tokenizer, threshold=threshold)

            input_ids_collection.extend(input_ids.cpu().tolist())
            preds_collection.extend(preds)
            labels_collection.extend(labels.cpu().tolist())

            precisions, recalls, f1s = _get_metric(input_ids, labels, preds)
            precision_averger.add_all(precisions)
            recall_averger.add_all(recalls)
            f1_averger.add_all(f1s)

    evaluation = {
        'loss': loss_averger.get(),
        'averge_precision': precision_averger.get(),
        'averge_recall': recall_averger.get(),
        'averge_f1': f1_averger.get(),
        'examples': list(zip(input_ids_collection, preds_collection, labels_collection))
    }
    return evaluation


def main(config):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.manual_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed and GPU setting
    random.seed(config.manual_seed)
    np.random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    save_checkpoint_dir = os.path.join('saved_models/', config.exp_name)
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    logger_file_path = os.path.join(save_checkpoint_dir, 'record.log')
    logger = get_logger(logger_file_path)

    logger.info(f'device: {device}')
    logger.info(f'now: {datetime.now()}')

    tokenizer = BertTokenizer.from_pretrained(config.model_source)

    train_texts, train_typos, _, _ = prepare_data(config.train_json)
    valid_texts, valid_typos, _, _ = prepare_data(config.valid_json)

    train_dataset = TypoDataset(tokenizer, train_texts, train_typos, max_length=config.max_len, for_train=True, for_detect=True)
    valid_dataset = TypoDataset(tokenizer, valid_texts, valid_typos, max_length=config.max_len, for_train=True, for_detect=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        collate_fn=train_dataset.create_mini_batch,
        shuffle=True,
        num_workers=config.num_workers
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        collate_fn=valid_dataset.create_mini_batch,
        num_workers=config.num_workers
    )

    model = BertForTokenClassification.from_pretrained(config.model_source, return_dict=True, num_labels=2)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    i = 1
    is_running = True
    train_loss_averager = RunningAverage()
    best_f1 = 0.

    while is_running:
        for train_data in train_loader:
            loss = train_batch(model, train_data, optimizer, device)
            train_loss_averager.add(loss)

            if i % config.val_interval == 0:
                # training
                train_loss = train_loss_averager.get()
                train_loss_averager.flush()

                # validation
                evaluation = evaluate(model, valid_loader, device, tokenizer)
                f1 = evaluation['averge_f1']

                # save model
                if f1 > best_f1:
                    path = os.path.join(save_checkpoint_dir, 'best_f1.pth')
                    torch.save(model.state_dict(), path)

                    best_f1 = f1

                # log
                logger.info(f'[{i}] train_loss={train_loss} valid_loss={loss} valid_f1={f1} best_f1={best_f1}')

                examples = evaluation['examples']
                sampled = random.sample(examples, k=8)
                log_examples(logger, tokenizer, sampled)

                logger.info(f'now: {datetime.now()}')

            if i >= config.num_iter:
                is_running = False
                break

            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config path')
    opt = parser.parse_args()

    config = load_config(opt.config)

    main(config)
