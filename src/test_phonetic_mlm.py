import os
import argparse
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification, BertForMaskedLM
from tqdm import tqdm

from dataset import prepare_data, TypoDataset
from utils import load_config, RunningAverage
from dimsim import prepare_dimsim
from module import PhoneticMLM
from evaluation_func import calculate_cer, get_csc_metric


def replace_correction(texts, correct_positions):
    corrected_texts = []
    for text, correct_position in zip(texts, correct_positions):
        for position, correct_char in correct_position:
            text = text[:position] + correct_char + text[position+1:]
        corrected_texts.append(text)
    return corrected_texts


def evaluate(texts, gt_texts, subsitute_gt_positions, pred_positions):
    cer_averger = RunningAverage()
    origin_cer_averager = RunningAverage()

    corrected_texts = replace_correction(texts, pred_positions)

    for text, gt_text, corrected_text in zip(texts, gt_texts, corrected_texts):
        cer = calculate_cer(gt_text, corrected_text, exec_detail=False)
        cer_averger.add(cer)

        origin_cer = calculate_cer(gt_text, text, exec_detail=False)
        origin_cer_averager.add(origin_cer)

    evaluation = {
        'avg_cer': cer_averger.get(),
        'avg_origin_cer': origin_cer_averager.get(),
        'csc_metric': get_csc_metric(subsitute_gt_positions, pred_positions)
    }
    return evaluation


def predict(texts, model, dataloader, tokenizer, device):
    model.eval()

    correct_positions = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='predict'):
            input_ids, token_type_ids, attention_mask, bopomofo_ids = [d.to(device) for d in data[:4]]
            infos = data[-1]

            logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                bopomofo_ids=bopomofo_ids
            )
            output_ids = logits.argmax(dim=-1)
            is_diffs = input_ids != output_ids

            output_ids = output_ids.cpu().tolist()
            is_diffs = is_diffs.cpu().tolist()

            for is_diff_seq, output_id_seq, info in zip(is_diffs, output_ids, infos):
                token2text = info['token2text']
                correct_position = []
                for i, (is_diff, output_id) in enumerate(zip(is_diff_seq, output_id_seq)):
                    if is_diff:
                        start, end = token2text[i - 1]  # remove [CLS]
                        assert start + 1 == end
                        position = start
                        correct_char = tokenizer.convert_ids_to_tokens(output_id)
                        correct_position.append((position, correct_char))
                correct_positions.append(correct_position)

    return correct_positions


def main(config, detector_config, test_json):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(config.model_source)

    bopomofo_dict, bopomofo_to_dist = prepare_dimsim(tokenizer.vocab)

    texts, typos, _, true_texts = prepare_data(test_json, obtain_true_text=True)

    dataset = TypoDataset(tokenizer, texts, typos, bopomofos=None, bopomofo_dict=bopomofo_dict, max_length=detector_config.max_len,
                          for_train=True, for_detect=False)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_mini_batch,
        num_workers=config.num_workers
    )

    detector = BertForTokenClassification.from_pretrained(detector_config.model_source, return_dict=True, num_labels=2)
    detector.load_state_dict(torch.load(config.detector_checkpoint_path, map_location=device))
    detector.to(device)

    mlm = BertForMaskedLM.from_pretrained(config.model_source)
    mlm.to(device)

    unknown_bopomofo_id = bopomofo_dict['UNK']
    model = PhoneticMLM(detector, mlm, tokenizer, bopomofo_to_dist, unknown_bopomofo_id, alpha=config.alpha, detector_threshold=config.detector_threshold)

    pred_positions = predict(texts, model, dataloader, tokenizer, device)

    subsitute_gt_positions = [[(i, char) for _, char, i, _ in typo_list] for typo_list in typos]
    evaluation = evaluate(texts, true_texts, subsitute_gt_positions, pred_positions)
    pprint(evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config path')
    parser.add_argument('--json', required=True, help='json path')

    opt = parser.parse_args()

    config = load_config(opt.config)
    detector_config = load_config(config.detector_config_path)

    main(config, detector_config, opt.json)
