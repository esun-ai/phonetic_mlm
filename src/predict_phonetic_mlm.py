import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification, BertForMaskedLM
from tqdm import tqdm

from dataset import TypoDataset
from utils import load_config
from test_phonetic_mlm import predict
from dimsim import prepare_dimsim
from module import PhoneticMLM


def main(config, detector_config, text_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(config.model_source)

    bopomofo_dict, bopomofo_to_dist = prepare_dimsim(tokenizer.vocab)

    texts = [line.rstrip() for line in open(text_path).readlines()]

    dataset = TypoDataset(tokenizer, texts, bopomofos=None, bopomofo_dict=bopomofo_dict,
                          max_length=detector_config.max_len, for_train=False, for_detect=False)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_mini_batch,
        num_workers=config.num_workers
    )

    detector = BertForTokenClassification.from_pretrained(detector_config.model_source, return_dict=True, num_labels=2)
    detector.load_state_dict(torch.load(config.detector_checkpoint_path))
    detector.to(device)

    mlm = BertForMaskedLM.from_pretrained(config.model_source)
    mlm.to(device)

    unknown_bopomofo_id = bopomofo_dict['UNK']
    model = PhoneticMLM(detector, mlm, tokenizer, bopomofo_to_dist, unknown_bopomofo_id, alpha=config.alpha, detector_threshold=config.detector_threshold)

    pred_positions = predict(texts, model, dataloader, tokenizer, device)

    for text, pred_position in zip(texts, pred_positions):
        print(text, pred_position)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config path')
    parser.add_argument('--text_path', required=True, help='text path')
    opt = parser.parse_args()

    config = load_config(opt.config)
    detector_config = load_config(config.detector_config_path)

    main(config, detector_config, opt.text_path)
