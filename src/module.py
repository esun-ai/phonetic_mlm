import torch
import torch.nn as nn

from detector_utils import obtain_valid_detection_preds
from dimsim import MAX_DIMSIM_DISTANCE


class PhoneticMLM(nn.Module):
    def __init__(self, detect_bert, mlm, tokenizer, bopomofo_to_dist, unknown_bopomofo_id, alpha=500, detector_threshold=0.5):
        super().__init__()

        self.detect_bert = detect_bert
        self.mlm = mlm
        self.tokenizer = tokenizer

        self.bopomofo_to_dist = bopomofo_to_dist
        self.unknown_bopomofo_id = unknown_bopomofo_id
        self.alpha = alpha
        self.dimsim_threshold = MAX_DIMSIM_DISTANCE
        self.detector_threshold = detector_threshold

        bopomofo_to_dist = torch.tensor(bopomofo_to_dist, dtype=torch.float32, device=detect_bert.device)
        self.dimsim_dist_query = nn.Embedding.from_pretrained(bopomofo_to_dist, freeze=True)

    def forward(self, input_ids, token_type_ids, attention_mask, bopomofo_ids):
        detect_outputs = self.detect_bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        is_detected = obtain_valid_detection_preds(detect_outputs.logits, input_ids, self.tokenizer,
                                                   threshold=self.detector_threshold)
        is_selected = (torch.tensor(is_detected, device=input_ids.device)  == 1) & (bopomofo_ids != self.unknown_bopomofo_id)

        mask_inputs_ids = torch.tensor([[self.tokenizer.mask_token_id]], device=input_ids.device)
        masked_input_ids = torch.where(is_selected, mask_inputs_ids, input_ids)

        outputs = self.mlm(
            masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        dimsim_dist = self.dimsim_dist_query(bopomofo_ids)
        INF = float('INF')
        revised_dimsim_dist = torch.where(
            dimsim_dist < self.dimsim_threshold,
            dimsim_dist,
            torch.full_like(dimsim_dist, INF)
        )
        # bert_logits + dimsim_logits = bert_logits + log(exp(-alpha * distance))
        # = bert_logits - alpha * distance
        revised_logits = outputs.logits - self.alpha * revised_dimsim_dist

        vocab_size = len(self.tokenizer.vocab)
        logits = torch.where(
            is_selected.unsqueeze(2).repeat(1, 1, vocab_size),
            revised_logits,
            nn.functional.one_hot(input_ids, num_classes=vocab_size).float()
        )

        return logits
