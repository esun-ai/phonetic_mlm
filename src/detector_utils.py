import re

import numpy as np

from utils import CHINESE_RE


def obtain_valid_detection_preds(detector_logits, input_ids, tokenizer, threshold=0.5):
    input_ids = input_ids.cpu().tolist()
    # (logit_positive - logit_negative) >= logit_threshold
    logit_threshold = np.log(threshold / (1 - threshold))
    preds = ((detector_logits[:,:,1] - detector_logits[:,:,0]) >= logit_threshold).int().cpu().tolist()

    skip_token_ids = set([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id])
    revised_preds = []
    for input_id_seq, pred_seq in zip(input_ids, preds):
        pred = []
        reach_end = False
        for token_id, is_detected in zip(input_id_seq, pred_seq):
            if reach_end:
                pred.append(0)
                continue
            if token_id == tokenizer.sep_token_id:
                reach_end = True
            if is_detected == 1:
                token = tokenizer.convert_ids_to_tokens(token_id)
                if token_id in skip_token_ids:
                    is_detected = 0
                elif len(token) > 1:
                    is_detected = 0
                elif not re.match(CHINESE_RE, token):  # allow only chinese
                    is_detected = 0
            pred.append(is_detected)
        revised_preds.append(pred)
    return revised_preds
