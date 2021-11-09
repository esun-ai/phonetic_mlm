model_source = '/datasets/data-nfs-if-fin-brain/yc/g2p_chinese/pretrained_models/bert-base-chinese'
max_len = 512
num_workers = 2
batch_size = 32

# for training
manual_seed = 1313
exp_name = 'bert_detection'
train_json = 'data/esun_record_asr_train.json'
valid_json = 'data/esun_record_asr_valid.json'
lr = 5e-5
val_interval = 100
num_iter = 10000
