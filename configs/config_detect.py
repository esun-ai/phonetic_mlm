model_source = 'bert-base-chinese'
max_len = 512
num_workers = 2
batch_size = 32

# for training
manual_seed = 1313
exp_name = 'bert_detection'
train_json = 'data/train.json'
valid_json = 'data/valid.json'
lr = 5e-5
val_interval = 100
num_iter = 10000
