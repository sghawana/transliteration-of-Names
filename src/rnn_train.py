import torch
import torch.nn as nn
import os

from rnn import RNNEncoderDecoderLM, RNNEncoderDecoderTrainer
from tokenizer import TokenizerDataset, Tokenizer
from data import preprocess, load_data, read_dataframe


DIRECTORY_NAME = 'runs'
DEVICE = torch.device('cuda') 

torch.manual_seed(42)

## -----------------Loading Data----------------->

train_file = 'data/data.train.csv'
valid_file = 'data/data.valid.csv'

train_url = "https://docs.google.com/spreadsheets/d/1JpK9nOuZ2ctMrjNL-C0ghUQ4TesTrMER1-dTD_torAA/gviz/tq?tqx=out:csv&sheet=data.train.csv"
valid_url = "https://docs.google.com/spreadsheets/d/1cKC0WpWpIQJkaqnFb7Ou7d0syFDsj6eEW7bM7GH3u2k/gviz/tq?tqx=out:csv&sheet=data.valid.csv"

load_data(train_url, valid_url, train_file, valid_file)

train_data      = read_dataframe("train")
validation_data = read_dataframe("valid")

print(f"Length of training data: {len(train_data)}\nLength of validation data: {len(validation_data)}")



## -------------------Build Tokenizer-------------->

print('\n Training Tokenizer')
src_tokenizer = Tokenizer(DEVICE)
tgt_tokenizer = Tokenizer(DEVICE)

SRC_VOCAB_SIZE = 260
TGT_VOCAB_SIZE = 260

src_tokenizer.train(list(train_data['Name']),SRC_VOCAB_SIZE)
tgt_tokenizer.train(list(train_data['Translation']),TGT_VOCAB_SIZE)

print('\n Tokenizer training complete')


## ----------------------- RNN Training----------------->
print('\n Training RNN')

EMBEDDING_DIMENSION = 300
HIDDEN_DIMENSION = 500

DROPOUT = 0
NUM_LAYERS = 4
BIDIRECTIONAL = True


rnn_enc_dec_params = {
    'src_vocab_size': SRC_VOCAB_SIZE,
    'tgt_vocab_size': TGT_VOCAB_SIZE,
    'embd_dims'     : EMBEDDING_DIMENSION,
    'hidden_size'   : HIDDEN_DIMENSION,
    'dropout'       : DROPOUT,
    'num_layers'    : NUM_LAYERS,
    'device'        : DEVICE,
    'bidirectional' : BIDIRECTIONAL
}

model = RNNEncoderDecoderLM(**rnn_enc_dec_params)

OPTIMIZER = torch.optim.Adam(model.parameters(), lr=0.001)
CRITERION = nn.NLLLoss()

NUM_EPOCHS = 8
BATCH_SIZE = 60
SAVE_STEPS = 150
EVAL_STEPS = 20


rnn_enc_dec_training_params = dict(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
)

trainer = RNNEncoderDecoderTrainer(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec"),
    model, CRITERION, OPTIMIZER, DEVICE
)

train_dataset      = TokenizerDataset(train_data, src_tokenizer, tgt_tokenizer)
validation_dataset = TokenizerDataset(validation_data, src_tokenizer, tgt_tokenizer)

rnn_enc_dec_train_data = dict(
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    collate_fn=train_dataset.collate
)

trainer.train(**rnn_enc_dec_train_data, **rnn_enc_dec_training_params)
