import torch
import torch.nn as nn
import os

from rnn import RNNEncoderDecoderLM, RNNEncoderDecoderTrainer
from tokenizer import Tokenizer
from data import load_data, read_dataframe

DEVICE = torch.device('cuda') 

torch.manual_seed(42)

DATA_FOLDER_PATH = '../data'


## -----------------Loading Data----------------->

train_file = f'{DATA_FOLDER_PATH}/data.train.csv'
valid_file = f'{DATA_FOLDER_PATH}/data.valid.csv'

train_url = "https://docs.google.com/spreadsheets/d/1JpK9nOuZ2ctMrjNL-C0ghUQ4TesTrMER1-dTD_torAA/gviz/tq?tqx=out:csv&sheet=data.train.csv"
valid_url = "https://docs.google.com/spreadsheets/d/1cKC0WpWpIQJkaqnFb7Ou7d0syFDsj6eEW7bM7GH3u2k/gviz/tq?tqx=out:csv&sheet=data.valid.csv"

load_data(train_url, valid_url, train_file, valid_file)

train_data      = read_dataframe("train", DATA_FOLDER_PATH)
validation_data = read_dataframe("valid", DATA_FOLDER_PATH)

print(f"Length of training data: {len(train_data)}\nLength of validation data: {len(validation_data)}")



## -------------------Build Tokenizer-------------->

SRC_VOCAB_SIZE = 1000
TRG_VOCAB_SIZE = 3000

SRC_TOK_PATH = f'../models'
TRG_TOK_PATH = f'../models'


src_tokenizer = None
tgt_tokenizer = None


if os.path.exists(os.path.join(SRC_TOK_PATH, f'srctok{SRC_VOCAB_SIZE}.pkl')):
    print('Loading Source Tokenizer')
    src_tokenizer = Tokenizer.load(os.path.join(SRC_TOK_PATH, f'srctok{SRC_VOCAB_SIZE}.pkl'))
else:
    print('Training Source Tokenizer')
    src_tokenizer = Tokenizer(DEVICE)
    src_tokenizer.train(list(train_data['Name']), SRC_VOCAB_SIZE)
    src_tokenizer.save(SRC_TOK_PATH, f'srctok{SRC_VOCAB_SIZE}')
    print('Source Tokenizer Training Complete and Saved')


if os.path.exists(os.path.join(TRG_TOK_PATH, f'trgtok{TRG_VOCAB_SIZE}.pkl')):
    print('Loading Target Tokenizer')
    tgt_tokenizer = Tokenizer.load(os.path.join(TRG_TOK_PATH, f'trgtok{TRG_VOCAB_SIZE}.pkl'))
else:
    print('Training Target Tokenizer')
    tgt_tokenizer = Tokenizer(DEVICE)
    tgt_tokenizer.train(list(train_data['Translation']), TRG_VOCAB_SIZE)
    tgt_tokenizer.save(TRG_TOK_PATH, f'trgtok{TRG_VOCAB_SIZE}')
    print('Target Tokenizer Training Complete and Saved')


## ----------------------- RNN Training----------------->

