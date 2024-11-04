import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import pickle

from rnn import RNNEncoderDecoderLM, RNNEncoderDecoderTrainer
from tokenizer import Tokenizer
from data import load_data, read_dataframe, collate_f, preprocess, Names


DEVICE = torch.device('cuda') 

## -----------------Hyper Parameters----------------->

torch.manual_seed(42)

DATA_FOLDER_PATH = '../data'

SRC_VOCAB_SIZE = 300
TRG_VOCAB_SIZE = 300

SRC_TOK_PATH = f'../tokenizer'
TRG_TOK_PATH = f'../tokenizer'


## -----------------Loading Data----------------->
preprocess()

train_file = f'{DATA_FOLDER_PATH}/data.train.csv'
valid_file = f'{DATA_FOLDER_PATH}/data.valid.csv'

train_url = "https://docs.google.com/spreadsheets/d/1JpK9nOuZ2ctMrjNL-C0ghUQ4TesTrMER1-dTD_torAA/gviz/tq?tqx=out:csv&sheet=data.train.csv"
valid_url = "https://docs.google.com/spreadsheets/d/1cKC0WpWpIQJkaqnFb7Ou7d0syFDsj6eEW7bM7GH3u2k/gviz/tq?tqx=out:csv&sheet=data.valid.csv"

load_data(train_url, valid_url, train_file, valid_file)

train_df      = read_dataframe("train", DATA_FOLDER_PATH)
valid_df = read_dataframe("valid", DATA_FOLDER_PATH)

print(f"Length of training data: {len(train_df)}\nLength of validation data: {len(valid_df)}")


## -------------------Build Tokenizer-------------->

src_tokenizer = None
trg_tokenizer = None

if os.path.exists(os.path.join(SRC_TOK_PATH, f'srctok{SRC_VOCAB_SIZE}.pkl')):
    print('Loading Source Tokenizer')
    src_tokenizer = Tokenizer.load(os.path.join(SRC_TOK_PATH, f'srctok{SRC_VOCAB_SIZE}.pkl'))
else:
    
    print('Training Source Tokenizer')
    src_tokenizer = Tokenizer(DEVICE)
    src_tokenizer.train(list(train_df['Name']), SRC_VOCAB_SIZE)
    
    src_tokenizer.save(SRC_TOK_PATH, f'srctok{SRC_VOCAB_SIZE}')
    
    src_vocab_path = os.path.join(SRC_TOK_PATH, f'srcvocab{SRC_VOCAB_SIZE}.txt')
    with open(src_vocab_path, 'w') as vocab_file:
        for idx, byte in src_tokenizer.vocab.items():
            vocab_file.write(f"{idx}: {byte.decode('utf-8', errors='ignore')}\n") 
    
    src_merges_path = os.path.join(SRC_TOK_PATH, f'srcmerges{SRC_VOCAB_SIZE}.txt')
    with open(src_merges_path, 'w') as merges_file:
        for idx, pair in src_tokenizer.merges.items() or []:  
            merges_file.write(f"{idx}: {pair}\n")
    
    print('Source Tokenizer Training Complete and Saved')


if os.path.exists(os.path.join(TRG_TOK_PATH, f'trgtok{TRG_VOCAB_SIZE}.pkl')):
    print('Loading Target Tokenizer')
    trg_tokenizer = Tokenizer.load(os.path.join(TRG_TOK_PATH, f'trgtok{TRG_VOCAB_SIZE}.pkl'))
else:
    
    print('Training Target Tokenizer')
    trg_tokenizer = Tokenizer(DEVICE)
    trg_tokenizer.train(list(train_df['Translation']), TRG_VOCAB_SIZE)
    
    trg_tokenizer.save(TRG_TOK_PATH, f'trgtok{TRG_VOCAB_SIZE}')
    
    trg_vocab_path = os.path.join(TRG_TOK_PATH, f'trgvocab{TRG_VOCAB_SIZE}.txt')
    with open(trg_vocab_path, 'w') as vocab_file:
        for idx, byte in trg_tokenizer.vocab.items():
            vocab_file.write(f"{idx}: {byte.decode('utf-8', errors='ignore')}\n") 
    
    trg_merges_path = os.path.join(TRG_TOK_PATH, f'trgmerges{TRG_VOCAB_SIZE}.txt')
    with open(trg_merges_path, 'w') as merges_file:
        for pair, idx in trg_tokenizer.merges.items() or []:  
            merges_file.write(f"{pair}: {idx}\n")
    
    print('Target Tokenizer Training Complete and Saved')


## -------------------Dataset and Dataloader-------------->

train_dataset = Names(train_df, src_tokenizer, trg_tokenizer)
valid_dataset = Names(valid_df, src_tokenizer, trg_tokenizer)

print('\nBuilding Train Dataloader')
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True,
                                collate_fn=collate_f)
print('Building Valid Dataloader')
valid_dataLoader = DataLoader(valid_dataset, batch_size=20, shuffle=False,
                                collate_fn=collate_f)
print('Dataloader Created')


## -------------------Model Training-------------->