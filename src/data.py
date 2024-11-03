import os
import requests
import subprocess
import matplotlib
import pandas as pd
import torch.utils
from torch.utils.data import Dataset, DataLoader

from tokenizer import*


def preprocess():
    result = subprocess.run([ 'fc-list', ':lang=hi', 'family' ], capture_output=True)
    found_hindi_fonts = result.stdout.decode('utf-8').strip().split('\n')

    print('Found Hindi Fonts:\n', found_hindi_fonts)

    matplotlib.rcParams['font.sans-serif'] = [
        'Source Han Sans TW', 'sans-serif', 'Arial Unicode MS',
        *found_hindi_fonts
    ]
    os.makedirs('data', exist_ok=True)


def load_data(train_url, valid_url, train_file, valid_file):
    with requests.get(train_url) as response:
        with open(train_file, 'wb') as f:
            f.write(response.content)
    print(f'Train data saved to {train_file}')

    with requests.get(valid_url) as response:
        with open(valid_file, 'wb') as f:
            f.write(response.content)
    print(f'Validation data saved to {valid_file}')


def read_dataframe(ds_type, path):
    df = pd.read_csv(f"{path}/data.{ds_type}.csv", header=0)
    df = df[~df.isna()]
    df['Name'] = df['Name'].astype(str)
    df['Translation'] = df['Translation'].astype(str)
    return df


class Names(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer):
        super(Names, self).__init__()
        
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
    def transform(self, name, translation):
        return srctok.encode(name), trgtok.encode(translation)

    def __getitem__(self, index):
        return  self.transform(self.data['Name'][index],
                               self.data['Translation'][index])
    
    def __len__(self):
        return len(self.data)
    

def collate_f(batch):
    
    name_list = []
    trans_list = []
    
    for names, trans in batch:
        name_list.append(names)
        trans_list.append(trans)

    return (torch.nn.utils.rnn.pad_sequence(names),
            torch.nn.utils.rnn.pad_sequence(trans))


if __name__ == '__main__':
    
    srctok = Tokenizer.load('../models/srctok1000.pkl')
    trgtok = Tokenizer.load('../models/trgtok3000.pkl')
    
    DATA_FOLDER_PATH = '../data'
    
    train_file = f'{DATA_FOLDER_PATH}/data.train.csv'
    valid_file = f'{DATA_FOLDER_PATH}/data.valid.csv'

    train_url = "https://docs.google.com/spreadsheets/d/1JpK9nOuZ2ctMrjNL-C0ghUQ4TesTrMER1-dTD_torAA/gviz/tq?tqx=out:csv&sheet=data.train.csv"
    valid_url = "https://docs.google.com/spreadsheets/d/1cKC0WpWpIQJkaqnFb7Ou7d0syFDsj6eEW7bM7GH3u2k/gviz/tq?tqx=out:csv&sheet=data.valid.csv"

    load_data(train_url, valid_url, train_file, valid_file)

    train_df = read_dataframe("train", DATA_FOLDER_PATH)
    valid_df = read_dataframe("valid", DATA_FOLDER_PATH)
    
    
    train_dataset = Names(train_df, srctok, trgtok)
    valid_dataset = Names(valid_df, srctok, trgtok)
    
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True,
                                  collate_fn=collate_f)
    valid_dataLoader = DataLoader(valid_dataset, batch_size=20, shuffle=False,
                                  collate_fn=collate_f)
    
    
    count = 0
    for _ in train_dataloader:
        count += 1
    print(count)
    print(len(train_df))