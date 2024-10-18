import os
import requests
import subprocess
import matplotlib
import pandas as pd


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


def read_dataframe(ds_type):
    df = pd.read_csv(f"data/data.{ds_type}.csv", header=0)
    df = df[~df.isna()]
    df['Name'] = df['Name'].astype(str)
    df['Translation'] = df['Translation'].astype(str)
    return df
