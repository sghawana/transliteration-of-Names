import os
import requests
import subprocess
import matplotlib
import pandas as pd


result = subprocess.run([ 'fc-list', ':lang=hi', 'family' ], capture_output=True)
found_hindi_fonts = result.stdout.decode('utf-8').strip().split('\n')

print('Found Hindi Fonts:\n', found_hindi_fonts)

matplotlib.rcParams['font.sans-serif'] = [
    'Source Han Sans TW', 'sans-serif', 'Arial Unicode MS',
    *found_hindi_fonts
]

os.makedirs('data', exist_ok=True)

train_url = "https://docs.google.com/spreadsheets/d/1JpK9nOuZ2ctMrjNL-C0ghUQ4TesTrMER1-dTD_torAA/gviz/tq?tqx=out:csv&sheet=data.train.csv"
valid_url = "https://docs.google.com/spreadsheets/d/1cKC0WpWpIQJkaqnFb7Ou7d0syFDsj6eEW7bM7GH3u2k/gviz/tq?tqx=out:csv&sheet=data.valid.csv"

train_file = 'data/data.train.csv'
valid_file = 'data/data.valid.csv'

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

train_data      = read_dataframe("train")
validation_data = read_dataframe("valid")

print(f"Length of training data: {len(train_data)}\nLength of validation data: {len(validation_data)}")