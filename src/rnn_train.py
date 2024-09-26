import torch
import torch.nn as nn
import os

from rnn import RNNEncoderDecoderLM, RNNEncoderDecoderTrainer
from tokenizer import TokenizerDataset, Tokenizer
from data import train_data, validation_data


DIRECTORY_NAME = 'runs'
DEVICE = torch.device('cuda') 

src_tokenizer = Tokenizer(DEVICE)
tgt_tokenizer = Tokenizer(DEVICE)


SRC_VOCAB_SIZE = 260
TGT_VOCAB_SIZE = 260

src_tokenizer.train(list(train_data['Name']),SRC_VOCAB_SIZE)
tgt_tokenizer.train(list(train_data['Translation']),TGT_VOCAB_SIZE)


rnn_enc_dec_params = {
    'src_vocab_size': 260,
    'tgt_vocab_size': 260,
    'embd_dims'     : 300,
    'hidden_size'   : 500,
    'dropout'       : 0,
    'num_layers'    : 4,
    'device'        : DEVICE
}

rnn_enc_dec_data_params = dict(
    src_padding=None,
    tgt_padding=None,
)


rnn_enc_dec_training_params = dict(
    num_epochs=8,
    batch_size=60,
    shuffle=True,
    save_steps=150,
    eval_steps=20,
)


torch.manual_seed(42)

model = RNNEncoderDecoderLM(**rnn_enc_dec_params)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

trainer = RNNEncoderDecoderTrainer(
    os.path.join(DIRECTORY_NAME, "rnn.enc-dec"),
    model, criterion, optimizer, DEVICE
)


train_dataset      = TokenizerDataset(train_data     , src_tokenizer, tgt_tokenizer, **rnn_enc_dec_data_params)
validation_dataset = TokenizerDataset(validation_data, src_tokenizer, tgt_tokenizer, **rnn_enc_dec_data_params)

rnn_enc_dec_train_data = dict(
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    collate_fn=train_dataset.collate
)

trainer.train(**rnn_enc_dec_train_data, **rnn_enc_dec_training_params)
