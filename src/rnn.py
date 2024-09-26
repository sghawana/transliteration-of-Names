import torch
import torch.nn as nn
from trainer import Trainer
from torch.utils.data import DataLoader


class RNNEncoderDecoderLM(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embd_dims,
                 hidden_size, device, num_layers=1, dropout=0.1,
                 bidirectional = True,
                 ):
        
        super(RNNEncoderDecoderLM, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embd_dims = embd_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.D = 2 if self.bidirectional == True else 1
        self.device = device

        self.encoder_emb = nn.Embedding(self.src_vocab_size, self.embd_dims, device=self.device)
        self.encoder_rnn = nn.GRU(self.embd_dims, self.hidden_size, num_layers=self.num_layers,
                                  dropout=self.dropout, batch_first=True,
                                  bidirectional=self.bidirectional, device=self.device)

        self.decoder_emb = nn.Embedding(self.tgt_vocab_size, self.embd_dims, device=self.device)
        self.decoder_rnn = nn.GRU(self.embd_dims, self.hidden_size, num_layers=self.num_layers,
                                  dropout=self.dropout, batch_first=True,
                                  bidirectional=self.bidirectional, device=self.device)
        self.decoder_fc = nn.Linear(self.D*hidden_size, self.tgt_vocab_size, device=self.device)


    def forward(self, inputs, decoder_inputs, decoder_hidden_state=None):
        
        if decoder_hidden_state is None:
          _, encoder_hidden_state = self.encoder_rnn(self.encoder_emb(inputs))
          decoder_hidden_state = encoder_hidden_state

        decoder_output, decoder_hidden_state = self.decoder_rnn(self.decoder_emb(decoder_inputs), decoder_hidden_state)
        decoder_output = self.decoder_fc(decoder_output)
        decoder_output = nn.LogSoftmax(dim = -1)(decoder_output)

        return decoder_output, decoder_hidden_state

    def log_probability(self, seq_x, seq_y):
        _, encoder_hidden_state = self.encoder_rnn(self.encoder_emb(seq_x))
        decoder_hidden_state = encoder_hidden_state
        log_prob = 0.0

        with torch.no_grad():
            for i in range(len(seq_y)):
                decoder_output, decoder_hidden_state = self.decoder_rnn(self.decoder_emb(seq_y[i]).unsqueeze(0), decoder_hidden_state)
                decoder_output = self.decoder_fc(decoder_output)
                decoder_output = nn.LogSoftmax(dim = -1)(decoder_output).squeeze()
                log_prob += decoder_output[seq_y[i]]
        return log_prob.item()
    

class RNNEncoderDecoderTrainer(Trainer):

    def __init__(self, directory, model, criterion, optimizer, device):
        super(RNNEncoderDecoderTrainer, self).__init__(directory, model, criterion, optimizer, device)

    @staticmethod
    def make_dataloader(dataset, shuffle_data=True, batch_size=8, collate_fn=None):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, collate_fn=collate_fn)

    def train_step(self, x_batch, y_batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = 0
        decoder_hidden_state = None

        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        for i in range(y_batch.size(1)-1):
            decoder_output, decoder_hidden_state = self.model(x_batch, y_batch[:,i].unsqueeze(1), decoder_hidden_state)
            target = y_batch[:,i+1]
            loss += self.criterion(decoder_output.squeeze(1), target)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, validation_dataloader):
        self.model.eval()
        loss = 0
        count = 0

        with torch.no_grad():
            for batch_data in validation_dataloader:
                count += 1
                x_batch, y_batch = batch_data

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                decoder_hidden_state = None

                for i in range(y_batch.size(1) - 1):
                    decoder_output, decoder_hidden_state = self.model(x_batch, y_batch[:, i].unsqueeze(1), decoder_hidden_state)
                    target = y_batch[:, i + 1]
                    loss += self.criterion(decoder_output.squeeze(1), target)

        avg_loss = loss / count
        return avg_loss.item()
