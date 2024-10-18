import os
import pickle
import torch
from torch.utils.data import TensorDataset
from tok_utils import merge_ids, generate_merges

class Tokenizer:
    def __init__(self, device):
        self.vocab = {idx : bytes([idx]) for idx in range(256)}
        self.special = None
        self.merges = None
        self.device = device
        self.isTrain = False
        
        
    @classmethod
    def load(cls, path):
        tokenizer_file = os.path.join(path, "tokenizer.pkl")

        if not os.path.exists(path) or not os.path.exists(os.path.join(path, "tokenizer.pkl")):
            raise ValueError(cls.load.__name__ + ": No tokenizer found at the specified directory")

        with open(tokenizer_file, "rb") as ifile:
            return pickle.load(ifile)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.pkl"), 'wb') as ofile:
            pickle.dump(self, ofile)

    def train(self, names: list[str], vocab_size) -> None: 
        id_list = [byte for name in names for byte in name.encode('utf-8', errors='replace')]
        ids = torch.tensor(id_list, dtype=torch.int32, device=self.device)

        num_merges = max(0, vocab_size - 256)
        self.merges = generate_merges(ids, num_merges)
        for pair, new_idx in self.merges.items():
            self.vocab[new_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.isTrain = True
        
        next_idx = max(self.vocab.keys()) + 1
        self.special = {
            '[PAD]': next_idx,
            '[SOS]': next_idx + 1,
            '[EOS]': next_idx + 2,
            '[UNK]': next_idx + 3
        }
        
        self.isTrain = True
      
    def encode(self, string, use_sos=True, use_eos=True, seq_len=None) -> torch.Tensor: 
        if self.isTrain == False: print('Warning: Tokenizer is not Trained!!!')        
        ids = torch.tensor(list(string.encode("utf-8", errors='replace')), dtype=torch.int32, device=self.device)
        for pair, target in self.merges.items():
            ids = merge_ids(ids, pair, target)
            
        if seq_len is not None:
            if len(ids) > seq_len:
                ids = ids[:seq_len]
                
        if use_sos:
            sos_token = torch.tensor([self.special['[SOS]']], dtype=torch.int32, device=self.device)
            ids = torch.cat((sos_token, ids))
        if use_eos:
            eos_token = torch.tensor([self.special['[EOS]']], dtype=torch.int32, device=self.device)
            ids = torch.cat((ids, eos_token))
        return ids

    def decode(self, tokens: torch.Tensor, strip_special=True) -> str:
        if not self.isTrain: print('Warning: Tokenizer is not Trained!!!') 

        byte_sequence = bytearray()
        for token in tokens:
            token_id = token.item()
            if token_id in {self.special['[PAD]'], self.special['[SOS]'], self.special['[EOS]']}:
                if not strip_special:
                    special_token = list(self.special.keys())[list(self.special.values()).index(token_id)]
                    byte_sequence += special_token.encode('utf-8')
            elif token_id == self.special['[UNK]']:
                byte_sequence += b'[UNK]'
            else:
                byte_sequence += self.vocab[token_id]
        return byte_sequence.decode('utf-8', errors='replace')


    def batch_encode(self, batch: list[str], use_sos=True, use_eos=True):
        encoded_batch = []
        for string in batch:
            encoded = self.encode(string, use_sos=use_sos, use_eos=use_eos)
            encoded_batch.append(encoded)
            
        padded_batch = torch.nn.utils.rnn.pad_sequence(
            encoded_batch, 
            batch_first=True, 
            padding_value=self.special['[PAD]']
        )
        return padded_batch
        

    def batch_decode(self, batch: torch.Tensor, strip_special: bool = True):
        decoded_batch = []
        for sequence in batch:
            decoded = self.decode(sequence, strip_special=strip_special)
            decoded_batch.append(decoded)
        return decoded_batch




class TokenizerDataset(TensorDataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def collate(self, batch):
        x_batch = [ data[0] for data in batch ]
        y_batch = [ data[1] for data in batch ]

        x_batch = self.src_tokenizer.batch_encode(x_batch)
        y_batch = self.tgt_tokenizer.batch_encode(y_batch)

        return x_batch, y_batch

    def __getitem__(self, index):
        return (
            self.data['Name'][index],
            self.data['Translation'][index]
        )

    def __len__(self):
        return len(self.data)