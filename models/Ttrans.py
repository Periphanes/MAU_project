import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
from typing import Tuple

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(torch.nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_heads = 2
        self.dropout = 0.1
        self.num_layers = 1
        self.model_dim = 128
        self.input_dim = 1882

        self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)
        self.encoder_layers = TransformerEncoderLayer(self.model_dim, self.num_heads, self.model_dim, self.dropout)
        self.trasnformer_encoder = TransformerEncoder(self.encoder_layers, self.num_layers)

        self.cls_token = torch.rand((1,1,128))

        self.init_fc = nn.Linear(self.input_dim, self.model_dim)

        # self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, \
        #                     num_layers = self.num_layers, batch_first = True)
        
        self.relu = nn.ReLU()
        self.regresser = nn.Linear(self.model_dim, 1)
    
    def forward(self, x):
        # print(x.shape)
        embedding = []
        for i in range(len(x[0][0])):
            embedding.append(self.init_fc(x[:,:,i]))
        # embedding = self.init_fc(x)
        embedding = torch.cat(embedding)
        #embedding = embedding.swapaxes(0,1)

        embedding = embedding.unsqueeze(0)
        # print(embedding.shape)
        input_pos = self.pos_encoder(embedding)
        # print(self.cls_token.shape)
        final_inp = torch.cat((self.cls_token, input_pos), axis=1)
        # print(final_inp.shape)

        output = self.trasnformer_encoder(final_inp)

        cls_output = output[:,0,:]

        regress = self.regresser(cls_output)
        # print(regress.shape)
        return regress