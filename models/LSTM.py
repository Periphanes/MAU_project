import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class LSTM(torch.nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.input_size = 15
        self.hidden_size = 16
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, \
                            num_layers = self.num_layers, batch_first = True)
        
        self.relu = nn.ReLU()
        self.regresser = nn.Linear(self.hidden_size, 1)
    
    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        
        output = output.squeeze()
        output = output[-1,:]

        regress = self.regresser(output)
        return regress