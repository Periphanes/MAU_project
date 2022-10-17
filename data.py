import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm

pickle_dir = 'C:\MAU_dataset\pickle\ '

with open(pickle_dir.strip() + 'finalDataKR.pickle', 'rb') as handle:
    finalData = pickle.load(handle)

with open(pickle_dir.strip() + 'finalDescriptionKR.pickle', 'rb') as handle:
    finalDescription = pickle.load(handle)

# Net national wealth to Net National Income Ratio (1995 ~ 2021) [~26]
# 26 2021, 25 2020, 24 2019, 23 2018, 22 2017, 2015 20 // train: 0 ~ 15, :16
wealth_ratio = [5.770171165, 5.975523472, 5.884808064, 6.399320602, 6.005833626,  \
                5.787640572, 5.750982761, 5.695618153, 6.018857956, 6.218288422,  \
                6.702404022, 7.152865887, 7.389669895, 7.569231033, 7.730065346,  \
                7.533388615, 7.565679073, 7.647249699, 7.637476921,7.681171417,   \
                7.638151646, 7.695356369, 7.775409698, 7.995022297,8.319354057, 8.814061165, 8.834832191]

train_x = torch.FloatTensor(finalData[:, :30])
train_y = torch.FloatTensor(np.array(wealth_ratio[:16]))

test_x = torch.FloatTensor(finalData[:, 16:])
test_y = torch.FloatTensor(np.array(wealth_ratio[16:21]))

print("Train X :", train_x.shape)
print("Train Y :", train_y.shape)

print("Test X :", test_x.shape)
print("Test Y", test_y.shape)


class LSTM(torch.nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.input_size = 15
        self.hidden_size = 2056
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

class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        # 1995 ~ 2015년 예측 -> length of 21
        return self.y.shape[0]
    
    def __getitem__(self, i):
        x = self.x[:, i:i+15]
        y = self.y[i]
        return x,y

train_dataset = SequenceDataset(train_x, train_y)
test_dataset = SequenceDataset(test_x, test_y)

# x, y = train_dataset[0]
# print(x)
# print(y)

parser = argparse.ArgumentParser()
parser.add_argument('--input-size', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()

lstmNet = LSTM(args)
optimizer = torch.optim.Adam(lstmNet.parameters(), lr=5e-5)
loss_func = torch.nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

def train_model(data_loader, model, criterion, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches

    print(f"Train Loss: {avg_loss}")

def test_model(data_loader, model, criterion):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += criterion(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test Loss: {avg_loss}")

print("Untrained Test \n----------")
test_model(test_loader, lstmNet, loss_func)
print("")


for epoch in range(args.epochs):
    print(f"Epoch {epoch}\n ---------")
    train_model(train_loader, lstmNet, loss_func, optimizer=optimizer)
    test_model(test_loader, lstmNet, loss_func, optimizer=optimizer)
    print("")