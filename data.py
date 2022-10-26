import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm

from models.LSTM import LSTM
from models.Ttrans import TransformerModel
from models.GRU import GRU


pickle_dir = 'C:\MAU_dataset\pickle\ '

with open(pickle_dir.strip() + 'finalDataKR.pickle', 'rb') as handle:
    finalData = pickle.load(handle)

with open(pickle_dir.strip() + 'finalDescriptionKR.pickle', 'rb') as handle:
    finalDescription = pickle.load(handle)


parser = argparse.ArgumentParser()
parser.add_argument('--input-size', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--target', type=str, default="ratio", choices=["ratio", "top1", "top10", "bot50"])

args = parser.parse_args()

# Net national wealth to Net National Income Ratio (1995 ~ 2021) [~26]
# 26 2021, 25 2020, 24 2019, 23 2018, 22 2017, 2015 20 // train: 0 ~ 15, :16
wealth_ratio = [5.770171165, 5.975523472, 5.884808064, 6.399320602, 6.005833626,  \
                5.787640572, 5.750982761, 5.695618153, 6.018857956, 6.218288422,  \
                6.702404022, 7.152865887, 7.389669895, 7.569231033, 7.730065346,  \
                7.533388615, 7.565679073, 7.647249699, 7.637476921,7.681171417,   \
                7.638151646, 7.695356369, 7.775409698, 7.995022297,8.319354057, 8.814061165, 8.834832191]

# Wealth Share of national top 1%
top_1 = [0.2355, 0.2392, 0.2364, 0.2293, 0.2321, 0.2342, 0.2359, 0.2374, 0.2385, \
        0.2392, 0.2367, 0.2482, 0.2501, 0.2514, 0.2504, 0.252, 0.2529, 0.2524,  \
        0.2532, 0.2534, 0.2568, 0.2565, 0.2584, 0.2617, 0.2572, 0.2527, 0.257]

# Wealth Share of national top 10%
top_10 = [0.5694, 0.574, 0.5732, 0.5671, 0.5701, 0.5723, 0.574, 0.5756, 0.5767, 0.5774, \
        0.5749, 0.5862, 0.5881, 0.5894, 0.5883, 0.5899, 0.5906, 0.5897, 0.59, 0.5902, \
        0.5925, 0.5927, 0.5938, 0.5955, 0.5931, 0.5906, 0.593]

# Wealth share of national bottom 50%
bottom_50 = [0.05, 0.0494, 0.0495, 0.0503, 0.0499, 0.0496, 0.0494, 0.0492, 0.049, 0.0489, \
            0.0493, 0.0477, 0.0475, 0.0473, 0.0474, 0.0472, 0.0471, 0.0472, 0.0472, 0.0472, \
            0.0469, 0.0468, 0.0467, 0.0465, 0.0468, 0.0471, 0.0468]

if args.target == "ratio":
    y = wealth_ratio
elif args.target == "top1":
    y = top_1
elif args.target == "top10":
    y = top_10
elif args.target == "bot50":
    y = bottom_50

train_x = torch.FloatTensor(finalData[:, :30])
train_y = torch.FloatTensor(np.array(y[:16]))

test_x = torch.FloatTensor(finalData[:, 16:])
test_y = torch.FloatTensor(np.array(y[16:21]))

print("Train X :", train_x.shape)
print("Train Y :", train_y.shape)

print("Test X :", test_x.shape)
print("Test Y", test_y.shape)

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

lstmNet = GRU(args)
optimizer = torch.optim.Adam(lstmNet.parameters(), lr=1e-2)
loss_func = torch.nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

train_loss = []
test_loss = []


def train_model(data_loader, model, criterion, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in tqdm(data_loader):
        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches

    train_loss.append(avg_loss)

    print(f"Train Loss: {avg_loss}")

def test_model(data_loader, model, criterion):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(data_loader):
            output = model(X)
            total_loss += criterion(output, y).item()

    avg_loss = total_loss / num_batches

    test_loss.append(avg_loss)

    print(f"Test Loss: {avg_loss}")

print("Untrained Test \n----------")
test_model(test_loader, lstmNet, loss_func)
print("")


for epoch in range(args.epochs):
    print(f"Epoch {epoch}\n ---------")
    train_model(train_loader, lstmNet, loss_func, optimizer=optimizer)
    test_model(test_loader, lstmNet, loss_func)
    print("")

# print("\n\nTrain Losses Array:")
# for l in train_loss:
#     print(l)

# print("\n\nTest Losses Array:")
# for l in test_loss:
#     print(l)
