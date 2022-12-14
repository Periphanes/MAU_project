from concurrent.futures import wait
import numpy as np

from os import path
import matplotlib.pyplot as plt

#scikit-learn related imports
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# pytorch relates imports
import torch
import torch.nn as nn
import torch.optim as optim

import captum

# imports from captum library
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

import pickle
from matplotlib import test
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from yaml import parse

from models.LSTM import LSTM
from models.singleTrans import TransformerModel
from models.singleGRU import GRU


pickle_dir = 'C:\MAU_dataset\pickle\ '

with open(pickle_dir.strip() + 'finalDataKR.pickle', 'rb') as handle:
    finalData = pickle.load(handle)

with open(pickle_dir.strip() + 'finalDescriptionKR.pickle', 'rb') as handle:
    finalDescription = pickle.load(handle)


parser = argparse.ArgumentParser()
parser.add_argument('--input-size', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--target', type=str, default="ratio", choices=["ratio", "top1", "top10", "bot50"])
parser.add_argument('--model', type=str, default="trans", choices=["trans", "lstm", "gru"])
parser.add_argument('--prediction-years', type=int, default=1)
parser.add_argument('--train-years', type=int, default=15)

# Transformer Hyper Parameter
parser.add_argument('--trans-head', type=int, default=2)
parser.add_argument('--trans-layer', type=int, default=1)
parser.add_argument('--trans-dropout', type=float, default=0.1)
parser.add_argument('--trans-dim', type=int, default=128)

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
        0.2532, 0.2534, 0.2568, 0.2565, 0.2584, 0.2617, 0.2572, 0.2527, 0.257] * 100

# Wealth Share of national top 10%
top_10 = [0.5694, 0.574, 0.5732, 0.5671, 0.5701, 0.5723, 0.574, 0.5756, 0.5767, 0.5774, \
        0.5749, 0.5862, 0.5881, 0.5894, 0.5883, 0.5899, 0.5906, 0.5897, 0.59, 0.5902, \
        0.5925, 0.5927, 0.5938, 0.5955, 0.5931, 0.5906, 0.593] * 100

# Wealth share of national bottom 50%
bottom_50 = [0.05, 0.0494, 0.0495, 0.0503, 0.0499, 0.0496, 0.0494, 0.0492, 0.049, 0.0489, \
            0.0493, 0.0477, 0.0475, 0.0473, 0.0474, 0.0472, 0.0471, 0.0472, 0.0472, 0.0472, \
            0.0469, 0.0468, 0.0467, 0.0465, 0.0468, 0.0471, 0.0468] * 100

if args.target == "ratio":
    y = wealth_ratio
elif args.target == "top1":
    y = top_1
elif args.target == "top10":
    y = top_10
elif args.target == "bot50":
    y = bottom_50


train_loss = []
test_loss = []

final_loss = []

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

    train_loss.append(avg_loss)

def test_model(data_loader, model, criterion):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            #print(X.shape)
            output = model(X)
            total_loss += criterion(output, y).item()

    avg_loss = total_loss / num_batches

    test_loss.append(avg_loss)

def final_test_model(data_loader, model, criterion):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += criterion(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"{avg_loss}")

    test_loss.append(avg_loss)
    final_loss.append(avg_loss)

class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        # 1995 ~ 2015??? ?????? -> length of 21
        return self.x.shape[-1] - 14
    
    def __getitem__(self, i):
        x = self.x[:, i:i+15]
        y = self.y[i:i+args.prediction_years]
        return x,y

final_loss = []

args.prediction_years = 1
pred_years = args.prediction_years
tr_years = args.train_years

start = 1995
end = min(2021 - (pred_years - 1), 2015)

possible_years = end - start + 1

test_years = possible_years // 5
train_years = possible_years - test_years

args.test_years = test_years

# print(pred_years, test_years, train_years)

train_x = torch.FloatTensor(finalData[:, :(1994 - 1980) + possible_years])
train_y = torch.FloatTensor(np.array(y[:possible_years + pred_years - 1]))

test_x = torch.FloatTensor(finalData[:, 21 - test_years:])
test_y = torch.FloatTensor(np.array(y[train_years:]))

final_x = torch.FloatTensor(finalData[:, -15:])
final_y = torch.FloatTensor(np.array(y))

print(test_x.shape)
print(train_x.shape)

# print(test_years, test_x.shape, possible_years)

train_dataset = SequenceDataset(train_x, train_y)
test_dataset = SequenceDataset(test_x, test_y)
final_dataset = SequenceDataset(final_x, final_y)

if args.model == "trans":
    model = TransformerModel(args)
elif args.model == "lstm":
    model = LSTM(args)
elif args.model == "gru":
    model = GRU(args)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

loss_func = torch.nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# print(test_dataset.__len__(), test_dataset.y)

for epoch in tqdm(range(args.epochs)):
    train_model(train_loader, model, loss_func, optimizer=optimizer)
    test_model(test_loader, model, loss_func)

final_test_model(test_loader, model, loss_func)

print("\n\n")

final_loader = DataLoader(final_dataset, batch_size=args.batch_size, shuffle=False)

# model.eval()
# with torch.no_grad():
#     for X, y in final_loader:
#         output = model(X)
#         print(output)


ig = IntegratedGradients(model)
ig_nt = NoiseTunnel(ig)
dl = DeepLift(model)
gs = GradientShap(model)
fa = FeatureAblation(model)

test_x = test_x[:,:-3].unsqueeze(0)
train_x = train_x[:,:-20].unsqueeze(0)

print("\nXAI\n")

ig_attr_test = ig.attribute(test_x, n_steps=50)
# print("IG_ATTR_TEST", ig_attr_test)
# print(torch.count_nonzero(ig_attr_test))
# print(ig_attr_test.shape)

# wait(1)

ig_nt_attr_test = ig_nt.attribute(test_x)
print("IG_NT_ATTR_TEST", ig_nt_attr_test)
print(torch.count_nonzero(ig_nt_attr_test))
print(ig_nt_attr_test.shape)

dl_attr_test = dl.attribute(test_x)
print("DL_ATTR_TEST", dl_attr_test)
print(torch.count_nonzero(dl_attr_test))
print(dl_attr_test.shape)

gs_attr_test = gs.attribute(test_x, train_x)
print("GS_ATTR_TEST", gs_attr_test)
print(torch.count_nonzero(gs_attr_test))
print(gs_attr_test.shape)

with open(pickle_dir.strip() + 'ig_nt_attr.pickle', 'wb') as handle:
    pickle.dump(ig_nt_attr_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(pickle_dir.strip() + 'gs_attr.pickle', 'wb') as handle:
    pickle.dump(gs_attr_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


# fa_attr_test = fa.attribute(test_x)
# print("FA_ATTR_TEST", fa_attr_test)
# print(torch.count_nonzero(fa_attr_test))
# print(fa_attr_test.shape)