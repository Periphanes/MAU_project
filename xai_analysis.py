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

pickle_dir = 'C:\MAU_dataset\pickle\ '

with open(pickle_dir.strip() + 'finalDataKR.pickle', 'rb') as handle:
    finalData = pickle.load(handle)

with open(pickle_dir.strip() + 'finalDescriptionKR.pickle', 'rb') as handle:
    finalDescription = pickle.load(handle)

with open(pickle_dir.strip() + 'ig_nt_attr.pickle', 'rb') as handle:
    ig_nt = pickle.load(handle)

with open(pickle_dir.strip() + 'gs_attr.pickle', 'rb') as handle:
    gs = pickle.load(handle)

ig_sum = torch.sum(ig_nt.squeeze(), dim=1)
print(ig_sum.shape)
print(torch.count_nonzero(ig_sum))

gs_sum = torch.sum(gs.squeeze(), dim=1)
print(gs_sum.shape)
print(torch.count_nonzero(gs_sum))


ig = torch.abs(ig_sum)
gs = torch.abs(gs_sum)

ig = (ig / ig.max()) * 100
gs = (gs / gs.max()) * 100

for i in range(18):
    print(ig[100*i:100*(i+1)])