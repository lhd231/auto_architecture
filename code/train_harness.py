from cmath import log
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
#from models import NeuroNet, SFNC
from neuro_net import NeuroNet
import pickle

data = np.concatenate(
    (np.load('../synthetic_data/class1.npy'),
     np.load('../synthetic_data/class2.npy')))

targets = np.concatenate((
    np.zeros((1200, )),
    np.ones((1200, ))
))

M = np.zeros([6,6])
M[0, 3] = 1
M[3, 3] = 1
M[3, 5] = 1

#[(0, 3), (3, 3), (3, 5)]
L = M.flatten()
#N = M.flatten()
N = np.outer(L,L)


overlap_check = N*(1-np.eye(36))

x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, stratify=targets)
x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, shuffle=True, batch_size=128)
test_loader = DataLoader(test_ds, shuffle=False, batch_size=256)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = SFNC((60, 60), nn.Conv2d, (3, 3),
#             3, (64, 128, 256, 256, 64),
#             nn.ReLU, 1).to(device)
"""
The two dicts:
arch_dict is the dictionary of all of your possible architectures.
hp_dict is the dictionary of all of your global hyperparameters. These can be whatever you want,
but generally would be input_size, output_size, etc
test1.p has been added to the repo for you
"""
hp_dict = pickle.load(open("test1.p",'rb'))
arch_dict = hp_dict[-1]
hp_dict = hp_dict[0]
print(hp_dict["out_channels"])
model = NeuroNet(arch_dict,hp_dict).to(device)
print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)

epochs = 10
for epoch in range(epochs):
    acc = 0.0
    model.train()
    for (i, (x, y_true)) in enumerate(train_loader):
        x = x.to(device)
        y_true = y_true.to(device)
        y = (y_true >= 1).float()
        logit_pred, ix = model(x)
        logit_pred = logit_pred.squeeze()
        acc += (torch.round(torch.sigmoid(logit_pred)) == y).sum()
        loss = criterion(logit_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f'Train epoch {epoch}: {acc / x_train.size(0)}')
    model.eval()
    acc = 0.0
    for (i, (x, y_true)) in enumerate(test_loader):
        x = x.to(device)
        y_true = y_true.to(device)
        y = (y_true >= 1).float()
        with torch.no_grad():
            logit_pred, ix = model(x)
            logit_pred = logit_pred.squeeze()
            acc += (torch.round(torch.sigmoid(logit_pred)) == y).sum()
    print(f'Test epoch {epoch}: {acc / x_test.size(0)}')
torch.set_printoptions(4, sci_mode=False)
ix, attention = ix
#ix = ix * 5
#values, indices = torch.sort(attention[0], descending=True)[:5]
attention = attention.mean(0)
attention_grid = torch.zeros((36, 36)).to(device)
tril_indices = torch.tril_indices(36, 36, device=device, offset=-1)
row_ix, col_ix = tril_indices
attention_grid[row_ix, col_ix] = attention
attention_grid.T[row_ix, col_ix] = attention
attention_grid = attention_grid.cpu().numpy()
#attention_grid
print(np.sum(attention_grid*(1-np.eye(36))/np.sum(attention_grid)*overlap_check))
plt.imshow(attention_grid)
plt.savefig("atts_tst.png")