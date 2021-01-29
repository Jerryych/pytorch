from linear_reg import Linear_Regression, CNN
from pathlib import Path
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import gzip


def get_model(m='linear', lr=0.1):
    if m == 'lr':
        model = Linear_Regression()
    else:
        model = CNN()

    return model, optim.SGD(model.parameters(), lr=lr, momentum=0.9)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs),
    )

def fit(dev, epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.view(-1, 1, 28, 28).to(dev)
            yb = yb.to(dev)
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()
    
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb.view(-1, 1, 28, 28).to(dev)), yb.to(dev)) for xb, yb in valid_dl)
        
        print(epoch, valid_loss / len(valid_dl))


DATA_PATH = Path('data')
PATH = DATA_PATH / 'mnist'
FILE_NAME = 'mnist.pkl.gz'

with gzip.open((PATH / FILE_NAME).as_posix(), 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

lr = 0.1
epochs = 15
bs = 64
loss_func = F.cross_entropy

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model('cnn', lr=lr)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', dev)
print()

if dev.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
    model.to(dev)
    print()
fit(dev, epochs, model, loss_func, opt, train_dl, valid_dl)