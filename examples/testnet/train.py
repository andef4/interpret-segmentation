import os
import inspect
import torch
from torch import nn
from torch import optim
from pathlib import Path

from dataset import load_dataset
from training_loop import train_model
from unet import UNet

path = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = load_dataset(path / 'dataset', 10)

model = UNet(in_channels=1, out_channels=1)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

train_model(
    path,
    'testnet',
    model,
    {'train': train_loader, 'val': test_loader},
    criterion,
    optimizer,
    device,
    num_epochs=1000
)
