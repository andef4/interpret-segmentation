import torch
from torch import nn
from torch import optim

from dataset import load_dataset
from training_loop import train_model
from unet import UNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = load_dataset(10)

model = UNet(in_channels=1, out_channels=1)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

train_model(
    '3_testnet',
    model,
    {'train': train_loader, 'val': test_loader},
    criterion,
    optimizer,
    device,
    num_epochs=1000
)
