import matplotlib.pyplot as plt
import torch
from basic_unet import UNet
from dataset import BratsDataset
from torchvision import transforms
from pathlib import Path
import hdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=4, out_channels=1)
state_dict = torch.load('models/3_basic_unet_flat_criterion_279_0.00000.pth')
model.load_state_dict(state_dict)
model = model.to(device)
transform = transforms.Compose([
    transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
])
dataset = BratsDataset(Path('data/processed/'), transform)


sample = dataset.get_sample('Brats18_2013_17_1', 'L1')
segment = sample['segment']
image = sample['input']

explainer = hdm.HausdorffDistanceMasks(240, 240)
explainer.generate_masks(circle_size=15, offset=5, normalize=True)

result = explainer.explain(model, image, segment, device)

raw = result.circle_map(hdm.RAW)
better = result.circle_map(hdm.BETTER_ONLY)
worse = result.circle_map(hdm.WORSE_ONLY)

plt.imshow(image, cmap='gray')
plt.imshow(raw, alpha=0.8, cmap='Blues')
plt.suptitle(f'Raw')
plt.show()

plt.imshow(image, cmap='gray')
plt.imshow(better, alpha=0.8, cmap='Greens')
plt.suptitle(f'Better')
plt.show()

plt.imshow(image, cmap='gray')
plt.imshow(worse, alpha=0.8, cmap='Reds')
plt.suptitle(f'Worse')
plt.show()
