from interpret_segmentation import hdm
import matplotlib.pyplot as plt
import torch
from testnet.unet import UNet
from testnet.dataset import TestnetDataset
from torchvision import transforms
from pathlib import Path
from skimage.feature import canny


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1)
state_dict = torch.load('testnet/testnet.pth', map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
transform = transforms.Compose([
    transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
])
dataset = TestnetDataset(Path('testnet/dataset/'), transform)


sample = dataset.get_sample('1')
segment = sample['segment']
image = sample['input']

explainer = hdm.HausdorffDistanceMasks(240, 240)
explainer.generate_masks(circle_size=25, offset=5)

result = explainer.explain(model, image, segment, device)

raw = result.circle_map(hdm.RAW, color_map='Blues')
better = result.circle_map(hdm.BETTER_ONLY, color_map='Greens')
worse = result.circle_map(hdm.WORSE_ONLY, color_map='Reds')

edges = canny(image[0].numpy(), sigma=0.01)

plt.imshow(raw)
plt.imshow(edges, alpha=0.5, cmap='gray_r')
plt.suptitle(f'Raw')
plt.savefig('hdm_raw.png')
print('hdm_raw.png generated')

plt.imshow(better)
plt.imshow(edges, alpha=0.5, cmap='gray_r')
plt.suptitle(f'Better')
plt.savefig('hdm_better.png')
print('hdm_better.png generated')

plt.imshow(worse)
plt.imshow(edges, alpha=0.5, cmap='gray_r')
plt.suptitle(f'Worse')
plt.savefig('hdm_worse.png')
print('hdm_worse.png generated')
