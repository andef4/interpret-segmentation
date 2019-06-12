from interpret_segmentation.rise import SegmentationRISE
import matplotlib.pyplot as plt
import torch
from testnet.unet import UNet
from testnet.dataset import TestnetDataset
from torchvision import transforms
from pathlib import Path


batch_size = 1
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
segment = segment.squeeze()
image = sample['input'].unsqueeze(0).to(device)

masks_path = Path('rise_masks.npy')
explainer = SegmentationRISE(model, (240, 240), device, batch_size)
if not masks_path.exists():
    explainer.generate_masks(N=3000, s=8, p1=0.1, savepath=masks_path)
else:
    explainer.load_masks(masks_path)

with torch.set_grad_enabled(False):
    result = explainer(image)

# Saliency map, Saliency map overlayed on binarized network output (max)
rise_max = result.max()
_, plots = plt.subplots(1, 2, figsize=(10, 5))
plots[0].imshow(rise_max, cmap='jet')
plots[1].imshow(sample['input'].squeeze())
plots[1].imshow(rise_max, cmap='jet', alpha=0.5)
plt.savefig('rise_max.png')
print('examples/rise_max.png generated')

# Saliency map, Saliency map overlayed on binarized network output (mean)
rise_mean = result.mean()
_, plots = plt.subplots(1, 2, figsize=(10, 5))
plots[0].imshow(rise_mean, cmap='jet')
plots[1].imshow(sample['input'].squeeze())
plots[1].imshow(rise_mean, cmap='jet', alpha=0.5)
plt.savefig('rise_mean.png')
print('examples/rise_mean.png generated')
