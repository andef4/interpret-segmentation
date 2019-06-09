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
image = sample['input'].to(device)
output = model(image)
output = output.detach().cpu().squeeze().numpy()
output = (output > output.mean())

masks_path = Path('rise_masks.npy')
explainer = SegmentationRISE(model, (240, 240), batch_size)
if not masks_path.exists():
    explainer.generate_masks(N=3000, s=8, p1=0.1, savepath=masks_path)
else:
    explainer.load_masks(masks_path)

saliencies = None
with torch.set_grad_enabled(False):
    saliencies = explainer(image)

print('Ground truth')
plt.imshow(segment)
plt.show()

print('Binarized network output')
plt.imshow(output)
plt.show()


print('Saliency map, Saliency map overlayed on binarized network output (max)')

merged = torch.cat(saliencies)
maxed = torch.max(merged, dim=0)[0]
_, plots = plt.subplots(1, 2, figsize=(10, 5))
plots[0].imshow(maxed.cpu(), cmap='jet')
plots[1].imshow(output)
plots[1].imshow(maxed.cpu(), cmap='jet', alpha=0.5)
plt.show()

print('Saliency map, Saliency map overlayed on binarized network output (mean)')
mean = torch.mean(merged, dim=0)
_, plots = plt.subplots(1, 2, figsize=(10, 5))
plots[0].imshow(mean.cpu(), cmap='jet')
plots[1].imshow(output)
plots[1].imshow(mean.cpu(), cmap='jet', alpha=0.5)
plt.show()
