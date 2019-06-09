# based on: https://github.com/eclique/RISE/blob/fac5d54225977091dc18cc71ef8e07f726c3bc20/explanations.py

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm


class SegmentationRISE(nn.Module):
    def __init__(self, model, input_size, device, gpu_batch=100):
        super(SegmentationRISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.device = device

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.to(self.device)
        self.N = N

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().to(self.device)
        self.N = self.masks.shape[0]

    def forward(self, x):
        mask_count = self.N
        _, _, H, W = x.size()

        # generate new images by putting mask on top of original image
        stack = torch.mul(self.masks, x.data)

        output = self.model(x).squeeze()
        output = (output > output.mean())

        pixels = []
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                if output[x][y]:
                    pixels.append((x, y))

        pixels_per_batch = 1000
        saliencies = []
        for i in range(0, len(pixels), pixels_per_batch):
            current_pixels = pixels[i:i+pixels_per_batch]

            # run generated images through the model
            p = []
            for i in range(0, mask_count, self.gpu_batch):
                output_mask = self.model(stack[i:min(i + self.gpu_batch, mask_count)])
                pixel_classes = []
                for x, y in current_pixels:
                    pixel_classes.append(output_mask[0][x][y])
                p.append(torch.tensor([pixel_classes]))
            p = torch.cat(p)
            p = p.to(self.device)

            # Number of classes
            CL = p.size(1)

            sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(mask_count, H * W))

            sal = sal.view((CL, H, W))
            sal /= mask_count
            saliencies.append(sal)
        return saliencies
