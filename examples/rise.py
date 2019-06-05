
masks_path = Path('rise_masks.npy')
explainer = SegmentationRISE(model, (240, 240), batch_size)
if not masks_path.exists():
    explainer.generate_masks(N=3000, s=8, p1=0.1, savepath=masks_path)
else:
    explainer.load_masks(masks_path)

saliencies = None
with torch.set_grad_enabled(False):
    saliencies = explainer(image)


# In[5]:

plot_image_row([segment, output], labels=['Ground truth', 'Binarized network output'])

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
