import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Uses pyplot to display the passed in image
def plot_single(image):
	# if image.shape[2] == 1:
	#     image = image.squeeze()
	plt.imshow(image, cmap="Greys_r")
	plt.show() 

# Plots a batch of images
def plot_batch(batch, mb_size):
	fig = plt.figure(figsize=(1,mb_size))
	gs = gridspec.GridSpec(1,mb_size)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(batch):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		if sample.shape[2] == 1:
			sample = sample.squeeze()
		plt.imshow(sample, cmap="Greys_r")
	plt.show() 
