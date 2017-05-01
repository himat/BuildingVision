
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Uses pyplot to display the passed in image
def plot_save_single(image, save_only=False, dir=None, name=None):
    if image.shape[2] == 1:
        image = image.squeeze()
    plt.imshow(image, cmap="Greys_r")
    plt.axis('off')

    if save_only:
        if dir == None or name == None:
            raise ValueError("Need a dir/name for saving image to disk")

        save_name = os.path.join(dir, name)
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Plots a batch of images and optionally saves to disk
def plot_save_batch(batch, iter_num, save_only=False, prefix=""):
    output_dir = "out"

    (mb_size, width, height, channels) = batch.shape

    sqrt = math.sqrt(mb_size)
    if sqrt.is_integer():
        fig_shape = (int(sqrt), int(sqrt))
    else:
        fig_shape = (1, mb_size)

    fig = plt.figure(figsize=fig_shape)
    gs = gridspec.GridSpec(*fig_shape)
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

    if save_only:
        plt.savefig(os.path.join(output_dir, prefix+str(iter_num).zfill(3)+".png"), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

