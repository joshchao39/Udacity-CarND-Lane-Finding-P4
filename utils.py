import matplotlib.pyplot as plt
import numpy as np


def save_2_images(img1, img2, title1, title2, filepath, cmap1=None, cmap2=None):
    plt.axis('off')
    fig = plt.figure(figsize=(24.0, 15.0))
    font_size = 30
    fig.add_subplot(1, 2, 1)
    plt.title(title1, fontsize=font_size)
    plt.imshow(img1, cmap1)

    fig.add_subplot(1, 2, 2)
    plt.title(title2, fontsize=font_size)
    plt.imshow(img2, cmap2)
    axes = fig.get_axes()
    for axis in axes:
        axis.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0, wspace=0.05)
    plt.savefig(filepath, bbox_inches='tight')
    plt.figure()


def plot_to_ndarray(fig):
    plt.axis('off')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
