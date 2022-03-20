#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from quantimpy import segmentation as sg

# Load data
image = np.load("rock_2d.npy")

# Apply anisotropic diffusion filter
diffusion = sg.anisodiff(image, niter=5)

# Compute minimum and maximum thresholds
thrshld_min, thrshld_max = sg.gradient(diffusion, alpha=1.2)

# Apply bi-level segmentation
binary = sg.bilevel(diffusion, thrshld_min, thrshld_max)

# Show results
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(131)  # left side
ax2 = fig.add_subplot(132)  # right side
ax3 = fig.add_subplot(133)  # right side
ax1.imshow(image)
ax2.imshow(diffusion)
ax3.imshow(binary)
plt.show()
