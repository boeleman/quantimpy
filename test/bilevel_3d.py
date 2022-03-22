#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from quantimpy import segmentation as sg

# Load data
image = np.load("rock_3d.npy")

# Apply anisotropic diffusion filter
diffusion = sg.anisodiff(image, niter=3)

# Compute minimum and maximum thresholds
thrshld_min, thrshld_max = sg.gradient(diffusion, alpha=1.5)

# Apply bi-level segmentation
binary = sg.bilevel(diffusion, thrshld_min, thrshld_max)

# Show results
fig = plt.figure()
plt.gray()  # show the result in grayscale
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(image[50,:,:])
ax2.imshow(diffusion[50,:,:])
ax3.imshow(binary[50,:,:])
plt.show()
