#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from quantimpy import filters

# Load data
image = np.load("rock_2d.npy")

# Filter image
result = filters.anisodiff(image, niter=3)

# Edge detection
laplace = ndimage.laplace(result)

# Compute histpgram
hist, bins = filters.histogram(laplace)
width = bins[1] - bins[0]

# Compute unimodal threshold
thrshld = filters.unimodal(hist)

# Plot histogram
plt.bar(bins, hist,width=width)
plt.scatter(bins[thrshld], hist[thrshld])
plt.show()

# Compute unimodal threshold
thrshld = filters.unimodal(hist, side="left")

# Plot histogram
plt.bar(bins, hist, width=width)
plt.scatter(bins[thrshld], hist[thrshld])
plt.show()
