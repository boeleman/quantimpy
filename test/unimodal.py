#!/usr/bin/python3

import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
from quantimpy import filters

# Create 8uint image
image = misc.ascent()
image = image.astype("uint8") # Fix data type

# Filter image
result = filters.anisodiff(image)

# Edge detection
laplace = ndimage.laplace(result)

# Compute histpgram
hist, bins = filters.histogram(laplace)

# Compute unimodal threshold
thrshld = filters.unimodal(hist)

# Plot histogram
plt.bar(bins, hist, width=5e-3)
plt.scatter(bins[thrshld], hist[thrshld])
plt.show()
