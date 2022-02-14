#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.util import random_noise
from quantimpy import filters

# Create image with noise
image = misc.ascent()
image = random_noise(image, mode='speckle', mean=0.1)

# Filter image
result = filters.anisodiff(image, niter=5)

# Show results
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ax1.imshow(image)
ax2.imshow(result)
plt.show()
