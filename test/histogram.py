#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.util import random_noise
from quantimpy import filters

# Create 8uint image
image = misc.ascent()
image = image.astype(np.uint8) # Fix data type

# Compute histpgram
hist, bins = filters.histogram(image)

print(bins[0],bins[-1])

# Plot histogram
plt.bar(bins,hist)
plt.show()

# Create 16uint image
image = misc.ascent()
image = image*65535/255
image = image.astype(np.uint16) # Fix data type

# Compute histpgram
hist, bins = filters.histogram(image)

print(bins[0],bins[-1])

# Create 8int image
image = misc.ascent()
image = image-128
image = image.astype(np.int8) # Fix data type

dtype_min = np.iinfo(image.dtype).min 
dtype_max = np.iinfo(image.dtype).max 

# Compute histpgram
hist, bins = filters.histogram(image)

print(bins[0],bins[-1])

# Create 16int image
image = misc.ascent()
image = image*65535/255 - 65536    
image = image.astype(np.int16) # Fix data type

# Compute histpgram
hist, bins = filters.histogram(image)

print(bins[0],bins[-1])

# Create float image 0 to 1
image = misc.ascent()
image = image/255.
image = image.astype(np.float64) # Fix data type

# Compute histpgram
hist, bins = filters.histogram(image)

print(bins[0],bins[-1])

# Create float image -1 to 1
image = misc.ascent()
image = 2.*image/255. - 1.
image = image.astype(np.float64) # Fix data type

# Compute histpgram
hist, bins = filters.histogram(image)

print(bins[0],bins[-1])
