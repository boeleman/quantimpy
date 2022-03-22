#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from quantimpy import segmentation as sg

################################################################################

# Load data uint16
image = np.load("rock_3d.npy")

# Compute histpgram
hist, bins = sg.histogram(image, bits=8)
width = bins[1] - bins[0]

print(bins[0],bins[-1])

# Plot histogram
plt.bar(bins,hist,width=width)
plt.show()

################################################################################

# Create 16int image
image = np.load("rock_3d.npy")
image = image-32768
image = image.astype(np.int16) # Fix data type

# Compute histpgram
hist, bins = sg.histogram(image, bits=8)
width = bins[1] - bins[0]

print(bins[0],bins[-1])

# Plot histogram
plt.bar(bins,hist,width=width)
plt.show()

################################################################################

# Create 8uint image
image = np.load("rock_3d.npy")
image = image/65535*255
image = image.astype(np.uint8) # Fix data type

# Compute histpgram
hist, bins = sg.histogram(image, bits=8)
width = bins[1] - bins[0]

print(bins[0],bins[-1])

# Plot histogram
plt.bar(bins,hist,width=width)
plt.show()

################################################################################

# Create 8int image
image = np.load("rock_3d.npy")
image = image/65535*255 - 128
image = image.astype(np.int8) # Fix data type

# Compute histpgram
hist, bins = sg.histogram(image, bits=8)
width = bins[1] - bins[0]

print(bins[0],bins[-1])

# Plot histogram
plt.bar(bins,hist,width=width)
plt.show()

################################################################################

# Create float image 0 to 1
image = np.load("rock_3d.npy")
image = image/65535.
image = image.astype(np.float64) # Fix data type

# Compute histpgram
hist, bins = sg.histogram(image, bits=8)
width = bins[1] - bins[0]

print(bins[0],bins[-1])

# Plot histogram
plt.bar(bins,hist,width=width)
plt.show()

################################################################################

# Create float image -1 to 1
image = np.load("rock_3d.npy")
image = 2.*image/65535. - 1.
image = image.astype(np.float64) # Fix data type

# Compute histpgram
hist, bins = sg.histogram(image, bits=8)
width = bins[1] - bins[0]

print(bins[0],bins[-1])

# Plot histogram
plt.bar(bins,hist,width=width)
plt.show()

################################################################################
