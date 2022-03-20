#!/usr/bin/python3

import numpy as np
from quantimpy import segmentation as sg

# Load data
image = np.load("rock_2d.npy")

# Apply anisotropic diffusion filter
diffusion = sg.anisodiff(image, niter=5)

# Compute minimum and maximum thresholds
thrshld_min, thrshld_max = sg.gradient(diffusion)

# Print results 
print(thrshld_min, thrshld_max) 
