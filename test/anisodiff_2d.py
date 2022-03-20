#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from quantimpy import segmentation as sg

# Load data
image = np.load("rock_2d.npy")

# Apply anisotropic diffusion filter
diffusion = sg.anisodiff(image, niter=5)

# Show results
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ax1.imshow(image)
ax2.imshow(diffusion)
plt.show()
