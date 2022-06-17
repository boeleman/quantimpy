#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from quantimpy import brisque as bq

# Load data
image = np.load("rock_2d.npy")

# Compute MSCN coefficients
mscn = bq.mscn(image)

# Show coefficients
plt.gray()
plt.imshow(mscn)
plt.show()
