#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from quantimpy import brisque as bq

# Load data
image = np.load("rock_3d.npy")

# Compute MSCN coefficients
mscn = bq.mscn(image)

# Show coefficients
plt.gray()
plt.imshow(mscn[50,:,:])
plt.show()
