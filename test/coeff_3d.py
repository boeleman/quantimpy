#!/usr/bin/python3

import numpy as np
from quantimpy import brisque as bq

# Load data
image = np.load("rock_3d.npy")

# Compute MSCN coefficients
mscn = bq.mscn(image)

print(bq.coeff(mscn))
