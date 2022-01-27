#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import ball
from quantimpy import minkowski as mk
from math import prod

image0 = np.zeros([64,64,64],dtype=bool)
image0[8:57,8:57,8:57] = ball(24,dtype=bool)
res0 = np.array([2.0, 2.0, 2.0])
ext0 = [0, image0.shape[0]*res0[0], 0, image0.shape[1]*res0[1]]

image1 = np.zeros([128,128,128],dtype=bool)
image1[16:113,16:113,16:113] = ball(48,dtype=bool)

image2 = np.zeros([256,256,256],dtype=bool)
image2[32:225,32:225,32:225] = ball(96,dtype=bool)
res2 = np.array([0.5, 0.5, 0.5])
ext2 = [0, image2.shape[0]*res2[0], 0, image2.shape[1]*res2[1]]

image3 = np.zeros([128,256,256],dtype=bool)
image3[16:113,32:225,32:225] = ball(96,dtype=bool)[0::2,:]
res3 = np.array([1.0, 0.5, 0.5])
ext3 = [0, image3.shape[0]*res3[0], 0, image3.shape[1]*res3[1]]

plt.gray()
plt.imshow(image0[:,:,32],extent=ext0)
plt.show()

plt.gray()
plt.imshow(image1[:,:,64])
plt.show()

plt.gray()
plt.imshow(image2[:,:,128],extent=ext2)
plt.show()

plt.gray()
plt.imshow(image3[:,:,128],extent=ext3)
plt.show()

minkowski0 = mk.functionals(image0,res0)
minkowski1 = mk.functionals(image1)
minkowski2 = mk.functionals(image2,res2)
minkowski3 = mk.functionals(image3,res3)

print(minkowski0)
print(minkowski1)
print(minkowski2)
print(minkowski3)
print([4/3*np.pi*48**3, 1/2*np.pi*48**2, 4*np.pi*48/(2*np.pi**2), 3/(4*np.pi)])
print()

minkowski0 = mk.functionals(image0,res0, norm=True)
minkowski1 = mk.functionals(image1, norm=True)
minkowski2 = mk.functionals(image2,res2, norm=True)
minkowski3 = mk.functionals(image3,res3, norm=True)

print(minkowski0)
print(minkowski1)
print(minkowski2)
print(minkowski3)
print(np.array([4/3*np.pi*48**3, 1/2*np.pi*48**2, 4*np.pi*48/(2*np.pi**2), 3/(4*np.pi)])/prod(image1.shape))
print()
