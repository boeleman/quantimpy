#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import square
from quantimpy import minkowski as mk
from quantimpy import morphology as mp
from math import prod

image0 = np.zeros([64,64],dtype=bool)
image0[8:56,8:56] = square(48,dtype=bool)
res0 = np.array([2.0, 2.0])
ext0 = [0, image0.shape[0]*res0[0], 0, image0.shape[1]*res0[1]]

image1 = np.zeros([128,128],dtype=bool)
image1[16:112,16:112] = square(96,dtype=bool)

image2 = np.zeros([256,256],dtype=bool)
image2[32:224,32:224] = square(192,dtype=bool)
res2 = np.array([0.5, 0.5])
ext2 = [0, image2.shape[0]*res2[0], 0, image2.shape[1]*res2[1]]

image3 = np.zeros([128,256],dtype=bool)
image3[16:112,32:224] = square(192,dtype=bool)[0::2,:]
res3 = np.array([1.0, 0.5])
ext3 = [0, image3.shape[0]*res3[0], 0, image3.shape[1]*res3[1]]

#plt.gray()
#plt.imshow(image0[:,:],extent=ext0)
#plt.show()
#
#plt.gray()
#plt.imshow(image1[:,:])
#plt.show()
#
#plt.gray()
#plt.imshow(image2[:,:],extent=ext2)
#plt.show()
#
#plt.gray()
#plt.imshow(image3[:,:],extent=ext3)
#plt.show()

erosion0 = mp.erode(image0,12,res0)
opening0  = mp.open(erosion0,12,res0)

erosion1 = mp.erode(image1,12)
opening1  = mp.open(erosion1,12)

erosion2 = mp.erode(image2,12,res2)
opening2  = mp.open(erosion2,12,res2)

erosion3 = mp.erode(image3,12,res3)
opening3  = mp.open(erosion3,12,res3)

#plt.gray()
#plt.imshow(erosion0[:,:],extent=ext0)
#plt.show()
#
#plt.gray()
#plt.imshow(erosion1[:,:])
#plt.show()
#
#plt.gray()
#plt.imshow(erosion2[:,:],extent=ext2)
#plt.show()
#
#plt.gray()
#plt.imshow(erosion3[:,:],extent=ext3)
#plt.show()

# These images should be the same (exept difference caused by resolution)
plt.gray()
plt.imshow(opening0[:,:],extent=ext0)
plt.show()

plt.gray()
plt.imshow(opening1[:,:])
plt.show()

plt.gray()
plt.imshow(opening2[:,:],extent=ext2)
plt.show()

plt.gray()
plt.imshow(opening3[:,:],extent=ext3)
plt.show()

opening0 = ~np.logical_and(image0,~opening0)
opening1 = ~np.logical_and(image1,~opening1)
opening2 = ~np.logical_and(image2,~opening2)
opening3 = ~np.logical_and(image3,~opening3)

plt.gray()
plt.imshow(opening0[:,:],extent=ext0)
plt.show()

plt.gray()
plt.imshow(opening1[:,:])
plt.show()

plt.gray()
plt.imshow(opening2[:,:],extent=ext2)
plt.show()

plt.gray()
plt.imshow(opening3[:,:],extent=ext3)
plt.show()
