#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import cube
from quantimpy import minkowski as mk
from quantimpy import morphology as mp
from math import prod

image0 = np.zeros([64,64,64],dtype=bool)
image0[8:32,8:32,8:32] = cube(24,dtype=bool)
image0[32:56,32:56,32:56] = cube(24,dtype=bool)
res0 = np.array([2.0, 2.0, 2.0])
ext0 = [0, image0.shape[0]*res0[0], 0, image0.shape[1]*res0[1]]

image1 = np.zeros([128,128,128],dtype=bool)
image1[16:64,16:64,16:64] = cube(48,dtype=bool)
image1[64:112,64:112,64:112] = cube(48,dtype=bool)

image2 = np.zeros([256,256,256],dtype=bool)
image2[32:128,32:128,32:128] = cube(96,dtype=bool)
image2[128:224,128:224,128:224] = cube(96,dtype=bool)
res2 = np.array([0.5, 0.5, 0.5])
ext2 = [0, image2.shape[0]*res2[0], 0, image2.shape[1]*res2[1]]

image3 = np.zeros([128,256,256],dtype=bool)
image3[16:64,32:128,32:128] = cube(96,dtype=bool)[0::2,:,:]
image3[64:112,128:224,128:224] = cube(96,dtype=bool)[0::2,:,:]
res3 = np.array([1.0, 0.5, 0.5])
ext3 = [0, image3.shape[0]*res3[0], 0, image3.shape[1]*res3[1]]

#plt.gray()
#plt.imshow(image0[:,:,32],extent=ext0)
#plt.show()
#
#plt.gray()
#plt.imshow(image1[:,:,64])
#plt.show()
#
#plt.gray()
#plt.imshow(image2[:,:,128],extent=ext2)
#plt.show()
#
#plt.gray()
#plt.imshow(image3[:,:,128],extent=ext3)
#plt.show()

dilation0 = mp.dilate(image0,12,res0)
closing0  = mp.close(dilation0,12,res0)

dilation1 = mp.dilate(image1,12)
closing1  = mp.close(dilation1,12)

dilation2 = mp.dilate(image2,12,res2)
closing2  = mp.close(dilation2,12,res2)

dilation3 = mp.dilate(image3,12,res3)
closing3  = mp.close(dilation3,12,res3)

#plt.gray()
#plt.imshow(dilation0[:,:,32],extent=ext0)
#plt.show()
#
#plt.gray()
#plt.imshow(dilation1[:,:,64])
#plt.show()
#
#plt.gray()
#plt.imshow(dilation2[:,:,128],extent=ext2)
#plt.show()
#
#plt.gray()
#plt.imshow(dilation3[:,:,128],extent=ext3)
#plt.show()

# These images should be the same (exept difference caused by resolution)
plt.gray()
plt.imshow(closing0[:,:,32],extent=ext0)
plt.show()

plt.gray()
plt.imshow(closing1[:,:,64])
plt.show()

plt.gray()
plt.imshow(closing2[:,:,128],extent=ext2)
plt.show()

plt.gray()
plt.imshow(closing3[:,:,128],extent=ext3)
plt.show()

closing0 = ~np.logical_and(~image0,closing0)
closing1 = ~np.logical_and(~image1,closing1)
closing2 = ~np.logical_and(~image2,closing2)
closing3 = ~np.logical_and(~image3,closing3)

plt.gray()
plt.imshow(closing0[:,:,32],extent=ext0)
plt.show()

plt.gray()
plt.imshow(closing1[:,:,64])
plt.show()

plt.gray()
plt.imshow(closing2[:,:,128],extent=ext2)
plt.show()

plt.gray()
plt.imshow(closing3[:,:,128],extent=ext3)
plt.show()
