#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import square
from quantimpy import minkowski as mk
from quantimpy import morphology as mp
from math import prod

image0 = np.zeros([64,64],dtype=bool)
image0[8:32,8:32] = square(24,dtype=bool)
image0[32:56,32:56] = square(24,dtype=bool)
res0 = np.array([2.0, 2.0])
ext0 = [0, image0.shape[0]*res0[0], 0, image0.shape[1]*res0[1]]

image1 = np.zeros([128,128],dtype=bool)
image1[16:64,16:64] = square(48,dtype=bool)
image1[64:112,64:112] = square(48,dtype=bool)

image2 = np.zeros([256,256],dtype=bool)
image2[32:128,32:128] = square(96,dtype=bool)
image2[128:224,128:224] = square(96,dtype=bool)
res2 = np.array([0.5, 0.5])
ext2 = [0, image2.shape[0]*res2[0], 0, image2.shape[1]*res2[1]]

image3 = np.zeros([128,256],dtype=bool)
image3[16:64,32:128] = square(96,dtype=bool)[0::2,:]
image3[64:112,128:224] = square(96,dtype=bool)[0::2,:]
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

dilation_map0 = mp.dilate_map(image0,res0)
closing_map0  = mp.close_map(dilation_map0,res0)

dilation_map1 = mp.dilate_map(image1)
closing_map1  = mp.close_map(dilation_map1)

dilation_map2 = mp.dilate_map(image2,res2)
closing_map2  = mp.close_map(dilation_map2,res2)

dilation_map3 = mp.dilate_map(image3,res3)
closing_map3  = mp.close_map(dilation_map3,res3)

#plt.gray()
#plt.imshow(dilation_map0[:,:],extent=ext0)
#plt.show()
#
#plt.gray()
#plt.imshow(dilation_map1[:,:])
#plt.show()
#
#plt.gray()
#plt.imshow(dilation_map2[:,:],extent=ext2)
#plt.show()
#
#plt.gray()
#plt.imshow(dilation_map3[:,:],extent=ext3)
#plt.show()

# These images should be the same (exept difference caused by resolution)
plt.gray()
plt.imshow(closing_map0[:,:],extent=ext0)
plt.show()

plt.gray()
plt.imshow(closing_map1[:,:])
plt.show()

plt.gray()
plt.imshow(closing_map2[:,:],extent=ext2)
plt.show()

plt.gray()
plt.imshow(closing_map3[:,:],extent=ext3)
plt.show()
