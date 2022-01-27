#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import ball
from quantimpy import minkowski as mk
from quantimpy import morphology as mp
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

erosion_map0 = mp.erode_map(image0,res0)
erosion_map1 = mp.erode_map(image1)
erosion_map2 = mp.erode_map(image2,res2)
erosion_map3 = mp.erode_map(image3,res3)

#plt.gray()
#plt.imshow(erosion_map0[:,:,32],extent=ext0)
#plt.show()
#
#plt.gray()
#plt.imshow(erosion_map1[:,:,64])
#plt.show()
#
#plt.gray()
#plt.imshow(erosion_map2[:,:,128],extent=ext2)
#plt.show()
#
#plt.gray()
#plt.imshow(erosion_map3[:,:,128],extent=ext3)
#plt.show()

dist0, volume0, surface0, curvature0, euler0 = mk.functions_close(erosion_map0,res0)
dist1, volume1, surface1, curvature1, euler1 = mk.functions_close(erosion_map1)
dist2, volume2, surface2, curvature2, euler2 = mk.functions_close(erosion_map2,res2)
dist3, volume3, surface3, curvature3, euler3 = mk.functions_close(erosion_map3,res3)

volume, surface, curvature, euler = mk.functionals(image1)

plt.plot(48,volume,marker="o")
plt.plot(dist0,volume0)
plt.plot(dist1,volume1)
plt.plot(dist2,volume2)
plt.plot(dist3,volume3)
plt.show()

plt.plot(48,surface,marker="o")
plt.plot(dist0,surface0)
plt.plot(dist1,surface1)
plt.plot(dist2,surface2)
plt.plot(dist3,surface3)
plt.show()

plt.plot(48,curvature,marker="o")
plt.plot(dist0,curvature0)
plt.plot(dist1,curvature1)
plt.plot(dist2,curvature2)
plt.plot(dist3,curvature3)
plt.show()

plt.plot(48,euler,marker="o")
plt.plot(dist0,euler0)
plt.plot(dist1,euler1)
plt.plot(dist2,euler2)
plt.plot(dist3,euler3)
plt.show()

dist0, volume0, surface0, curvature0, euler0 = mk.functions_close(erosion_map0,res0,norm=True)
dist1, volume1, surface1, curvature1, euler1 = mk.functions_close(erosion_map1,norm=True)
dist2, volume2, surface2, curvature2, euler2 = mk.functions_close(erosion_map2,res2,norm=True)
dist3, volume3, surface3, curvature3, euler3 = mk.functions_close(erosion_map3,res3,norm=True)

volume, surface, curvature, euler = mk.functionals(image1,norm=True)

plt.plot(48,volume,marker="o")
plt.plot(dist0,volume0)
plt.plot(dist1,volume1)
plt.plot(dist2,volume2)
plt.plot(dist3,volume3)
plt.show()

plt.plot(48,surface,marker="o")
plt.plot(dist0,surface0)
plt.plot(dist1,surface1)
plt.plot(dist2,surface2)
plt.plot(dist3,surface3)
plt.show()

plt.plot(48,curvature,marker="o")
plt.plot(dist0,curvature0)
plt.plot(dist1,curvature1)
plt.plot(dist2,curvature2)
plt.plot(dist3,curvature3)
plt.show()

plt.plot(48,euler,marker="o")
plt.plot(dist0,euler0)
plt.plot(dist1,euler1)
plt.plot(dist2,euler2)
plt.plot(dist3,euler3)
plt.show()
