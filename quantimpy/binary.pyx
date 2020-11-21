import numpy as np
cimport numpy as np

np.import_array()

cpdef GetDistOpenMap(np.ndarray image, res=None, int gval=1, int gstep=1):

    if (image.dtype == 'bool'):
        image = image.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input image needs to be binary of data type bool')

    res = res.astype(np.double)

    if (image.ndim == 2):

# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1
            res1 = 1
        else:
            res0 = res[0]
            res1 = res[1]

        return GetDistOpenMap2D(image, res0, res1, gval, gstep)
    elif (image.ndim == 3):
        return image, image 
#        return GetDistOpenMap3D(image, gval, gstep)
    else:
        raise ValueError('Can only handle 2D or 3D images')

cdef extern from "binaryc.h":
    bint cGetDistOpenMap2D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, double res0, double res1, int gval, int gstep)

def GetDistOpenMap2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, double res0, double res1, int gval, int gstep):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] distance = np.empty_like(image,dtype=np.uint16)
    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] opened   = np.empty_like(image,dtype=np.uint16)
    
    status = cGetDistOpenMap2D(
        &image[0,0],
        &distance[0,0],
        &opened[0,0],
        image.shape[1],
        image.shape[0],
        res0,
        res1,
        gval,
        gstep,
    )

    assert status == 0
    return distance, opened 

#cpdef GetDistOpenMap3D(np.ndarray[np.uint8_t,cast=True,ndim=3] image, int gval, int gstep):
#    return image
