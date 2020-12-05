import numpy as np
cimport numpy as np

np.import_array()

################################################################################
# {{{ ErodeCirc

cpdef ErodeCirc(np.ndarray image, res=None, int rad=10, int mode=1):

    if (image.dtype == 'bool'):
        image = image.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input image needs to be binary (data type bool)')
    
    if (image.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return ErodeCirc2D(image, res0, res1, rad, mode)
    elif (image.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return ErodeCirc3D(image, res0, res1, res2, rad, mode)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "binaryc.h":
    bint cErodeCirc2D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, double res0, double res1, int rad, int mode)

################################################################################

def ErodeCirc2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, double res0, double res1, int rad, int mode):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] outImage = np.empty_like(image,dtype=np.uint16)
    
    status = cErodeCirc2D(
        &image[0,0],
        &outImage[0,0],
        image.shape[0],
        image.shape[1],
        res0,
        res1,
        rad,
        mode,
    )

    assert status == 0
    return outImage.astype(np.bool) 

################################################################################

cdef extern from "binaryc.h":
    bint cErodeCirc3D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, int dim2, double res0, double res1, double res2, int rad, int mode)

################################################################################

def ErodeCirc3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, double res0, double res1, double res2, int rad, int mode):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] outImage = np.empty_like(image,dtype=np.uint16)
    
    status = cErodeCirc3D(
        &image[0,0,0],
        &outImage[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        res0,
        res1,
        res2,
        rad,
        mode,
    )

    assert status == 0
    return outImage.astype(np.bool) 

# }}}
################################################################################
# {{{ GetDistOpenMap

cpdef GetDistOpenMap(np.ndarray image, res=None, int gval=1, int gstep=1):

    if (image.dtype == 'bool'):
        image = image.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input image needs to be binary (data type bool)')

    if (image.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return GetDistOpenMap2D(image, res0, res1, gval, gstep)
    elif (image.ndim == 3):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
            res2 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]
            res2 = res[2]

        return GetDistOpenMap3D(image, res0, res1, res2, gval, gstep)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "binaryc.h":
    bint cGetDistOpenMap2D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, double res0, double res1, int gval, int gstep)

################################################################################

def GetDistOpenMap2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, double res0, double res1, int gval, int gstep):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] distance = np.empty_like(image,dtype=np.uint16)
    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] opened   = np.empty_like(image,dtype=np.uint16)
    
    status = cGetDistOpenMap2D(
        &image[0,0],
        &distance[0,0],
        &opened[0,0],
        image.shape[0],
        image.shape[1],
        res0,
        res1,
        gval,
        gstep,
    )

    assert status == 0
    return distance, opened 

################################################################################

cdef extern from "binaryc.h":
    bint cGetDistOpenMap3D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, int dim2, double res0, double res1, double res2, int gval, int gstep)

################################################################################

def GetDistOpenMap3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, double res0, double res1, double res2, int gval, int gstep):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] distance = np.empty_like(image,dtype=np.uint16)
    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] opened   = np.empty_like(image,dtype=np.uint16)
    
    status = cGetDistOpenMap3D(
        &image[0,0,0],
        &distance[0,0,0],
        &opened[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        res0,
        res1,
        res2,
        gval,
        gstep,
    )

    assert status == 0
    return distance, opened 

# }}}
################################################################################
