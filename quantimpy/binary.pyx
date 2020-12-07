import numpy as np
cimport numpy as np

np.import_array()

################################################################################
# {{{ ErodeDist

cpdef ErodeDist(np.ndarray image, int dist, res=None):

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

        return ErodeDist2D(image, dist, res0, res1)
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

        return ErodeDist3D(image, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "binaryc.h":
    bint cErodeDist2D(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dist, double res0, double res1)

################################################################################

def ErodeDist2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, int dist, double res0, double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] erosion = np.empty_like(image,dtype=np.uint16)
    
    status = cErodeDist2D(
        &image[0,0],
        &erosion[0,0],
        image.shape[0],
        image.shape[1],
        dist,
        res0,
        res1,
    )

    assert status == 0
    return erosion.astype(np.bool) 

################################################################################

cdef extern from "binaryc.h":
    bint cErodeDist3D(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2)

################################################################################

def ErodeDist3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, int dist, double res0, double res1, double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] erosion = np.empty_like(image,dtype=np.uint16)
    
    status = cErodeDist3D(
        &image[0,0,0],
        &erosion[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        dist,
        res0,
        res1,
        res2,
    )

    assert status == 0
    return erosion.astype(np.bool) 

# }}}
################################################################################
# {{{ DilateDist

cpdef DilateDist(np.ndarray image, int dist, res=None):

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

        return DilateDist2D(image, dist, res0, res1)
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

        return DilateDist3D(image, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "binaryc.h":
    bint cDilateDist2D(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dist, double res0, double res1)

################################################################################

def DilateDist2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, int dist, double res0, double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] dilation = np.empty_like(image,dtype=np.uint16)
    
    status = cDilateDist2D(
        &image[0,0],
        &dilation[0,0],
        image.shape[0],
        image.shape[1],
        dist,
        res0,
        res1,
    )

    assert status == 0
    return dilation.astype(np.bool) 

################################################################################

cdef extern from "binaryc.h":
    bint cDilateDist3D(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2)

################################################################################

def DilateDist3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, int dist, double res0, double res1, double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] dilation = np.empty_like(image,dtype=np.uint16)
    
    status = cDilateDist3D(
        &image[0,0,0],
        &dilation[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        dist,
        res0,
        res1,
        res2,
    )

    assert status == 0
    return dilation.astype(np.bool) 

# }}}
################################################################################
# {{{ OpenDist

cpdef OpenDist(np.ndarray erosion, int dist, res=None):

    if (erosion.dtype == 'bool'):
        erosion = erosion.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input images need to be binary (data type bool)')
    
    if (erosion.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return DilateDist2D(erosion, dist, res0, res1)
    elif (erosion.ndim == 3):
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

        return DilateDist3D(erosion, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

# }}}
################################################################################
# {{{ CloseDist

cpdef CloseDist(np.ndarray dilation, int dist, res=None):

    if (dilation.dtype == 'bool'):
        dilation = dilation.astype(np.uint16)*np.iinfo(np.uint16).max
    else:
        raise ValueError('Input images need to be binary (data type bool)')
    
    if (dilation.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return ErodeDist2D(dilation, dist, res0, res1)
    elif (dilation.ndim == 3):
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

        return ErodeDist3D(dilation, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

# }}}
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
# {{{ GetDistMap

cpdef GetDistMap(np.ndarray image, res=None, int gstep=1, int mode=1):

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

        return GetDistMap2D(image, res0, res1, gstep, mode)
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

        return GetDistMap3D(image, res0, res1, res2, gstep, mode)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "binaryc.h":
    bint cGetDistMap2D(unsigned short* image, unsigned short* distance, int dim0, int dim1, double res0, double res1, int mode)

################################################################################

def GetDistMap2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, double res0, double res1, int gstep, int mode):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] distance = np.empty_like(image,dtype=np.uint16)
    
    status = cGetDistMap2D(
        &image[0,0],
        &distance[0,0],
        image.shape[0],
        image.shape[1],
        res0,
        res1,
        mode,
    )

    assert status == 0
    return distance

################################################################################

cdef extern from "binaryc.h":
    bint cGetDistMap3D(unsigned short* image, unsigned short* distance, int dim0, int dim1, int dim2, double res0, double res1, double res2, int mode)

################################################################################

def GetDistMap3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, double res0, double res1, double res2, int gstep, int mode):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] distance = np.empty_like(image,dtype=np.uint16)
    
    status = cGetDistMap3D(
        &image[0,0,0],
        &distance[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        res0,
        res1,
        res2,
        mode,
    )

    assert status == 0
    return distance

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
