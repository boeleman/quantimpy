import numpy as np
cimport numpy as np

np.import_array()

################################################################################
# {{{ Erode

cpdef Erode(np.ndarray image, int dist, res=None):

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

        return Erode2D(image, dist, res0, res1)
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

        return Erode3D(image, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "morphologyc.h":
    bint cErode2D(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dist, double res0, double res1)

################################################################################

def Erode2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, int dist, double res0, double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] erosion = np.empty_like(image,dtype=np.uint16)
    
    status = cErode2D(
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

cdef extern from "morphologyc.h":
    bint cErode3D(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2)

################################################################################

def Erode3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, int dist, double res0, double res1, double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] erosion = np.empty_like(image,dtype=np.uint16)
    
    status = cErode3D(
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
# {{{ Dilate

cpdef Dilate(np.ndarray image, int dist, res=None):

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

        return Dilate2D(image, dist, res0, res1)
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

        return Dilate3D(image, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "morphologyc.h":
    bint cDilate2D(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dist, double res0, double res1)

################################################################################

def Dilate2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, int dist, double res0, double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] dilation = np.empty_like(image,dtype=np.uint16)
    
    status = cDilate2D(
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

cdef extern from "morphologyc.h":
    bint cDilate3D(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2)

################################################################################

def Dilate3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, int dist, double res0, double res1, double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] dilation = np.empty_like(image,dtype=np.uint16)
    
    status = cDilate3D(
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
# {{{ Open

cpdef Open(np.ndarray erosion, int dist, res=None):

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

        return Dilate2D(erosion, dist, res0, res1)
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

        return Dilate3D(erosion, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

# }}}
################################################################################
# {{{ Close

cpdef Close(np.ndarray dilation, int dist, res=None):

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

        return Erode2D(dilation, dist, res0, res1)
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

        return Erode3D(dilation, dist, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

# }}}
################################################################################
# {{{ ErodeMap

cpdef ErodeMap(np.ndarray image, res=None):

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

        return ErodeMap2D(image, res0, res1)
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

        return ErodeMap3D(image, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "morphologyc.h":
    bint cGetMap2D(unsigned short* image, unsigned short* distance, int dim0, int dim1, double res0, double res1, int mode)

################################################################################

def ErodeMap2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, double res0, double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] distance = np.empty_like(image,dtype=np.uint16)
    
    status = cGetMap2D(
        &image[0,0],
        &distance[0,0],
        image.shape[0],
        image.shape[1],
        res0,
        res1,
        1,
    )

    assert status == 0
    return distance

################################################################################

cdef extern from "morphologyc.h":
    bint cGetMap3D(unsigned short* image, unsigned short* distance, int dim0, int dim1, int dim2, double res0, double res1, double res2, int mode)

################################################################################

def ErodeMap3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, double res0, double res1, double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] distance = np.empty_like(image,dtype=np.uint16)
    
    status = cGetMap3D(
        &image[0,0,0],
        &distance[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        res0,
        res1,
        res2,
        1,
    )

    assert status == 0
    return distance

# }}}
################################################################################
# {{{ DilateMap

cpdef DilateMap(np.ndarray image, res=None):

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

        return DilateMap2D(image, res0, res1)
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

        return DilateMap3D(image, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

def DilateMap2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, double res0, double res1):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] distance = np.empty_like(image,dtype=np.uint16)
    
    status = cGetMap2D(
        &image[0,0],
        &distance[0,0],
        image.shape[0],
        image.shape[1],
        res0,
        res1,
        0,
    )

    assert status == 0
    return distance

################################################################################

def DilateMap3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, double res0, double res1, double res2):
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] distance = np.empty_like(image,dtype=np.uint16)
    
    status = cGetMap3D(
        &image[0,0,0],
        &distance[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        res0,
        res1,
        res2,
        0,
    )

    assert status == 0
    return distance

# }}}
################################################################################
# {{{ OpenMap

cpdef OpenMap(np.ndarray erosion, res=None):

    if not (erosion.dtype == 'uint16'):
        raise ValueError('Input image needs to be data type uint16')
    
    if (erosion.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return OpenMap2D(erosion, res0, res1)
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

        return OpenMap3D(erosion, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "morphologyc.h":
    bint cOpenMap2D(unsigned short* erosion, unsigned short* opening, int dim0, int dim1, double res0, double res1)

################################################################################

def OpenMap2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] erosion, double res0, double res1):
    
    erosion = np.ascontiguousarray(erosion)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] opening = np.empty_like(erosion,dtype=np.uint16)
    
    status = cOpenMap2D(
        &erosion[0,0],
        &opening[0,0],
        erosion.shape[0],
        erosion.shape[1],
        res0,
        res1,
    )

    assert status == 0
    return opening

################################################################################

cdef extern from "morphologyc.h":
    bint cOpenMap3D(unsigned short* erosion, unsigned short* opening, int dim0, int dim1, int dim2, double res0, double res1, double res2)

################################################################################

def OpenMap3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] erosion, double res0, double res1, double res2):
    
    erosion = np.ascontiguousarray(erosion)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] opening = np.empty_like(erosion,dtype=np.uint16)
    
    status = cOpenMap3D(
        &erosion[0,0,0],
        &opening[0,0,0],
        erosion.shape[0],
        erosion.shape[1],
        erosion.shape[2],
        res0,
        res1,
        res2,
    )

    assert status == 0
    return opening

# }}}
################################################################################
# {{{ CloseMap

cpdef CloseMap(np.ndarray dilation, res=None):

    if not (dilation.dtype == 'uint16'):
        raise ValueError('Input image needs to be data type uint16')
    
    if (dilation.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return CloseMap2D(dilation, res0, res1)
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

        return CloseMap3D(dilation, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "morphologyc.h":
    bint cCloseMap2D(unsigned short* dilation, unsigned short* closing, int dim0, int dim1, double res0, double res1)

################################################################################

def CloseMap2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] dilation, double res0, double res1):
    
    dilation = np.ascontiguousarray(dilation)

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] closing = np.empty_like(dilation,dtype=np.uint16)
    
    status = cCloseMap2D(
        &dilation[0,0],
        &closing[0,0],
        dilation.shape[0],
        dilation.shape[1],
        res0,
        res1,
    )

    assert status == 0
    return closing

################################################################################

cdef extern from "morphologyc.h":
    bint cCloseMap3D(unsigned short* dilation, unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2)

################################################################################

def CloseMap3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] dilation, double res0, double res1, double res2):
    
    dilation = np.ascontiguousarray(dilation)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] closing = np.empty_like(dilation,dtype=np.uint16)
    
    status = cCloseMap3D(
        &dilation[0,0,0],
        &closing[0,0,0],
        dilation.shape[0],
        dilation.shape[1],
        dilation.shape[2],
        res0,
        res1,
        res2,
    )

    assert status == 0
    return closing

# }}}
################################################################################
