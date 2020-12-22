import numpy as np
cimport numpy as np

np.import_array()

################################################################################
# {{{ Functionals

cpdef Functionals(np.ndarray image, res=None):

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

        return Functionals2D(image, res0, res1)
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

        return Functionals3D(image, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D images')

################################################################################

cdef extern from "minkowskic.h":
    bint cFunctionals2D(unsigned short* image, int dim0, int dim1, double res0, double res1, double* area, double* length, double* euler4, double* euler8)

################################################################################

def Functionals2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] image, double res0, double res1):
    cdef double area 
    cdef double length 
    cdef double euler4 
    cdef double euler8
    
    image = np.ascontiguousarray(image)
    
    status = cFunctionals2D(
        &image[0,0],
        image.shape[0],
        image.shape[1],
        res0,
        res1,
        &area, 
        &length, 
        &euler4, 
        &euler8
    )

    assert status == 0
    return area, length, euler8

################################################################################

cdef extern from "minkowskic.h":
    bint cFunctionals3D(unsigned short* image, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* volume, double* surface, double* curvature, double* euler6, double* euler26)

################################################################################

def Functionals3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] image, double res0, double res1, double res2):
    cdef double volume 
    cdef double surface 
    cdef double curvature 
    cdef double euler6
    cdef double euler26
    
    image = np.ascontiguousarray(image)

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] erosion = np.empty_like(image,dtype=np.uint16)
    
    status = cFunctionals3D(
        &image[0,0,0],
        image.shape[0],
        image.shape[1],
        image.shape[2],
        res0,
        res1,
        res2,
        &volume, 
        &surface, 
        &curvature, 
        &euler6,
        &euler26
    )

    assert status == 0
    return volume, surface, curvature, euler26

# }}}
################################################################################
# {{{ FunctionsOpen

cpdef FunctionsOpen(np.ndarray closing, res=None):

    if not (closing.dtype == 'uint16'):
        raise ValueError('Input image needs to be data type uint16')
    
    if (closing.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return FunctionsOpen2D(closing, res0, res1)
    elif (closing.ndim == 3):
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

        return FunctionsOpen3D(closing, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D closings')

################################################################################

cdef extern from "minkowskic.h":
    bint cFunctionsOpen2D(unsigned short* closing, int dim0, int dim1, double res0, double res1, double* dist, double* area, double* length, double* euler4, double* euler8)

################################################################################

def FunctionsOpen2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] closing, res0, res1):
    cdef int dim3

    closing = np.ascontiguousarray(closing)

    dim3 = np.amax(closing) - np.amin(closing)

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] dist   = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] area   = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] length = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler4 = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler8 = np.empty(dim3,dtype=np.double)

    status = cFunctionsOpen2D(
        &closing[0,0], 
        closing.shape[0],
        closing.shape[1],
        res0,
        res1,
        &dist[0],
        &area[0],
        &length[0],
        &euler4[0],
        &euler8[0]
    )

    assert status == 0
    return dist, area, length, euler8

################################################################################

cdef extern from "minkowskic.h":
    bint cFunctionsOpen3D(unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* dist, double* volume, double* surface, double* curvature, double* euler6, double* euler26)

################################################################################

def FunctionsOpen3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] closing, res0, res1, res2):
    cdef int dim3

    closing = np.ascontiguousarray(closing)

    dim3 = np.amax(closing) - np.amin(closing)

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] dist      = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] volume    = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] surface   = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] curvature = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler6    = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler26   = np.empty(dim3,dtype=np.double)

    status = cFunctionsOpen3D(
        &closing[0,0,0], 
        closing.shape[0],
        closing.shape[1],
        closing.shape[2],
        res0,
        res1,
        res2,
        &dist[0],
        &volume[0],
        &surface[0],
        &curvature[0],
        &euler6[0],
        &euler26[0]
    )

    assert status == 0
    return dist, volume, surface, curvature, euler26

# }}}
################################################################################
# {{{ FunctionsClose

cpdef FunctionsClose(np.ndarray closing, res=None):

    if not (closing.dtype == 'uint16'):
        raise ValueError('Input image needs to be data type uint16')
    
    if (closing.ndim == 2):
# Set default resolution (length/voxel)
        if (res is None):
            res0 = 1.0
            res1 = 1.0
        else:
            res  = res.astype(np.double)
            res0 = res[0]
            res1 = res[1]

        return FunctionsClose2D(closing, res0, res1)
    elif (closing.ndim == 3):
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

        return FunctionsClose3D(closing, res0, res1, res2)
    else:
        raise ValueError('Can only handle 2D or 3D closings')

################################################################################

cdef extern from "minkowskic.h":
    bint cFunctionsClose2D(unsigned short* closing, int dim0, int dim1, double res0, double res1, double* dist, double* area, double* length, double* euler4, double* euler8)

################################################################################

def FunctionsClose2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] closing, res0, res1):
    cdef int dim3

    closing = np.ascontiguousarray(closing)

    dim3 = np.amax(closing) - np.amin(closing)

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] dist   = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] area   = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] length = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler4 = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler8 = np.empty(dim3,dtype=np.double)

    status = cFunctionsClose2D(
        &closing[0,0], 
        closing.shape[0],
        closing.shape[1],
        res0,
        res1,
        &dist[0],
        &area[0],
        &length[0],
        &euler4[0],
        &euler8[0]
    )

    assert status == 0
    return dist, area, length, euler8

################################################################################

cdef extern from "minkowskic.h":
    bint cFunctionsClose3D(unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* dist, double* volume, double* surface, double* curvature, double* euler6, double* euler26)

################################################################################

def FunctionsClose3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] closing, res0, res1, res2):
    cdef int dim3

    closing = np.ascontiguousarray(closing)

    dim3 = np.amax(closing) - np.amin(closing)

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] dist      = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] volume    = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] surface   = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] curvature = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler6    = np.empty(dim3,dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] euler26   = np.empty(dim3,dtype=np.double)

    status = cFunctionsClose3D(
        &closing[0,0,0], 
        closing.shape[0],
        closing.shape[1],
        closing.shape[2],
        res0,
        res1,
        res2,
        &dist[0],
        &volume[0],
        &surface[0],
        &curvature[0],
        &euler6[0],
        &euler26[0]
    )

    assert status == 0
    return dist, volume, surface, curvature, euler26

# }}}
################################################################################
