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
