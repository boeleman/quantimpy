#include <quantimpyc.h>

/******************************************************************************/
// {{{ rPixel

unsigned short rPixel2D(int x, int y, unsigned short* image, int dim1) {
    int i;

    i = x*dim1 + y;

    return image[i];
}

/******************************************************************************/

unsigned short rPixel3D(int x, int y, int z, unsigned short* image, int dim1, int dim2) {
    int i;

    i = (x*dim1 + y)*dim2 + z;

    return image[i];
}

// }}}
/******************************************************************************/
// {{{ wPixel

void wPixel2D(int x, int y, unsigned short* image, int dim1, unsigned short value) {
    int i;

    i = x*dim1 + y;

    image[i] = value;
}

/******************************************************************************/

void wPixel3D(int x, int y, int z, unsigned short* image, int dim1, int dim2, unsigned short value) {
    int i;

    i = (x*dim1 + y)*dim2 + z;

    image[i] = value;
}

// }}}
/******************************************************************************/
// {{{ bin

void bin2D(int low, int value1, int value2, unsigned short* image, int dim0, int dim1) {
    int x, y, val;

    val = value2>USHRT_MAX? -1 : value2;

    for (y = 0; y < dim1; y++)
        for (x = 0; x < dim0; x++) {
            if (rPixel2D(x, y, image, dim1) <= low)
                wPixel2D(x, y, image, dim1, value1);
            else if (val != -1)
                wPixel2D(x, y, image, dim1, val);
    }
}

/******************************************************************************/

void bin3D(int low, int value1, int value2, unsigned short* image, int dim0, int dim1, int dim2) {
    int x, y, z, val;

    val = value2>USHRT_MAX? -1 : value2;

    for (z = 0; z < dim2; z++)
        for (y = 0; y < dim1; y++)
            for (x = 0; x < dim0; x++) {
                if (rPixel3D(x, y, z, image, dim1, dim2) <= low)
                    wPixel3D(x, y, z, image, dim1, dim2, value1);
                else if (val != -1)
                    wPixel3D(x, y, z, image, dim1, dim2, val);
    }
}

// }}}
/******************************************************************************/

