#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

/******************************************************************************/

unsigned short rPixel2D(int x, int y       , unsigned short* image, int dim1          );
unsigned short rPixel3D(int x, int y, int z, unsigned short* image, int dim1, int dim2);

void wPixel2D(int x, int y       , unsigned short* image, int dim1          , unsigned short value);
void wPixel3D(int x, int y, int z, unsigned short* image, int dim1, int dim2, unsigned short value);

void bin2D(int low, int value1, int value2, unsigned short* image, int dim0, int dim1          );
void bin3D(int low, int value1, int value2, unsigned short* image, int dim0, int dim1, int dim2);

/******************************************************************************/
