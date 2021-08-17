#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

/******************************************************************************/

// Define M_PI for Compilation on Windows
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

/******************************************************************************/

unsigned short r_pixel_2d(int x, int y       , unsigned short* image, int dim1          );
unsigned short r_pixel_3d(int x, int y, int z, unsigned short* image, int dim1, int dim2);

void w_pixel_2d(int x, int y       , unsigned short* image, int dim1          , unsigned short value);
void w_pixel_3d(int x, int y, int z, unsigned short* image, int dim1, int dim2, unsigned short value);

void bin_2d(int low, int value1, int value2, unsigned short* image, int dim0, int dim1          );
void bin_3d(int low, int value1, int value2, unsigned short* image, int dim0, int dim1, int dim2);

/******************************************************************************/
