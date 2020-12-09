#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

/******************************************************************************/

int cGetDistMap2D(unsigned short* image, unsigned short* outImage, int dim0, int dim1          , double res0, double res1             , int mode);
int cGetDistMap3D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, int dim2, double res0, double res1, double res2, int mode);


int cErodeDist2D(unsigned short* image, unsigned short* erosion, int dim0, int dim1          , int dist, double res0, double res1             );
int cErodeDist3D(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2);

int cDilateDist2D(unsigned short* image, unsigned short* dilation, int dim0, int dim1          , int dist, double res0, double res1             );
int cDilateDist3D(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2);

int cOpenMapDist2D(unsigned short* erosion, unsigned short* opening, int dim0, int dim1          , double res0, double res1             );
int cOpenMapDist3D(unsigned short* erosion, unsigned short* opening, int dim0, int dim1, int dim2, double res0, double res1, double res2);


int cGetDistOpenMap2D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, double res0, double res1, int gval, int gstep);
int cGetDistOpenMap3D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, int dim2, double res0, double res1, double res2, int gval, int gstep);

int cErodeCirc2D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, double res0, double res1, int rad, int mode);
int cErodeCirc3D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, int dim2, double res0, double res1, double res2, int rad, int mode);

/******************************************************************************/

unsigned short* erodeMirCond2D(unsigned short* image, unsigned short* cond, int dim0, int dim1, double d0, double d1, int step, int mode, int fast);
unsigned short* erodeMirCond3D(unsigned short* image, unsigned short* cond, int dim0, int dim1, int dim2, double res0, double res1, double res2, int step, int mode, int fast);

char* getCircElement(int rad, double rx, double ry);
char* getSphereElement(int rad, double rx, double ry, double rz);

int isNeigh2D(int x, int y, unsigned short* image, int dim0, int dim1);
int isNeigh3D(int x, int y, int z, unsigned short* image, int dim0, int dim1, int dim2);

int isBorder2D(int x, int y, int dim0, int dim1);
int isBorder3D(int x, int y, int z, int dim0, int dim1, int dim2);

unsigned short rPixel2D(int x, int y, unsigned short* image, int dim0);
unsigned short rPixel3D(int x, int y, int z, unsigned short* image, int dim1, int dim2);

void wPixel2D(int x, int y, unsigned short* image, int dim0, unsigned short value);
void wPixel3D(int x, int y, int z, unsigned short* image, int dim1, int dim2, unsigned short value);

void bin2D(int low, int value1, int value2, unsigned short* image, int dim0, int dim1);
void bin3D(int low, int value1, int value2, unsigned short* image, int dim0, int dim1, int dim2);

/******************************************************************************/
