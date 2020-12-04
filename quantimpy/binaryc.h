#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

int cGetDistOpenMap2D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, double res0, double res1, int gval, int gstep);
int cGetDistOpenMap3D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, int dim2, double res0, double res1, double res2, int gval, int gstep);

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

void bin2D(int LOW, int value1, int value2, unsigned short* image, int dim0, int dim1);
void bin3D(int LOW, int value1, int value2, unsigned short* image, int dim0, int dim1, int dim2);
