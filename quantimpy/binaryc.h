#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>


int cGetDistOpenMap2D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, double res0, double res1, int gval, int gstep);

unsigned short* erodeMirCond(unsigned short* image, unsigned short* cond, int dim0, int dim1, double d0, double d1, int step, int mode, int fast);

//unsigned char* getCircElement(int rad, double rx, double ry);
char* getCircElement(int rad, double rx, double ry);

int isNeigh(int x, int y, unsigned short* image, int dim0, int dim1);

int isBorder(int x, int y, int dim0, int dim1);

unsigned short rPixel(int x, int y, unsigned short* image, int dim0);

void wPixel(int x, int y, unsigned short* image, int dim0, unsigned short value);

void bin(int LOW, int value1, int value2, unsigned short* image, int dim0, int dim1);

