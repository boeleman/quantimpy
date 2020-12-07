#include <binaryc.h>

/* 8 Neighbours starting at left the central pixel */
static int neigh8x[8] = {-1,-1, 0, 1,1,1,0,-1};
static int neigh8y[8] = { 0,-1,-1,-1,0,1,1, 1};

/* 26 Neighbours starting at left the central pixel */
static int neigh26x[26] = {-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1};
static int neigh26y[26] = {-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1};
static int neigh26z[26] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1};

/******************************************************************************/
// {{{ cGetDistOpenMap

int cGetDistOpenMap2D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, double res0, double res1, int gval, int gstep) {

    int i, x, y, count, step;
    unsigned short* dilat;
    unsigned short* op;

    step  = 1;
    count = 1;
        
    for (i=0; i < dim0*dim1; i++) {
        distance[i] = image[i];
        opened[i]   = image[i];
    }

    while (count)
    {
        count = 0;
        printf("\r erosion step : %d \n",step);
        dilat = erodeMirCond2D(image, distance, dim0, dim1, res0, res1, step, 1, 1);
        for (x = 0; x < dim0; x++)
            for (y = 0; y < dim1; y++)
                if (!rPixel2D(x,y,dilat,dim1) && rPixel2D(x,y,distance,dim1) == USHRT_MAX) {
                    wPixel2D(x,y,distance,dim1,gval+step*gstep);
                    count++;
        }
    
        for (i=0; i < dim0*dim1; i++)
            dilat[i] = distance[i];

        bin2D(USHRT_MAX-1, 0, USHRT_MAX, dilat, dim0, dim1);
        op = erodeMirCond2D(dilat, opened, dim0, dim1, res0, res1, step, 0, 0);
        for (x = 0; x < dim0; x++)
            for (y = 0; y < dim1; y++)
                if (!rPixel2D(x,y,op,dim1) && rPixel2D(x,y,opened,dim1) == USHRT_MAX) {
                    wPixel2D(x,y,opened,dim1,gval+step*gstep);
                    count++;
        }
        step++;
    }

    free(dilat);
    free(op);

    return 0;
}

/******************************************************************************/

int cGetDistOpenMap3D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, int dim2, double res0, double res1, double res2, int gval, int gstep) {

    int i, x, y, z, count, step;
    unsigned short* dilat;
    unsigned short* op;

    step  = 1;
    count = 1;
        
    for (i=0; i < dim0*dim1*dim2; i++) {
        distance[i] = image[i];
        opened[i]   = image[i];
    }

    while (count) {
        count = 0;
        printf("\r erosion step : %d \n", step);
        dilat = erodeMirCond3D(image, distance, dim0, dim1, dim2, res0, res1, res2, step, 1, 1);
        for (x = 0; x < dim0; x++)
            for (y = 0; y < dim1; y++)
                for (z = 0; z < dim2; z++)
                    if (!rPixel3D(x,y,z,dilat,dim1,dim2) && rPixel3D(x,y,z,distance,dim1,dim2) == USHRT_MAX) {
                        wPixel3D(x,y,z,distance,dim1,dim2,gval+step*gstep);
                        count++;
        }
    
        for (i=0; i < dim0*dim1*dim2; i++)
            dilat[i] = distance[i];

        bin3D(USHRT_MAX-1, 0, USHRT_MAX, dilat, dim0, dim1, dim2);
        op = erodeMirCond3D(dilat, opened, dim0, dim1, dim2, res0, res1, res2, step, 0, 0);
        for (x = 0; x < dim0; x++)
            for (y = 0; y < dim1; y++)
                for (z = 0; z < dim2; z++)
                    if (!rPixel3D(x,y,z,op,dim1,dim2) && rPixel3D(x,y,z,opened,dim1,dim2) == USHRT_MAX) {
                        wPixel3D(x,y,z,opened,dim1,dim2,gval+step*gstep);
                        count++;
        }
        step++;
    }

    free(dilat);
    free(op);
  
    return 0;   
}

// }}}
/******************************************************************************/
// {{{ erodeMirCond

unsigned short* erodeMirCond2D(unsigned short* image, unsigned short* cond, int dim0, int dim1, double res0, double res1, int step, int mode, int fast) {
    int x, y, X, Y, XX, YY, Xx, Yy, i;
    int dim2, dim3;
    unsigned long count;
    char* se;
    unsigned short* outImage; 
    unsigned short* mirImage;

    se = getCircElement(step, res0, res1);

    dim2 = dim0+2*se[0];
    dim3 = dim1+2*se[1];

    outImage = (unsigned short *)malloc(dim0*dim1*sizeof(unsigned short));
    mirImage = (unsigned short *)malloc(dim2*dim3*sizeof(unsigned short));

    for (i=0; i < dim0*dim1; i++) {
        outImage[i] = image[i];
    }

    for (y = 0; y < dim3; y++)
        for (x = 0; x < dim2; x++) {
            XX = abs(x - step); YY = abs(y - step);
            Xx = XX/dim0; Yy = YY/dim1; 
            X  = XX - 2*(XX - dim0 + 1)*Xx; 
            Y  = YY - 2*(YY - dim1 + 1)*Yy; 
            wPixel2D(x,y,mirImage,dim3,rPixel2D(X,Y,image,dim1));
    }

    for (y = 0; y < dim1; y++)
        for (x = 0; x < dim0; x++) {
            if (rPixel2D(x, y, cond, dim1) == USHRT_MAX)
                if(!fast || isNeigh2D(x, y, cond, dim0, dim1)) {
                    count=0;
                    for (Y = -se[1]; Y <= se[1]; Y++)
                        for (X = -se[0]; X <= se[0]; X++) {
                            if (rPixel2D(x+X+se[0], y+Y+se[1], mirImage, dim3) != mode*USHRT_MAX && *(se+2+count/8) & 1<<count%8) {
                                if (mode) {
                                    wPixel2D(x,y,outImage,dim1,0);
                                    X = Y = dim0;
                                }
                                else {
                                    wPixel2D(x,y,outImage,dim1,USHRT_MAX);
                                    X = Y = dim0;
                                }
                            }
                            count++;
                    }
            }
    }
    
    free(se);
    free(mirImage);

    return outImage;
}

/******************************************************************************/

unsigned short* erodeMirCond3D(unsigned short* image, unsigned short* cond, int dim0, int dim1, int dim2, double res0, double res1, double res2, int step, int mode, int fast) {
    int x, y, z, X, Y, Z, XX, YY, ZZ, Xx, Yy, Zz, i;
    int dim3, dim4, dim5;
    unsigned long count;
    char* se;
    unsigned short* outImage; 
    unsigned short* mirImage;

    se = getSphereElement(step, res0, res1, res2);
    
    dim3 = dim0+2*se[0];
    dim4 = dim1+2*se[1];
    dim5 = dim2+2*se[2];

    outImage = (unsigned short *)malloc(dim0*dim1*dim2*sizeof(unsigned short));
    mirImage = (unsigned short *)malloc(dim3*dim4*dim5*sizeof(unsigned short));
    
    for (i=0; i < dim0*dim1*dim2; i++) {
        outImage[i] = image[i];
    }

	for (z = 0; z < dim5; z++)
        for (y = 0; y < dim4; y++)
	        for (x = 0; x < dim3; x++) {
                XX = abs(x - step); YY = abs(y - step); ZZ = abs(z - step);
	            Xx = XX/dim0; Yy = YY/dim1; Zz = ZZ/dim2; 
                X  = XX - 2*(XX - dim0 + 1)*Xx; 
                Y  = YY - 2*(YY - dim1 + 1)*Yy; 
                Z  = ZZ - 2*(ZZ - dim2 + 1)*Zz;     
                wPixel3D(x,y,z,mirImage,dim4,dim5,rPixel3D(X,Y,Z,image,dim1,dim2));
    }

    for (z = 0; z < dim2; z++)
        for (y = 0; y < dim1; y++)
            for (x = 0; x < dim0; x++)
                if (rPixel3D(x,y,z,cond,dim1,dim2) == USHRT_MAX)
                    if (!fast || isNeigh3D(x, y, z, cond, dim0, dim1, dim2)) {
                        count=0;
                        for (Z = -se[2]; Z <= se[2]; Z++)
                            for (Y = -se[1]; Y <= se[1]; Y++)
                                for (X = -se[0]; X <= se[0]; X++) {
                                    if (rPixel3D(x+X+se[0], y+Y+se[1], z+Z+se[2], mirImage, dim4, dim5) != mode*USHRT_MAX && *(se+3+count/8) & 1<<count%8) {
                                        if (mode) {
                                            wPixel3D(x,y,z,outImage,dim1,dim2,0);
                                            X = Y = Z = dim0;
                                        }
                                        else {
                                            wPixel3D(x,y,z,outImage,dim1,dim2,USHRT_MAX);
                                            X = Y = Z = dim0;
                                        }
                                    }
                                    count++;
                        }
    }

    free(se);
    free(mirImage);

    return outImage;
}

// }}}
/******************************************************************************/
// {{{ getElement

char* getCircElement(int rad, double rx, double ry) {
    int x, y, i, N;
    long count=0;
    double Min;
    char* s;

/*** scale resolution towards 1 ***/
    Min = rx <= ry?rx:ry;
    rx /= Min; ry /= Min;

    N = 2+(rad*2+1)*(rad*2+1)/8+1;
    s = (char *)malloc(N);

    for (i = 0; i < N; i++) s[i] = 0;

    s[0] = (char) (rad); 
    s[1] = (char) (rad);

    for (y = -s[1]; y <= s[1]; y++)  
        for (x = -s[0]; x <= s[0]; x++) {
            if (sqrt((double)y*ry*y*ry + x*rx*x*rx) <= rad*1.01) *(s+2+count/8) |= 1<<count%8;
            count++;
    }

/*** print the structuring element ***/
//	for(i=0;i<count;i++){
//		if(i%(2*s[0]+1)==0) printf("\n");	
//
//	 	if(*(s+2+i/8)&1<<i%8)
//			printf(" 1 ");
//		else
//			printf(" 0 ");
//	}
//	printf("s[0]=%d\n",s[0]);

    return s;
}

char *getSphereElement(int rad, double rx, double ry, double rz) {
    int x, y, z, i, N;
	long count = 0;
    double Min;
	char *s;

/*** scale resolution towards 1 ***/
    Min = rx <= ry?rx:ry; Min = Min <= rz?Min:rz;
    rx/=Min; ry/=Min; rz/=Min;
	
	N = 3+(rad*2+1)*(rad*2+1)*(rad*2+1)/8+1;
	s = (char *)malloc(N);

	for (i = 0; i < N; i++) s[i] = 0;

	s[0] = (char)(rad); 
    s[1] = (char)(rad); 
    s[2] = (char)(rad);

	for (z = -s[2]; z <= s[2]; z++)
        for (y = -s[1]; y <= s[1]; y++)
            for (x = -s[0]; x <= s[0]; x++) {
                if (sqrt((double)z*rz*z*rz + y*ry*y*ry + x*rx*x*rx) <= rad*1.01) *(s+3+count/8) |= 1<<count%8;
	 	        count++;
	}

///***************** print the structuring element**************	  
//	for(i=0;i<count;i++){
//		if(i%(2*s[0]+1)==0) printf("\n");	
//		if(i%((2*s[0]+1)*(2*s[1]+1))==0) printf("\n");	
//
//	 	if(*(s+3+i/8)&1<<i%8)
//			printf(" 1 ");
//		else
//			printf(" 0 ");
//	}
//	printf("\n");
//**********/

    return s;
}

// }}}
/******************************************************************************/
// {{{ isNeigh

int isNeigh2D(int x, int y, unsigned short* image, int dim0, int dim1) {
    int i;

    for (i = 0; i < 8; i++)
        if (!isBorder2D(x+neigh8x[i], y+neigh8y[i], dim0, dim1))
            if (rPixel2D(x+neigh8x[i], y+neigh8y[i], image, dim1) != USHRT_MAX)
                return 1;    
    return 0;
}

/******************************************************************************/

int isNeigh3D(int x, int y, int z, unsigned short* image, int dim0, int dim1, int dim2) {
    int i;

    for (i = 0; i < 26; i++)
        if (!isBorder3D(x+neigh26x[i], y+neigh26y[i], z+neigh26z[i], dim0, dim1, dim2))
            if (rPixel3D(x+neigh26x[i], y+neigh26y[i], z+neigh26z[i], image, dim1, dim2) != USHRT_MAX)
                return 1;    

    return 0;
}

// }}}
/******************************************************************************/
// {{{ isBorder

int isBorder2D(int x, int y, int dim0, int dim1) {
    if (x < 0 || x >= dim0 || y < 0 || y >= dim1)
        return 1;
    else
        return 0;
}

/******************************************************************************/

int isBorder3D(int x, int y, int z, int dim0, int dim1, int dim2) {
    if (x < 0 || x >= dim0 || y < 0 || y >= dim1 || z < 0 || z >= dim2)
        return 1;
    else
        return 0;
}

// }}}
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
// {{{ cErodeCirc

int cErodeCirc2D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, double res0, double res1, int rad, int mode) {
    int x, y, X, Y, XX, YY, Xx, Yy, i;
    int dim2, dim3;
    unsigned long count;
	char* se;
    unsigned short* mirImage;

    se = getCircElement(rad, res0, res1);

    dim2 = dim0+2*se[0];
    dim3 = dim1+2*se[1];

    mirImage = (unsigned short *)malloc(dim2*dim3*sizeof(unsigned short));

    for (i=0; i < dim0*dim1; i++) {
        outImage[i] = image[i];
    }

    for (y = 0; y < dim3; y++)
        for (x = 0; x < dim2; x++) {
            XX = abs(x - rad); YY = abs(y - rad);
	        Xx = XX/dim0; Yy = YY/dim1; 
	        X  = XX - 2*(XX - dim0 + 1)*Xx; 
	    	Y  = YY - 2*(YY - dim1 + 1)*Yy; 
      	    wPixel2D(x,y,mirImage,dim3,rPixel2D(X,Y,image,dim1));
    }

    for (y = 0; y < dim1; y++)
        for (x = 0; x < dim0; x++)
            if (rPixel2D(x,y,image,dim1) == mode*USHRT_MAX) {
                count=0;
                for (Y = -se[1]; Y <= se[1]; Y++)
                    for (X = -se[0]; X <= se[0]; X++) {
                        if (rPixel2D(x+X+se[0],y+Y+se[1],mirImage,dim3) != mode*USHRT_MAX && *(se+2+count/8) & 1<<count%8) {
                            if (mode) {
                                wPixel2D(x,y,outImage,dim1,0);
                                X = Y = dim0;
                            }
                            else {
                                wPixel2D(x,y,outImage,dim1,USHRT_MAX);
                                X = Y = dim0;
                            }
                        }
                        count++;
                }
    }
    
    free(se);
    free(mirImage);
	
    return 0;
}  

/******************************************************************************/

int cErodeCirc3D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, int dim2, double res0, double res1, double res2, int rad, int mode) {
	int x, y, z, X, Y, Z, XX, YY, ZZ, Xx, Yy, Zz, i;
    int dim3, dim4, dim5;
	unsigned long count;
	char* se;
    unsigned short* mirImage;

    se = getSphereElement(rad, res0, res1, res2);
    
    dim3 = dim0+2*se[0];
    dim4 = dim1+2*se[1];
    dim5 = dim2+2*se[2];

    mirImage = (unsigned short *)malloc(dim3*dim4*dim5*sizeof(unsigned short));
    
    for (i=0; i < dim0*dim1*dim2; i++) {
        outImage[i] = image[i];
    }

    for (z = 0; z < dim5; z++)
        for (y = 0; y < dim4; y++)
            for (x = 0; x < dim3; x++) {
	    	    XX = abs(x - rad); YY = abs(y - rad); ZZ = abs(z - rad);
	    		Xx = XX/dim0; Yy = YY/dim1; Zz = ZZ/dim2; 
	    		X  = XX - 2*(XX - dim0 + 1)*Xx; 
	    		Y  = YY - 2*(YY - dim1 + 1)*Yy; 
	    		Z  = ZZ - 2*(ZZ - dim2 + 1)*Zz;     
                wPixel3D(x,y,z,mirImage,dim4,dim5,rPixel3D(X,Y,Z,image,dim1,dim2));
    }

    for (z = 0; z < dim2; z++)
        for (y = 0; y < dim1; y++)
            for (x = 0; x < dim0; x++)
                if (rPixel3D(x,y,z,image,dim1,dim2) == mode*USHRT_MAX) {
		            count=0;
		            for (Z = -se[2]; Z <= se[2]; Z++)
		                for (Y = -se[1]; Y <= se[1]; Y++)
		                    for (X = -se[0]; X <= se[0]; X++) {
                                if (rPixel3D(x+X+se[0],y+Y+se[1],z+Z+se[2],mirImage,dim4,dim5) != mode*USHRT_MAX && *(se+3+count/8) & 1<<count%8) {
                                    if (mode) {
                                        wPixel3D(x,y,z,outImage,dim1,dim2,0);
                                        X = Y = Z = dim0;
                                    }
                                    else {
                                        wPixel3D(x,y,z,outImage,dim1,dim2,USHRT_MAX);
                                        X = Y = Z = dim0;
                                    }
                                }
                                count++;
		            }
    }		

    return 0;
}  

// }}}
/******************************************************************************/
// {{{ cErodeDist

int cErodeDist2D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, double res0, double res1, int rad, int mode) {
    int status;

    status = cGetDistMap2D(image, outImage, dim0, dim1, res0, res1, 1, mode);

    if (mode) {
        bin2D((unsigned short)(rad), 0, USHRT_MAX, outImage, dim0, dim1);
    }
    else {
        bin2D(USHRT_MAX-(unsigned short)(rad), 0, USHRT_MAX, outImage, dim0, dim1);
    }

    return status;
}

/******************************************************************************/

int cErodeDist3D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, int dim2, double res0, double res1, double res2, int rad, int mode) {
    int status;

    status = cGetDistMap3D(image, outImage, dim0, dim1, dim2, res0, res1, res2, 1, mode);

    if (mode) {
        bin3D((unsigned short)(rad), 0, USHRT_MAX, outImage, dim0, dim1, dim2);
    }
    else {
        bin3D(USHRT_MAX-(unsigned short)(rad), 0, USHRT_MAX, outImage, dim0, dim1, dim2);
    }

    return status;
}

// }}}
/******************************************************************************/
// {{{ cGetDistMap

int cGetDistMap2D(unsigned short* image, unsigned short* distance, int dim0, int dim1, double res0, double res1, int gstep, int mode) {
    int x, y;
	int dist; 
    unsigned int distsq;
	unsigned int* matrix1;
	unsigned int* matrix2;
    double fact;

	matrix1 = (unsigned int*)malloc(dim0*dim1*sizeof(unsigned int));
	matrix2 = (unsigned int*)malloc(dim0*dim1*sizeof(unsigned int));

// Initialise for maximum distance.
	for (x = 0; x < dim0; x++)
        for (y = 0; y < dim1; y++)
            matrix1[x+y*dim0] = matrix2[x+y*dim0] = INT_MAX;
	
// Look for minimum distance between phases.
	for (y = 0; y < dim1; y++) {
        for (dist = 1; dist < dim0; dist++) {
            distsq = (unsigned int)dist*dist;
            for (x = 0; x < dim0 - dist; x++) {
                if (rPixel2D(x,y,image,dim1) != rPixel2D(x+dist,y,image,dim1)) {
                    if (matrix1[x+y*dim0] > distsq) {
                        matrix1[x+y*dim0] = distsq;
                    }
                    if (matrix1[x+dist+y*dim0] > distsq) {
                        matrix1[x+dist+y*dim0] = distsq;
                    }
                }
            }
        }
    }

	fact = res1/res0;
	fact *= fact;

    for (x = 0; x < dim0; x++) {
        for (dist = 0; dist < dim1; dist++) {
            distsq = (unsigned int)((double)(dist*dist)*fact);
            for (y = 0; y < dim1 - dist; y++) {
                if (rPixel2D(x,y,image,dim1) != rPixel2D(x,y+dist,image,dim1)) {
// Look for minimum distance between phases
					if (matrix2[x+y*dim0] > distsq) {
                        matrix2[x+y*dim0] = distsq;
                    }
					if (matrix2[x+(dist+y)*dim0] > distsq) {
                        matrix2[x+(dist+y)*dim0] = distsq;
                    }
                } 
// Minimum in two directions
                else {
                    if (matrix2[x+y*dim0] > distsq+matrix1[x+(dist+y)*dim0]) {
                        matrix2[x+y*dim0] = distsq+matrix1[x+(dist+y)*dim0];
                    }
					if (matrix2[x+(dist+y)*dim0] > distsq+matrix1[x+y*dim0]) {
                        matrix2[x+(dist+y)*dim0] = distsq+matrix1[x+y*dim0];
                    }
                }
            }
        }
    }
	
	for (x = 0; x < dim0; x++)
        for (y = 0; y < dim1; y++) {
            matrix2[x+y*dim0] = (unsigned int)floor(gstep*sqrt(matrix2[x+y*dim0]));
            if (mode) {
                if (rPixel2D(x,y,image,dim1) == 0) {
                    wPixel2D(x,y,distance,dim1,0);
                }
                else {
                    wPixel2D(x,y,distance,dim1,(unsigned short)(matrix2[x+y*dim0]));
                }
            }
            else {
                if (rPixel2D(x,y,image,dim1) == USHRT_MAX) {
                    wPixel2D(x,y,distance,dim1,USHRT_MAX);
                }
                else {
                    wPixel2D(x,y,distance,dim1,USHRT_MAX-(unsigned short)(matrix2[x+y*dim0]));
                }
            }
    }

    free(matrix1);
    free(matrix2);

    return 0;
}

/******************************************************************************/

int cGetDistMap3D(unsigned short* image, unsigned short* distance, int dim0, int dim1, int dim2, double res0, double res1, double res2, int gstep, int mode) {
    int x, y, z;
	int dist; 
    unsigned int distsq;
	unsigned int* matrix1;
	unsigned int* matrix2;
    double fact;

	matrix1 = (unsigned int*)malloc(dim0*dim1*dim2*sizeof(unsigned int));
	matrix2 = (unsigned int*)malloc(dim0*dim1*dim2*sizeof(unsigned int));

// Initialise for maximum distance.
	for (x = 0; x < dim0; x++)
        for (y = 0; y < dim1; y++)
			for (z = 0; z < dim2; z++)
				matrix1[x+y*dim0+z*dim1*dim0] = matrix2[x+y*dim0+z*dim1*dim0] = INT_MAX;
	
// Look for minimum distance between phases.
	for (y = 0; y < dim1; y++)
		for (z = 0; z < dim2; z++) {
			for (dist = 1; dist < dim0; dist++) {
				distsq = (unsigned int)dist*dist;
				for (x = 0; x < dim0-dist; x++) {
					if (rPixel3D(x,y,z,image,dim1,dim2) != rPixel3D(x+dist,y,z,image,dim1,dim2)) {
						if (matrix1[x+y*dim0+z*dim1*dim0] > distsq) {
							matrix1[x+y*dim0+z*dim1*dim0] = distsq;
						}
						if (matrix1[x+dist+y*dim0+z*dim1*dim0] > distsq) {
							matrix1[x+dist+y*dim0+z*dim1*dim0] = distsq;
						}	
					}
				}
			}
		}

	fact = res1/res0;
	fact *=fact;

	for (x = 0; x < dim0; x++)
		for (z = 0; z < dim2; z++) {
			for (dist = 0; dist < dim1; dist++) {
				distsq = (unsigned int)((double)(dist*dist)*fact);
				for (y = 0; y < dim1 - dist; y++) {
					if (rPixel3D(x,y,z,image,dim1,dim2) != rPixel3D(x,y+dist,z,image,dim1,dim2)) {
// Look for minimum distance between phases
			            if (matrix2[x+y*dim0+z*dim1*dim0] > distsq) {
							matrix2[x+y*dim0+z*dim1*dim0] = distsq;
						}
						if (matrix2[x+(dist+y)*dim0+z*dim1*dim0] > distsq) {
							matrix2[x+(dist+y)*dim0+z*dim1*dim0] = distsq;
						}
					}
// Minimum in two directions
                    else {
                        if (matrix2[x+y*dim0+z*dim1*dim0] > distsq+matrix1[x+(dist+y)*dim0+z*dim1*dim0]){
							matrix2[x+y*dim0+z*dim1*dim0] = distsq+matrix1[x+(dist+y)*dim0+z*dim1*dim0];
						}
						if (matrix2[x+(dist+y)*dim0+z*dim1*dim0] > distsq+matrix1[x+y*dim0+z*dim1*dim0]){
							matrix2[x+(dist+y)*dim0+z*dim1*dim0] = distsq+matrix1[x+y*dim0+z*dim1*dim0];
						}
					}
				}
			}
		}
	for (x = 0; x < dim0; x++)
		for(y = 0;y < dim1; y++)
			for (z = 0; z < dim2; z++)
				matrix1[x+y*dim0+z*dim1*dim0] = UINT_MAX;

	fact = res2/res0;
	fact *=fact;

	for (x = 0; x < dim0; x++)
		for (y = 0; y < dim1; y++) {
			for (dist = 0; dist < dim2; dist++) {
				distsq = (unsigned int)((double)(dist*dist)*fact);
				for (z = 0; z < dim2 - dist; z++) {
					if (rPixel3D(x,y,z,image,dim1,dim2) != rPixel3D(x,y,z+dist,image,dim1,dim2)) {
						if (matrix1[x+y*dim0+z*dim1*dim0] > distsq) {
							matrix1[x+y*dim0+z*dim1*dim0] = distsq;
							
						}
						if (matrix1[x+y*dim0+(dist+z)*dim1*dim0] > distsq) {
							matrix1[x+y*dim0+(dist+z)*dim1*dim0] = distsq;
							
						}	
					}
                    else {
						if (matrix1[x+y*dim0+z*dim1*dim0] > distsq+matrix2[x+y*dim0+(dist+z)*dim1*dim0]){
							matrix1[x+y*dim0+z*dim1*dim0] = distsq+matrix2[x+y*dim0+(dist+z)*dim1*dim0];
						}
						if (matrix1[x+y*dim0+(dist+z)*dim1*dim0] > distsq+matrix2[x+y*dim0+z*dim1*dim0]){
							matrix1[x+y*dim0+(dist+z)*dim1*dim0] = distsq+matrix2[x+y*dim0+z*dim1*dim0];
								
						}
					}
					
				}
			}
    }

	for (x = 0; x < dim0; x++)
		for (y = 0; y < dim1; y++)
			for (z = 0; z < dim2; z++) {
			    matrix1[x+y*dim0+z*dim1*dim0] = (unsigned int)floor(gstep*sqrt(matrix1[x+y*dim0+z*dim1*dim0]));
                if (mode) {
                    if (rPixel3D(x,y,z,image,dim1,dim2) == 0) {
                        wPixel3D(x,y,z,distance,dim1,dim2,0);
                    }
                    else {
                        wPixel3D(x,y,z,distance,dim1,dim2,(unsigned short)(matrix1[x+y*dim0+z*dim1*dim0]));
                    }
                }
                else {
                    if (rPixel3D(x,y,z,image,dim1,dim2) == USHRT_MAX) {
                        wPixel3D(x,y,z,distance,dim1,dim2,USHRT_MAX);
                    }
                    else {
                        wPixel3D(x,y,z,distance,dim1,dim2,USHRT_MAX-(unsigned short)(matrix1[x+y*dim0+z*dim1*dim0]));
                    }
                }
    }

	free(matrix1);
	free(matrix2);

    return 0;
}

// }}}
/******************************************************************************/
