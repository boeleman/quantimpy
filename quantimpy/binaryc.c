#include <binaryc.h>

/*8 Neighbours starting at left the central pixel*/
static int neigh8x[8] = {-1,-1, 0, 1,1,1,0,-1};
static int neigh8y[8] = { 0,-1,-1,-1,0,1,1, 1};

int cGetDistOpenMap2D(unsigned short* image, unsigned short* distance, unsigned short* opened, int dim0, int dim1, double res0, double res1, int gval, int gstep)
{
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
        dilat = erodeMirCond(image, distance, dim0, dim1, res0, res1, step, 1, 1);
        for (x = 0; x < dim0; x++)
            for (y = 0; y < dim1; y++)
                if (!rPixel(x,y,dilat,dim0) && rPixel(x,y,distance,dim0) == USHRT_MAX) {
                    wPixel(x,y,distance,dim0,gval+step*gstep);
                    count++;
        }
    
        for (i=0; i < dim0*dim1; i++)
            dilat[i] = distance[i];

        bin(USHRT_MAX-1, 0, USHRT_MAX, dilat, dim0, dim1);
        op = erodeMirCond(dilat, opened, dim0, dim1, res0, res1, step, 0,0);
        for (x = 0; x < dim0; x++)
            for (y = 0; y < dim1; y++)
                if (!rPixel(x, y, op, dim0) && rPixel(x, y, opened, dim0) == USHRT_MAX) {
                    wPixel(x, y, opened, dim0, gval+step*gstep);
                    count++;
        }

        step++;
    }

    free(dilat);
    free(op);

    return 0;
}

unsigned short* erodeMirCond(unsigned short* image, unsigned short* cond, int dim0, int dim1, double res0, double res1, int step, int mode, int fast) {
    int x, y, X, Y, XX, YY, Xx, Yy, i;
    int dim2, dim3;
    unsigned long count;
//    unsigned char* se;
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

    for (y = 0; y < dim2; y++)
        for (x = 0; x < dim3; x++) {
            XX = abs(x-step); YY=abs(y-step);
            Xx = XX/dim0; Yy=YY/dim1; 
            X  = XX-2*(XX-dim0+1)*Xx; 
            Y  = YY-2*(YY-dim1+1)*Yy; 
            wPixel(x,y,mirImage,dim2,rPixel(X,Y,image,dim0));
    }

    for (y = 0; y < dim1; y++)
        for (x = 0; x < dim0; x++) {
            if (rPixel(x, y, cond, dim0) == USHRT_MAX)
                if(!fast || isNeigh(x, y, cond, dim0, dim1)) {
                    count=0;
                    for (Y = -se[1]; Y <= se[1]; Y++)
                        for (X = -se[0]; X <= se[0]; X++) {
                            if (rPixel(x+X+se[0], y+Y+se[1], mirImage, dim2) != mode*USHRT_MAX && *(se+2+count/8) & 1<<count%8) {
                                if (mode) {
                                    wPixel(x,y,outImage,dim0,0);
                                    X = Y = dim0;
                                }
                                else {
                                    wPixel(x,y,outImage,dim0,USHRT_MAX);
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
            if (sqrt((double)y*ry*y*ry+x*rx*x*rx) <= rad*1.01) *(s+2+count/8) |= 1<<count%8;
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

//char *getCircElement(int rad, double rx, double ry) {
//    int x, y, i, N;
//	long count=0;
//	double rady;
//	char *s;
//
//	rady = rad*rx/ry;
//
//	N = 2+((int)rad*2+1)*((int)rady*2+1)/8+1;
//	s = (char *)malloc(N);
//
//	for (i = 0; i < N; i++) s[i]=0;
//	s[0] = (char)rad; 
//    s[1] = (char)rady;
//	
//	for (y = -s[1]; y <= s[1]; y++)
//        for (x = -s[0]; x <= s[0]; x++) {
//            if (y*ry*y*ry+x*rx*x*rx <= rad*rad) *(s+2+count/8) |= 1<<count%8;
//	 	count++;
//	}
//	
//    return s;
//}

int isNeigh(int x, int y, unsigned short* image, int dim0, int dim1) {
    int i;

    for (i = 0; i < 8; i++)
        if (!isBorder(x+neigh8x[i], y+neigh8y[i], dim0, dim1))
            if (rPixel(x+neigh8x[i], y+neigh8y[i], image, dim0) != USHRT_MAX)
                return 1;    
    return 0;
}

int isBorder(int x, int y, int dim0, int dim1) {
    if (x < 0 || x >= dim0 || y < 0 || y >= dim1)
        return 1;
    else
        return 0;
}

unsigned short rPixel(int x, int y, unsigned short* image, int dim0) {
    int i;

    i = y*dim0 + x;

    return image[i];
}

void wPixel(int x, int y, unsigned short* image, int dim0, unsigned short value) {
    int i;

    i = y*dim0 + x;

    image[i] = value;
}

void bin(int LOW, int value1, int value2, unsigned short* image, int dim0, int dim1) {
    int x, y, val;

    val = value2>USHRT_MAX? -1 : value2;

    for (y = 0; y < dim1; y++)
        for (x = 0; x < dim0; x++) {
            if (rPixel(x, y, image, dim0) <= LOW)
                wPixel(x, y, image, dim0, value1);
            else if (val != -1)
                wPixel(x, y, image, dim0, val);
    }
}

