#include <quantimpyc.h>
#include <morphologyc.h>

/******************************************************************************/
// {{{ cErode

int cErode2D(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dist, double res0, double res1) {
    int status;

    status = cGetMap2D(image, erosion, dim0, dim1, res0, res1, 1);

    bin2D(dist, 0, USHRT_MAX, erosion, dim0, dim1);

    return status > 0 ? status : 0;
}

/******************************************************************************/

int cErode3D(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2) {
    int status;

    status = cGetMap3D(image, erosion, dim0, dim1, dim2, res0, res1, res2, 1);

    bin3D(dist, 0, USHRT_MAX, erosion, dim0, dim1, dim2);

    return status > 0 ? status : 0;
}

// }}}
/******************************************************************************/
// {{{ cDilate

int cDilate2D(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dist, double res0, double res1) {
    int status;

    status = cGetMap2D(image, dilation, dim0, dim1, res0, res1, 0);

    bin2D(USHRT_MAX-dist-1, 0, USHRT_MAX, dilation, dim0, dim1);

    return status > 0 ? status : 0;
}

/******************************************************************************/

int cDilate3D(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2) {
    int status;

    status = cGetMap3D(image, dilation, dim0, dim1, dim2, res0, res1, res2, 0);

    bin3D(USHRT_MAX-dist-1, 0, USHRT_MAX, dilation, dim0, dim1, dim2);

    return status > 0 ? status : 0;
}

// }}}
/******************************************************************************/
// {{{ cOpenMap

int cOpenMap2D(unsigned short* erosion, unsigned short* opening, int dim0, int dim1, double res0, double res1) {
    int x, y, i;
    int step;
    unsigned long count;
    unsigned short* image; 

    count = 1;
    step  = 1;
    
    image = (unsigned short *)malloc(dim0*dim1*sizeof(unsigned short));

    for (i = 0; i < dim0*dim1; i++) {
        opening[i] = erosion[i];
        image[i]   = erosion[i];
    }

    while (count) {
        count = 0;
        printf("\r Dilation step : %d \n",step);

        for (i = 0; i < dim0*dim1; i++) {
            image[i]   = erosion[i];
        }

        bin2D(step, 0, USHRT_MAX, image, dim0, dim1);
        
        cDilate2D(image, image, dim0, dim1, step, res0, res1);
        
        for (y = 0; y < dim1; y++)
            for (x = 0; x < dim0; x++)
                if (rPixel2D(x, y, erosion, dim1) != USHRT_MAX && rPixel2D(x, y, image, dim1) != 0) {
                    wPixel2D(x, y, opening, dim1, step);
                    count++;
        }

        step++;
    }

    free(image);

    return 0;
}

/******************************************************************************/

int cOpenMap3D(unsigned short* erosion, unsigned short* opening, int dim0, int dim1, int dim2, double res0, double res1, double res2) {
    int x, y, z, i;
    int step;
    unsigned long count;
    unsigned short* image; 

    count = 1;
    step  = 1;
    
    image = (unsigned short *)malloc(dim0*dim1*dim2*sizeof(unsigned short));

    for (i = 0; i < dim0*dim1*dim2; i++) {
        opening[i] = erosion[i];
        image[i]   = erosion[i];
    }

    while (count) {
        count = 0;
        printf("\r Dilation step : %d \n",step);

        for (i = 0; i < dim0*dim1*dim2; i++) {
            image[i] = erosion[i];
        }

        bin3D(step, 0, USHRT_MAX, image, dim0, dim1, dim2);
        
        cDilate3D(image, image, dim0, dim1, dim2, step, res0, res1, res2);
        
        for (z = 0; z < dim2; z++)
            for (y = 0; y < dim1; y++)
                for (x = 0; x < dim0; x++)
                    if (rPixel3D(x,y,z,erosion,dim1,dim2) != USHRT_MAX && rPixel3D(x,y,z,image,dim1,dim2) != 0) {
                        wPixel3D(x,y,z,opening,dim1,dim2, step);
                        count++;
        }

        step++;
    }

    free(image);

    return 0;
}

// }}}
/******************************************************************************/
// {{{ cCloseMap

int cCloseMap2D(unsigned short* dilation, unsigned short* closing, int dim0, int dim1, double res0, double res1) {
    int x, y, i;
    int step;
    unsigned long count;
    unsigned short* image; 

    count = 1;
    step  = 1;
    
    image = (unsigned short *)malloc(dim0*dim1*sizeof(unsigned short));

    for (i = 0; i < dim0*dim1; i++) {
        closing[i] = dilation[i];
        image[i]   = dilation[i];
    }

    while (count) {
        count = 0;
        printf("\r Erosion step : %d \n",step);

        for (i = 0; i < dim0*dim1; i++) {
            image[i] = dilation[i];
        }

        bin2D(USHRT_MAX-step-1, 0, USHRT_MAX, image, dim0, dim1);

        cErode2D(image, image, dim0, dim1, step, res0, res1);
        
        for (y = 0; y < dim1; y++)
            for (x = 0; x < dim0; x++)
                if (rPixel2D(x, y, dilation, dim1) != 0 && rPixel2D(x, y, image, dim1) != USHRT_MAX) {
                    wPixel2D(x, y, closing, dim1, USHRT_MAX-step);
                    count++;
        }

        step++;
    }

    free(image);

    return 0;
}

/******************************************************************************/

int cCloseMap3D(unsigned short* dilation, unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2) {
    int x, y, z, i;
    int step;
    unsigned long count;
    unsigned short* image; 

    count = 1;
    step  = 1;
    
    image = (unsigned short *)malloc(dim0*dim1*dim2*sizeof(unsigned short));

    for (i = 0; i < dim0*dim1*dim2; i++) {
        closing[i] = dilation[i];
        image[i]   = dilation[i];
    }

    while (count) {
        count = 0;
        printf("\r Erosion step : %d \n",step);

        for (i = 0; i < dim0*dim1*dim2; i++) {
            image[i] = dilation[i];
        }

        bin3D(USHRT_MAX-step-1, 0, USHRT_MAX, image, dim0, dim1, dim2);

        cErode3D(image, image, dim0, dim1, dim2, step, res0, res1, res2);
        
        for (z = 0; z < dim2; z++)
            for (y = 0; y < dim1; y++)
                for (x = 0; x < dim0; x++)
                    if (rPixel3D(x,y,z,dilation,dim1,dim2) != 0 && rPixel3D(x,y,z,image,dim1,dim2) != USHRT_MAX) {
                        wPixel3D(x,y,z,closing,dim1,dim2,USHRT_MAX-step);
                        count++;
        }

        step++;
    }

    free(image);

    return 0;
}

// }}}
/******************************************************************************/
// {{{ cGetMap

int cGetMap2D(unsigned short* image, unsigned short* distance, int dim0, int dim1, double res0, double res1, int mode) {
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
            matrix2[x+y*dim0] = (unsigned int)ceil(sqrt(matrix2[x+y*dim0]));
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

int cGetMap3D(unsigned short* image, unsigned short* distance, int dim0, int dim1, int dim2, double res0, double res1, double res2, int mode) {
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
			    matrix1[x+y*dim0+z*dim1*dim0] = (unsigned int)ceil(sqrt(matrix1[x+y*dim0+z*dim1*dim0]));
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
