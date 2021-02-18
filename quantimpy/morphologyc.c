#include <quantimpyc.h>
#include <morphologyc.h>

/******************************************************************************/
// {{{ c_erode

int c_erode_2d(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dist, double res0, double res1) {
    int status;

    status = c_get_map_2d(image, erosion, dim0, dim1, res0, res1, 1);

    bin_2d(dist, 0, USHRT_MAX, erosion, dim0, dim1);

    return status > 0 ? status : 0;
}

/******************************************************************************/

int c_erode_3d(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2) {
    int status;

    status = c_get_map_3d(image, erosion, dim0, dim1, dim2, res0, res1, res2, 1);

    bin_3d(dist, 0, USHRT_MAX, erosion, dim0, dim1, dim2);

    return status > 0 ? status : 0;
}

// }}}
/******************************************************************************/
// {{{ c_dilate

int c_dilate_2d(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dist, double res0, double res1) {
    int status;

    status = c_get_map_2d(image, dilation, dim0, dim1, res0, res1, 0);

    bin_2d(USHRT_MAX-dist-1, 0, USHRT_MAX, dilation, dim0, dim1);

    return status > 0 ? status : 0;
}

/******************************************************************************/

int c_dilate_3d(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2) {
    int status;

    status = c_get_map_3d(image, dilation, dim0, dim1, dim2, res0, res1, res2, 0);

    bin_3d(USHRT_MAX-dist-1, 0, USHRT_MAX, dilation, dim0, dim1, dim2);

    return status > 0 ? status : 0;
}

// }}}
/******************************************************************************/
// {{{ c_open_map

int c_open_map_2d(unsigned short* erosion, unsigned short* opening, int dim0, int dim1, double res0, double res1) {
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

        bin_2d(step, 0, USHRT_MAX, image, dim0, dim1);
        
        c_dilate_2d(image, image, dim0, dim1, step, res0, res1);
        
        for (y = 0; y < dim1; y++)
            for (x = 0; x < dim0; x++)
                if (r_pixel_2d(x, y, erosion, dim1) != USHRT_MAX && r_pixel_2d(x, y, image, dim1) != 0) {
                    w_pixel_2d(x, y, opening, dim1, step);
                    count++;
        }

        step++;
    }

    free(image);

    return 0;
}

/******************************************************************************/

int c_open_map_3d(unsigned short* erosion, unsigned short* opening, int dim0, int dim1, int dim2, double res0, double res1, double res2) {
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

        bin_3d(step, 0, USHRT_MAX, image, dim0, dim1, dim2);
        
        c_dilate_3d(image, image, dim0, dim1, dim2, step, res0, res1, res2);
        
        for (z = 0; z < dim2; z++)
            for (y = 0; y < dim1; y++)
                for (x = 0; x < dim0; x++)
                    if (r_pixel_3d(x,y,z,erosion,dim1,dim2) != USHRT_MAX && r_pixel_3d(x,y,z,image,dim1,dim2) != 0) {
                        w_pixel_3d(x,y,z,opening,dim1,dim2, step);
                        count++;
        }

        step++;
    }

    free(image);

    return 0;
}

// }}}
/******************************************************************************/
// {{{ c_close_map

int c_close_map_2d(unsigned short* dilation, unsigned short* closing, int dim0, int dim1, double res0, double res1) {
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

        bin_2d(USHRT_MAX-step-1, 0, USHRT_MAX, image, dim0, dim1);

        c_erode_2d(image, image, dim0, dim1, step, res0, res1);
        
        for (y = 0; y < dim1; y++)
            for (x = 0; x < dim0; x++)
                if (r_pixel_2d(x, y, dilation, dim1) != 0 && r_pixel_2d(x, y, image, dim1) != USHRT_MAX) {
                    w_pixel_2d(x, y, closing, dim1, USHRT_MAX-step);
                    count++;
        }

        step++;
    }

    free(image);

    return 0;
}

/******************************************************************************/

int c_close_map_3d(unsigned short* dilation, unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2) {
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

        bin_3d(USHRT_MAX-step-1, 0, USHRT_MAX, image, dim0, dim1, dim2);

        c_erode_3d(image, image, dim0, dim1, dim2, step, res0, res1, res2);
        
        for (z = 0; z < dim2; z++)
            for (y = 0; y < dim1; y++)
                for (x = 0; x < dim0; x++)
                    if (r_pixel_3d(x,y,z,dilation,dim1,dim2) != 0 && r_pixel_3d(x,y,z,image,dim1,dim2) != USHRT_MAX) {
                        w_pixel_3d(x,y,z,closing,dim1,dim2,USHRT_MAX-step);
                        count++;
        }

        step++;
    }

    free(image);

    return 0;
}

// }}}
/******************************************************************************/
// {{{ c_get_map

int c_get_map_2d(unsigned short* image, unsigned short* distance, int dim0, int dim1, double res0, double res1, int mode) {
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
                if (r_pixel_2d(x,y,image,dim1) != r_pixel_2d(x+dist,y,image,dim1)) {
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
                if (r_pixel_2d(x,y,image,dim1) != r_pixel_2d(x,y+dist,image,dim1)) {
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
                if (r_pixel_2d(x,y,image,dim1) == 0) {
                    w_pixel_2d(x,y,distance,dim1,0);
                }
                else {
                    w_pixel_2d(x,y,distance,dim1,(unsigned short)(matrix2[x+y*dim0]));
                }
            }
            else {
                if (r_pixel_2d(x,y,image,dim1) == USHRT_MAX) {
                    w_pixel_2d(x,y,distance,dim1,USHRT_MAX);
                }
                else {
                    w_pixel_2d(x,y,distance,dim1,USHRT_MAX-(unsigned short)(matrix2[x+y*dim0]));
                }
            }
    }

    free(matrix1);
    free(matrix2);

    return 0;
}

/******************************************************************************/

int c_get_map_3d(unsigned short* image, unsigned short* distance, int dim0, int dim1, int dim2, double res0, double res1, double res2, int mode) {
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
					if (r_pixel_3d(x,y,z,image,dim1,dim2) != r_pixel_3d(x+dist,y,z,image,dim1,dim2)) {
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
					if (r_pixel_3d(x,y,z,image,dim1,dim2) != r_pixel_3d(x,y+dist,z,image,dim1,dim2)) {
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
					if (r_pixel_3d(x,y,z,image,dim1,dim2) != r_pixel_3d(x,y,z+dist,image,dim1,dim2)) {
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
                    if (r_pixel_3d(x,y,z,image,dim1,dim2) == 0) {
                        w_pixel_3d(x,y,z,distance,dim1,dim2,0);
                    }
                    else {
                        w_pixel_3d(x,y,z,distance,dim1,dim2,(unsigned short)(matrix1[x+y*dim0+z*dim1*dim0]));
                    }
                }
                else {
                    if (r_pixel_3d(x,y,z,image,dim1,dim2) == USHRT_MAX) {
                        w_pixel_3d(x,y,z,distance,dim1,dim2,USHRT_MAX);
                    }
                    else {
                        w_pixel_3d(x,y,z,distance,dim1,dim2,USHRT_MAX-(unsigned short)(matrix1[x+y*dim0+z*dim1*dim0]));
                    }
                }
    }

	free(matrix1);
	free(matrix2);

    return 0;
}

// }}}
/******************************************************************************/
