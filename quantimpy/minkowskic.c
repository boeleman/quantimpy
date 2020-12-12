#include <quantimpyc.h>
#include <minkowskic.h>

/******************************************************************************/
// {{{ cFunctionals

int cFunctionals2D(unsigned short* image, int dim0, int dim1, double res0, double res1, double* area, double* length, double* euler4, double* euler8) {
    double norm;
    long int* h;

    norm = (double)(dim0-1)/dim0 * (dim1-1)/dim1;

    h = quant2D(image, dim0, dim1);

    *area   = norm * areaDens(h);
    *length = norm * lengthDens(h,res0,res1);
    *euler4 = norm * euler4Dens(h,res0,res1);
    *euler8 = norm * euler8Dens(h,res0,res1);

    free(h);

    return 0;
}

/******************************************************************************/

int cFunctionals3D(unsigned short* image, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* volume, double* surface, double* curvature, double* euler6, double* euler26) {
    double norm;
    long int* h;

    norm = (double)(dim0-1)/dim0 * (dim1-1)/dim1 * (dim2-1)/dim2;
	
    h = quant3D(image, dim0, dim1, dim2);

    *volume    = norm * volumeDens(h);
    *surface   = norm * surfaceDens(h, res0, res1, res2);
    *curvature = norm * curvatureDens(h, res0, res1, res2);
    *euler6    = norm * euler6Dens(h, res0, res1, res2);
    *euler26   = norm * euler26Dens(h, res0, res1, res2);

    free(h);

    return 0;
}

// }}}
/******************************************************************************/
// {{{ quant

long int* quant2D(unsigned short* image, int dim0, int dim1) {
    int x, y, i;
    int mask;
	long int *h;
	
    h = (long int*)malloc(16*sizeof(long int));

 	for (i = 0; i < 16; i++) h[i] = 0;

    for (x = 0; x < dim0 - 1; x++) {
        mask =  (rPixel2D(x  ,0,image,dim1) == USHRT_MAX) 
             + ((rPixel2D(x+1,0,image,dim1) == USHRT_MAX) << 1); 
        for (y = 1; y < dim1; y++) {
            mask += ((rPixel2D(x  ,y,image,dim1) == USHRT_MAX) << 2) 
                  + ((rPixel2D(x+1,y,image,dim1) == USHRT_MAX) << 3);
		    h[mask]++;
		    mask >>= 2;
        }
 	}
 	
    return h;
}


/******************************************************************************/

long int* quant3D(unsigned short* image, int dim0, int dim1, int dim2) {
	int x, y, z, i;
    int mask;
	long int *h;
	
	h = (long int*)malloc(256*sizeof(long int));

 	for (i = 0; i < 256; i++) h[i] = 0;

    for (x = 0; x < dim0-1; x++)
        for (y = 0; y < dim1-1; y++) {
		    mask =  (rPixel3D(x  ,y  ,0,image,dim1,dim2) == USHRT_MAX) 
                 + ((rPixel3D(x+1,y  ,0,image,dim1,dim2) == USHRT_MAX) << 1) 
                 + ((rPixel3D(x  ,y+1,0,image,dim1,dim2) == USHRT_MAX) << 2)
                 + ((rPixel3D(x+1,y+1,0,image,dim1,dim2) == USHRT_MAX) << 3); 
    		for (z = 1; z < dim2; z++) {
                mask += ((rPixel3D(x  ,y  ,z,image,dim1,dim2) == USHRT_MAX) << 4)
                      + ((rPixel3D(x+1,y  ,z,image,dim1,dim2) == USHRT_MAX) << 5)
                      + ((rPixel3D(x  ,y+1,z,image,dim1,dim2) == USHRT_MAX) << 6)
                      + ((rPixel3D(x+1,y+1,z,image,dim1,dim2) == USHRT_MAX) << 7);
    		    h[mask]++;
		        mask >>= 4;
		    }
    }
    
    return h;
}

// }}}
/******************************************************************************/
// {{{ areaDens

double areaDens(long int *h) {
    int i;
	unsigned long int iChi = 0, iVol = 0;
 
    for (i = 0; i < 16; i++) {
        iChi += (i&1)*h[i];
		iVol += h[i];
	}

	if(!iVol) return 0;
 	else return (double)iChi/iVol;
}

// }}}
/******************************************************************************/
// {{{ lengthDens

double lengthDens(long int *h, double res0, double res1) {
	unsigned int i, l;
	long int  numpix=0, ii;
	double II=0, LI=0, w[4], r[4];
	int kl[4][2] = {{1,2},{1,4},{1,8},{2,4}};
	
	r[0] = res0;
	r[1] = res1;
	r[2] = r[3] = sqrt(r[0]*r[0] + r[1]*r[1]);

	w[0] = atan(res0/res1)/M_PI;
	w[1] = atan(res1/res0)/M_PI;
	w[2] = w[3] = (1 - w[0] - w[1])/2;

	for (l = 0; l < 16; l++) numpix += h[l];

	for (i = 0; i < 4; i++) {
        ii = 0;
 	    for (l = 0; l < 16; l++) {
	        ii += h[l] * (l == (l|kl[i][0])) * (0 == (l&kl[i][1]));
		    ii += h[l] * (l == (l|kl[i][1])) * (0 == (l&kl[i][0]));
	    }	  
	    II += (double)w[i]*ii;
	    LI += (double)w[i]*numpix*r[i];
	}

	if(!LI) return 0;
	else return M_PI/2 * II/LI;
}

// }}}
/******************************************************************************/
// {{{ volumeDens

double volumeDens(long int *h) {
	int i;
	unsigned long int iChi = 0, iVol = 0;

	for (i = 0; i < 256; i++) {
        iVol += h[i]; 
	    if (i&1) iChi += h[i]; 
	}

	if(!iVol) return 0;
	else return (double)iChi/(double)iVol;
}

// }}}
/******************************************************************************/
// {{{ surfaceDens

double surfaceDens(long int *h, double res0, double res1, double res2) {
    int i, l;
	unsigned long sv, le=0;
	double wi[13], r[13], Sv, Lv, *Delta, *weight;	
	int kl[13][2] = {{1,2},{1,4},{1,16},{1,8},{2,4},{1,32},{2,16},{1,64},{4,16},{1,128},{2,64},{4,32},{8,16}};

	Delta  = (double *)malloc(3*sizeof(double)); 
	weight = (double *)malloc(7*sizeof(double));

	for (i = 0; i < 7; i++) weight[i] = 0;

	r[0] = Delta[0] = res0; 
	r[1] = Delta[1] = res1; 
	r[2] = Delta[2] = res2;
	r[3] = r[4]  = sqrt(r[0]*r[0] + r[1]*r[1]);
	r[5] = r[6]  = sqrt(r[0]*r[0] + r[2]*r[2]);
	r[7] = r[8]  = sqrt(r[1]*r[1] + r[2]*r[2]);
	r[9] = r[10] = r[11] = r[12] = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);

	weights(Delta, weight);
 	wi[0]  = weight[0]; wi[1] = weight[1]; wi[2]  = weight[2]; wi[3]  = weight[3];
	wi[4]  = weight[3]; wi[5] = weight[5]; wi[6]  = weight[5]; wi[7]  = weight[4];
	wi[8]  = weight[4]; wi[9] = weight[6]; wi[10] = weight[6]; wi[11] = weight[6];
	wi[12] = weight[6];

    for (l = 0; l < 256; l++) le += h[l];

    Sv = Lv =0;
    for (i = 0; i < 13; i++) {
        if (wi[i]) {
            sv = 0;
            for (l = 0; l < 256; l++) {
                sv += h[l] * (l == (l|kl[i][0])) * (0 == (l&kl[i][1]));
	            sv += h[l] * (l == (l|kl[i][1])) * (0 == (l&kl[i][0]));
 	        }
	        Sv += wi[i]*(double)sv;
	        Lv += wi[i]*(double)le*r[i];
 	    }
 	}

	free(Delta);
	free(weight);

	if(!Lv) return 0;
	else return 2.0 * Sv/Lv;
}

// }}}
/******************************************************************************/
// {{{ curvatureDens

double curvatureDens(long int *h, double res0, double res1, double res2) {
/* Mean curvature in 3D is related to the 2D-Euler number on a 2D section plane.
 * Within a 2x2x2 cube 13 different planes can be defined (see lang+99). The
 * results for the different planes are weighted by the sin of the plane to the
 * vertical axis */
    int i, k, l;
	unsigned long iVol = 0;
  	int kr[9][4] = {{1, 2, 4,  8},{1, 2,16,32},{1, 4,16, 64},
                    {1, 2,64,128},{4,16, 8,32},{1,32, 4,128},
                    {2, 8,16, 64},{2, 4,32,64},{1,16, 8,128}};
	int kt[8][3] = {{1,64, 32},{2,16,128},{8,64,32},{4,16,128},
			        {2, 4,128},{8, 1, 64},{2, 4,16},{8, 1,32}};
	double wi[13], a[13], r[13], s, mc, atr;
	double r01, r02, r12;
	double *Delta, *weight;

	Delta = (double *)malloc(3*sizeof(double));

	mc = 0;

	r[0] = Delta[0] = res0; 
	r[1] = Delta[1] = res1; 
	r[2] = Delta[2] = res2;
	r01 = sqrt(r[0]*r[0] + r[1]*r[1]);
	r02 = sqrt(r[0]*r[0] + r[2]*r[2]);
	r12 = sqrt(r[1]*r[1] + r[2]*r[2]);
	s = (r01 + r02 + r12)/2;	

	a[0] = r[0]*r[1]; a[1] = r[0]*r[2]; a[2] = r[1]*r[2];
	a[3] = a[4] = r[2]*r01;
	a[5] = a[6] = r[1]*r02;
	a[7] = a[8] = r[0]*r12;
	atr = sqrt(s*(s-r01)*(s-r02)*(s-r12));
	a[9] = a[10] = a[11] = a[12] = 2*atr;

	weight=(double *)malloc(7*sizeof(double));

	for (i = 0; i < 7; i++) weight[i] = 0;

	weights(Delta, weight);
 	wi[0]  = weight[0]; wi[1] = weight[1]; wi[2]  = weight[2]; wi[3]  = weight[3];
	wi[4]  = weight[3]; wi[5] = weight[5]; wi[6]  = weight[5]; wi[7]  = weight[4];
	wi[8]  = weight[4]; wi[9] = weight[6]; wi[10] = weight[6]; wi[11] = weight[6];
	wi[12] = weight[6];


	for (l = 0; l < 256; l++) {
	    iVol += h[l];
		for (i = 0; i < 9; i++)
		    for (k = 0; k < 4; k++)
                mc += (double)h[l]*wi[i]/(4*a[i])
			        * ((l==(l|kr[i][k])) * (0==(l&kr[i][(k+1)%4]))
			                             * (0==(l&kr[i][(k+2)%4]))
                                         * (0==(l&kr[i][(k+3)%4]))
			        -  (l==(l|kr[i][k])) * (l==(l|kr[i][(k+1)%4]))
                                         * (l==(l|kr[i][(k+2)%4]))
                                         * (0==(l&kr[i][(k+3)%4])));
		for (i = 9; i < 13; i++)
            for (k = 0; k < 3; k++)
                mc += (double)h[l]*wi[i]/(3*a[i])
			        * ((l==(l|kt[i-9][k])) * (0==(l&kt[i-9][(k+1)%3]))
			                               * (0==(l&kt[i-9][(k+2)%3]))
                    -  (l==(l|kt[i-5][k])) * (l==(l|kt[i-5][(k+1)%3]))
			                               * (0==(l&kt[i-5][(k+2)%3])));
    }

//	return (double)4*M_PI*mc/(double)(iVol*res0*res1*res2);
    return (double)4*M_PI*mc/(double)(iVol);
}

// }}}
/******************************************************************************/
// {{{ eulerDens

double euler4Dens(long int *h, double res0, double res1) {
	int i;
	long int iChi = 0, iVol = 0;
	int iu[16] = {0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0};
 
 	for (i = 0; i < 16; i++) {
		iChi += iu[i]*h[i];
		iVol += h[i];
	}

 	return (double)iChi/((double)iVol*res0*res1);
}

/******************************************************************************/

double euler8Dens(long int *h, double res0, double res1) {
	int i;
	long int iChi = 0, iVol = 0;
	int iu[16] = {0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0};
 
 	for(i = 0; i < 16; i++) {
		iChi += iu[i]*h[i];
		iVol += h[i];
	}

    return (double)iChi/((double)iVol*res0*res1);
}

/******************************************************************************/

double euler6Dens(long int *h, double res0, double res1, double res2) {
    unsigned int i, p;
	long int iChi = 0, iVol = 0, hi[256];
	int iu[256] = {
         0, 0,  0, 0,  0, 0, 0, 0,  0, 0,  0, 0,  0, 0, 0, 0, 
	     0, 0,  0, 0,  0, 0, 1, 0,  0, 0,  0, 0,  0, 0, 1, 0, 
	     0, 0,  0, 0, -1, 0, 0, 0, -1, 0,  0, 0, -1, 0, 0, 0, 
	     0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
	     0, 0, -1, 0,  0, 0, 0, 0, -1, 0, -1, 0,  0, 0, 0, 0,
	     0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
	    -1, 0, -1, 0, -1, 0, 0, 0, -2, 0, -1, 0, -1, 0, 0, 0,
	     0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
	     1, 0,  0, 0,  0, 0, 0, 0,  0, 0,  0, 0,  0, 0, 0, 0,
	     0, 0,  0, 0,  0, 0, 1, 0,  0, 0,  0, 0,  0, 0, 1, 0, 
	     0, 0,  0, 0, -1, 0, 0, 0, -1, 0,  0, 0, -1, 0, 0, 0,
	     0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
	     0, 0, -1, 0,  0, 0, 0, 0, -1, 0, -1, 0,  0, 0, 0, 0,
	     0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
	    -1, 0, -1, 0, -1, 0, 0, 0, -2, 0, -1, 0, -1, 0, 0, 0,
	     0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
	};
 
	for (i = 0; i < 256; i++) {
        p = i^255; 
        hi[i] = h[p];
    }

 	for (i = 0; i < 256; i++) {
        iChi += iu[i]*hi[i];
		iVol += hi[i];
	}

	if(!iVol) return 0;
 	else return (double)iChi/((double)iVol*res0*res1*res2);
}

/******************************************************************************/

double euler26Dens(long int *h, double res0, double res1, double res2) {
	int i;
	long int iChi = 0, iVol = 0;
	int iu[256] = {
         0, 0,  0, 0,  0, 0, 0, 0,  0, 0,  0, 0,  0, 0, 0, 0, 
         0, 0,  0, 0,  0, 0, 1, 0,  0, 0,  0, 0,  0, 0, 1, 0, 
         0, 0,  0, 0, -1, 0, 0, 0, -1, 0,  0, 0, -1, 0, 0, 0, 
         0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
         0, 0, -1, 0,  0, 0, 0, 0, -1, 0, -1, 0,  0, 0, 0, 0,
         0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
        -1, 0, -1, 0, -1, 0, 0, 0, -2, 0, -1, 0, -1, 0, 0, 0,
         0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
         1, 0,  0, 0,  0, 0, 0, 0,  0, 0,  0, 0,  0, 0, 0, 0,
         0, 0,  0, 0,  0, 0, 1, 0,  0, 0,  0, 0,  0, 0, 1, 0, 
         0, 0,  0, 0, -1, 0, 0, 0, -1, 0,  0, 0, -1, 0, 0, 0,
         0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
         0, 0, -1, 0,  0, 0, 0, 0, -1, 0, -1, 0,  0, 0, 0, 0,
         0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
        -1, 0, -1, 0, -1, 0, 0, 0, -2, 0, -1, 0, -1, 0, 0, 0,
         0, 0,  0, 0,  0, 0, 1, 0, -1, 0,  0, 0,  0, 0, 1, 0,
	};

 	for (i = 0; i < 256; i++) {
        iChi += iu[i]*h[i];
		iVol += h[i];
	}

	if(!iVol) return 0;
 	else return (double)iChi/((double)iVol*res0*res1*res2);
}

// }}}
/******************************************************************************/
// {{{ weights

void weights(double *Delta,double *weight) {
/* Parameters: the side lengths of the unit cell Delta[0..2], Returns: the
 * weights weights[0..6] for the computation of the surface denstiy, 
 * Initial: N.O. Aragones Salazar , 18 Sep 1999 */
    int i, j, k; 
    double delta_xy, delta_yz, delta_zx, delta; 
    double v[8][3][6];   /* vertex of the polyhedron */
    double dir[8][3][6]; /* projection of the vertex over the unit sphere */ 
    double  prod0, prod1, prod2; 
  
    delta_xy = sqrt(Delta[0]*Delta[0]+Delta[1]*Delta[1]); 
    delta_yz = sqrt(Delta[1]*Delta[1]+Delta[2]*Delta[2]); 
    delta_zx = sqrt(Delta[2]*Delta[2]+Delta[0]*Delta[0]); 
    delta = sqrt(Delta[0]*Delta[0]+Delta[1]*Delta[1]+Delta[2]*Delta[2]);

    v[0][0][0] = 1;
    v[0][1][0] = (delta_xy-Delta[0])/Delta[1];
    v[0][2][0] = (delta-delta_xy)/Delta[2];
    v[1][0][0] = 1;
    v[1][1][0] = (delta-delta_zx)/Delta[1];
    v[1][2][0] = (delta_zx-Delta[0])/Delta[2];
    v[2][0][0] = v[1][0][0]; v[2][1][0] = -v[1][1][0]; v[2][2][0] =  v[1][2][0];
    v[3][0][0] = v[0][0][0]; v[3][1][0] = -v[0][1][0]; v[3][2][0] =  v[0][2][0];
    v[4][0][0] = v[0][0][0]; v[4][1][0] = -v[0][1][0]; v[4][2][0] = -v[0][2][0];
    v[5][0][0] = v[1][0][0]; v[5][1][0] = -v[1][1][0]; v[5][2][0] = -v[1][2][0];
    v[6][0][0] = v[1][0][0]; v[6][1][0] =  v[1][1][0]; v[6][2][0] = -v[1][2][0];
    v[7][0][0] = v[0][0][0]; v[7][1][0] =  v[0][1][0]; v[7][2][0] = -v[0][2][0];

    v[0][0][1] = (delta-delta_yz)/Delta[0];
    v[0][1][1] = 1;
    v[0][2][1] = (delta_yz-Delta[1])/Delta[2];
    v[1][0][1] = (delta_xy-Delta[1])/Delta[0];
    v[1][1][1] = 1;
    v[1][2][1] = (delta-delta_xy)/Delta[2];
    v[2][0][1] = v[1][0][1];
    v[2][1][1] = v[1][1][1]; v[2][2][1] = -v[1][2][1]; v[3][0][1] =  v[0][0][1];
    v[3][1][1] = v[0][1][1]; v[3][2][1] = -v[0][2][1]; v[4][0][1] = -v[0][0][1];
    v[4][1][1] = v[0][1][1]; v[4][2][1] = -v[0][2][1]; v[5][0][1] = -v[1][0][1];
    v[5][1][1] = v[1][1][1]; v[5][2][1] = -v[1][2][1]; v[6][0][1] = -v[1][0][1];
    v[6][1][1] = v[1][1][1]; v[6][2][1] =  v[1][2][1]; v[7][0][1] = -v[0][0][1];
    v[7][1][1] = v[0][1][1]; v[7][2][1] =  v[0][2][1];

    v[0][0][2] = (delta_zx-Delta[2])/Delta[0];
    v[0][1][2] = (delta-delta_zx)/Delta[1];
    v[0][2][2] = 1;
    v[1][0][2] = (delta-delta_yz)/Delta[0];
    v[1][1][2] = (delta_yz-Delta[2])/Delta[1];
    v[1][2][2] = 1;
    v[2][0][2] = -v[1][0][2]; v[2][1][2] =  v[1][1][2]; v[2][2][2] = v[1][2][2];
    v[3][0][2] = -v[0][0][2]; v[3][1][2] =  v[0][1][2]; v[3][2][2] = v[0][2][2];
    v[4][0][2] = -v[0][0][2]; v[4][1][2] = -v[0][1][2]; v[4][2][2] = v[0][2][2];
    v[5][0][2] = -v[1][0][2]; v[5][1][2] = -v[1][1][2]; v[5][2][2] = v[1][2][2];
    v[6][0][2] =  v[1][0][2]; v[6][1][2] = -v[1][1][2]; v[6][2][2] = v[1][2][2];
    v[7][0][2] =  v[0][0][2]; v[7][1][2] = -v[0][1][2]; v[7][2][2] = v[0][2][2];

    for (k = 0; k <= 2; k++)
        for (i = 0; i <= 7; i++)
            for (j = 0; j <= 2; j++)
                dir[i][j][k] = v[i][j][k]/sqrt(pow(v[i][0][k],2)
                             + pow(v[i][1][k],2)
                             + pow(v[i][2][k],2));
    
    for (k = 0; k <= 2; k++) {
        for (i = 0; i <= 7; i++) {
            prod0 = dir[i%8][0][k]*dir[(i+2)%8][0][k] 
                  + dir[i%8][1][k]*dir[(i+2)%8][1][k] 
                  + dir[i%8][2][k]*dir[(i+2)%8][2][k]; 
            prod1 = dir[i%8][0][k]*dir[(i+1)%8][0][k] 
                  + dir[i%8][1][k]*dir[(i+1)%8][1][k] 
                  + dir[i%8][2][k]*dir[(i+1)%8][2][k]; 
            prod2 = dir[(i+1)%8][0][k]*dir[(i+2)%8][0][k] 
                  + dir[(i+1)%8][1][k]*dir[(i+2)%8][1][k] 
                  + dir[(i+1)%8][2][k]*dir[(i+2)%8][2][k]; 
            weight[k] = weight[k] 
                      + acos((prod0-prod1*prod2)/(sqrt(1.-prod1*prod1)*sqrt(1.-prod2*prod2))); 
        } 
        weight[k] = (weight[k]-6.*M_PI)/(4.*M_PI); 
    } 
    
    v[0][0][3] =  v[0][0][0]; v[0][1][3] = v[0][1][0]; v[0][2][3] =  v[0][2][0]; 
    v[1][0][3] =  v[0][0][0]; v[1][1][3] = v[0][1][0]; v[1][2][3] = -v[0][2][0]; 
    v[2][0][3] =  v[1][0][1]; v[2][1][3] = v[1][1][1]; v[2][2][3] = -v[1][2][1]; 
    v[3][0][3] =  v[1][0][1]; v[3][1][3] = v[1][1][1]; v[3][2][3] =  v[1][2][1]; 
    v[0][0][4] =  v[0][0][1]; v[0][1][4] = v[0][1][1]; v[0][2][4] =  v[0][2][1]; 
    v[1][0][4] = -v[0][0][1]; v[1][1][4] = v[0][1][1]; v[1][2][4] =  v[0][2][1]; 
    v[2][0][4] = -v[1][0][2]; v[2][1][4] = v[1][1][2]; v[2][2][4] =  v[1][2][2]; 
    v[3][0][4] =  v[1][0][2]; v[3][1][4] = v[1][1][2]; v[3][2][4] =  v[1][2][2]; 
    
    v[0][0][5] = v[0][0][2]; v[0][1][5] =  v[0][1][2]; v[0][2][5] = v[0][2][2]; 
    v[1][0][5] = v[0][0][2]; v[1][1][5] = -v[0][1][2]; v[1][2][5] = v[0][2][2]; 
    v[2][0][5] = v[1][0][0]; v[2][1][5] = -v[1][1][0]; v[2][2][5] = v[1][2][0]; 
    v[3][0][5] = v[1][0][0]; v[3][1][5] =  v[1][1][0]; v[3][2][5] = v[1][2][0]; 
    
    for (k = 3; k <= 5; k++)
        for (i = 0; i <= 3; i++)
            for (j = 0; j <= 2; j++)
                dir[i][j][k] = v[i][j][k]/sqrt(pow(v[i][0][k],2) 
                             + pow(v[i][1][k],2) 
                             + pow(v[i][2][k],2)); 
    
    for (k = 3; k <= 5; k++) {
        for (i = 0; i <= 3; i++) {
            prod0 = dir[i%4][0][k]*dir[(i+2)%4][0][k] 
                  + dir[i%4][1][k]*dir[(i+2)%4][1][k] 
                  + dir[i%4][2][k]*dir[(i+2)%4][2][k]; 
            prod1 = dir[i%4][0][k]*dir[(i+1)%4][0][k] 
                  + dir[i%4][1][k]*dir[(i+1)%4][1][k] 
                  + dir[i%4][2][k]*dir[(i+1)%4][2][k]; 
            prod2 = dir[(i+1)%4][0][k]*dir[(i+2)%4][0][k] 
                  + dir[(i+1)%4][1][k]*dir[(i+2)%4][1][k] 
                  + dir[(i+1)%4][2][k]*dir[(i+2)%4][2][k]; 
            weight[k] = weight[k] 
                      + acos((prod0-prod1*prod2)/(sqrt(1-pow(prod1,2))*sqrt(1-pow(prod2,2)))); 
        } 
        weight[k] = (weight[k]-2*M_PI)/(4*M_PI); 
    } 
    
    weight[6] = (1-2*(weight[0]+weight[1]+weight[2])-4*(weight[3]+weight[4]+weight[5]))/8; 
} 

// }}}    
/******************************************************************************/
