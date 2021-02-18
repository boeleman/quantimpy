/******************************************************************************/

int c_functionals_2d(unsigned short* image, int dim0, int dim1          , double res0, double res1                             , double* area  , double* length , double* euler4   , double* euler8 );
int c_functionals_3d(unsigned short* image, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* volume, double* surface, double* curvature, double* euler6, double* euler26);

/******************************************************************************/

int c_functions_open_2d(unsigned short* closing, int dim0, int dim1          , double res0, double res1             , double* dist                , double* area   , double* length   , double* euler4, double* euler8 );
int c_functions_open_3d(unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* dist, double* volume, double* surface, double* curvature, double* euler6, double* euler26);

/******************************************************************************/

int c_functions_close_2d(unsigned short* closing, int dim0, int dim1          , double res0, double res1             , double* dist                , double* area   , double* length   , double* euler4, double* euler8 );
int c_functions_close_3d(unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* dist, double* volume, double* surface, double* curvature, double* euler6, double* euler26);

/******************************************************************************/

long int* quant_2d(unsigned short* image, int dim0, int dim1          );
long int* quant_3d(unsigned short* image, int dim0, int dim1, int dim2);

double area_dens_2d(long int *h);
double leng_dens_2d(long int *h, double res0, double res1);
double eul4_dens_2d(long int *h, double res0, double res1);
double eul8_dens_2d(long int *h, double res0, double res1);

double volu_dens_3d(long int *h);
double surf_dens_3d(long int *h, double res0, double res1, double res2);
double curv_dens_3d(long int *h, double res0, double res1, double res2);
double eul6_dens_3d(long int *h, double res0, double res1, double res2);
double eu26_dens_3d(long int *h, double res0, double res1, double res2);

/******************************************************************************/

void weights(double *Delta,double *weight);

/******************************************************************************/

