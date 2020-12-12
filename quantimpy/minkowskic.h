/******************************************************************************/

int cFunctionals2D(unsigned short* image, int dim0, int dim1          , double res0, double res1             , double* area, double* length, double* euler4, double* euler8);
int cFunctionals3D(unsigned short* image, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* volume, double* surface, double* curvature, double* euler6, double* euler26);

/******************************************************************************/

long int* quant2D(unsigned short* image, int dim0, int dim1          );
long int* quant3D(unsigned short* image, int dim0, int dim1, int dim2);


double areaDens(long int *h);
double lengthDens(long int *h, double res0, double res1);

double volumeDens(long int *h);
double surfaceDens(long int *h, double res0, double res1, double res2);
double curvatureDens(long int *h, double res0, double res1, double res2);

double euler4Dens(long int *h, double res0, double res1);
double euler8Dens(long int *h, double res0, double res1);

double euler6Dens(long int *h, double res0, double res1, double res2);
double euler26Dens(long int *h, double res0, double res1, double res2);

/******************************************************************************/

void weights(double *Delta,double *weight);

/******************************************************************************/

