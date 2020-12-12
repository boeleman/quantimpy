/******************************************************************************/

int cErode2D(unsigned short* image, unsigned short* erosion, int dim0, int dim1          , int dist, double res0, double res1             );
int cErode3D(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2);

int cDilate2D(unsigned short* image, unsigned short* dilation, int dim0, int dim1          , int dist, double res0, double res1             );
int cDilate3D(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2);

int cOpenMap2D(unsigned short* erosion, unsigned short* opening, int dim0, int dim1          , double res0, double res1             );
int cOpenMap3D(unsigned short* erosion, unsigned short* opening, int dim0, int dim1, int dim2, double res0, double res1, double res2);

int cCloseMap2D(unsigned short* dilation, unsigned short* closing, int dim0, int dim1          , double res0, double res1             );
int cCloseMap3D(unsigned short* dilation, unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2);

int cGetMap2D(unsigned short* image, unsigned short* outImage, int dim0, int dim1          , double res0, double res1             , int mode);
int cGetMap3D(unsigned short* image, unsigned short* outImage, int dim0, int dim1, int dim2, double res0, double res1, double res2, int mode);

/******************************************************************************/
