/******************************************************************************/

int c_erode_2d(unsigned short* image, unsigned short* erosion, int dim0, int dim1          , int dist, double res0, double res1             );
int c_erode_3d(unsigned short* image, unsigned short* erosion, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2);

int c_dilate_2d(unsigned short* image, unsigned short* dilation, int dim0, int dim1          , int dist, double res0, double res1             );
int c_dilate_3d(unsigned short* image, unsigned short* dilation, int dim0, int dim1, int dim2, int dist, double res0, double res1, double res2);

int c_open_map_2d(unsigned short* erosion, unsigned short* opening, int dim0, int dim1          , double res0, double res1             );
int c_open_map_3d(unsigned short* erosion, unsigned short* opening, int dim0, int dim1, int dim2, double res0, double res1, double res2);

int c_close_map_2d(unsigned short* dilation, unsigned short* closing, int dim0, int dim1          , double res0, double res1             );
int c_close_map_3d(unsigned short* dilation, unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2);

int c_get_map_2d(unsigned short* image, unsigned short* outImage, int dim0, int dim1          , double res0, double res1             , int mode);
int c_get_map_3d(unsigned short* image, unsigned short* outImage, int dim0, int dim1, int dim2, double res0, double res1, double res2, int mode);

/******************************************************************************/
