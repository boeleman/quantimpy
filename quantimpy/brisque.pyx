r"""Functions for BRISQUE no-reference image quality assesment (NR-IQA)

This module contains various functions for the computation of the BRISQUE [1]_
no-reference image quality assesment (NR-IQA) for both 2D and 3D NumPy [2]_ arrays.

"""

import cython
import numpy as np
import scipy.optimize as op
import scipy.special as sc
from scipy.ndimage import gaussian_filter
from scipy.stats import gennorm
from scipy.stats import rv_continuous
import matplotlib.pyplot as plt
cimport numpy as np

class asgennorm_gen(rv_continuous):
    r"""
    Asymmetric generalized normal continuous random variable

    This is an asymmetric generalized Gaussian distribution (AGGD) class based
    on the SciPy [3]_ `scipy.stats.rv_continuous` class. The probability density
    function for `asgennorm` is [4]_:

    .. math:: f(x, \alpha, \beta) = \frac{\beta}{(1 + a)
        \Gamma{(\frac{1}{\beta})}} \exp{(-|\frac{x}{a}|^{\beta})}

    where :math:`a = \alpha` if :math:`x < 0` and :math:`a = 1` if  :math:`x \ge
    0`, :math:`\alpha` is a measure for the standard deviation, :math:`\Gamma`
    is the gamma function (`scipy.special.gamma`), and :math:`\beta` is the
    shape parameter. For :math:`\beta = 1`, it is identical to a Laplace
    distribution and for  :math:`\beta = 2`, it is identical to a normal
    distribution.

    Some of the methods of this class include:
    
    Methods
    -------
    pdf(x, alpha, beta)
        Probability distribution function
    cdf(x, alpha, beta)
        Cumulative distribution function
    fit(data)    
        Fitting the probability distribution function 

    See Also
    --------
    ~quantimpy.brisque.coeff
    
    Examples
    --------
    This example uses the NumPy [2]_, and Matplotlib [5]_ packages.

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from quantimpy.brisque import asgennorm

        # Create dataset following the asymmetric generalized Gaussian distribution 
        data = asgennorm.rvs(2,1,size=10000)

        # Fit the dataset    
        alpha, beta, loc, scale = asgennorm.fit(data)

        x = np.linspace(
            asgennorm.ppf(0.001, alpha, beta, loc=loc, scale=scale),
            asgennorm.ppf(0.999, alpha, beta, loc=loc, scale=scale), 101)
        
        # Plot both the dataset and fit
        plt.plot(x, asgennorm.pdf(x, alpha, beta, loc=loc, scale=scale), 'r-')
        plt.hist(data, density=True, bins=51)
        plt.show()
    
    """
    def _pdf(self, x, alpha, beta):
        return np.exp(self._logpdf(x, alpha, beta))

    def _logpdf(self, x, alpha, beta):
# Beta depends on the sign of x                
        coef = np.log(beta) - np.log(alpha + 1.0) - sc.gammaln(1.0/beta)
        f = lambda x, a: coef - (np.abs(x/a))**beta
        return np.where(x < 0, f(x, alpha), f(x, 1.0))

    def _fitstart(self,data):
        def estimate_phi(beta):
            numerator = sc.gamma(2 / beta) ** 2
            denominator = sc.gamma(1 / beta) * sc.gamma(3 / beta)
            return numerator / denominator

        def estimate_r_hat(x):
            size = np.prod(x.shape)
            return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

        def estimate_R_hat(r_hat, gamma):
            numerator = (gamma ** 3 + 1) * (gamma + 1)
            denominator = (gamma ** 2 + 1) ** 2
            return r_hat * numerator / denominator

        def mean_squares_sum(x, filter = lambda z: z == z):
            filtered_values = x[filter(x)]
            squares_sum = np.sum(filtered_values ** 2)
            return squares_sum / ((filtered_values.shape))

        def estimate_gamma(x):
            left_squares = mean_squares_sum(x, lambda z: z < 0)
            right_squares = mean_squares_sum(x, lambda z: z >= 0)

            return np.sqrt(left_squares) / np.sqrt(right_squares)

        def estimate_beta(x):
            r_hat = estimate_r_hat(x)
            gamma = estimate_gamma(x)
            R_hat = estimate_R_hat(r_hat, gamma)

            solution = op.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x

            return solution[0]

        def estimate_sigma(x, beta, filter = lambda z: z < 0):
            return np.sqrt(mean_squares_sum(x, filter))
        
        def estimate_mean(beta, sigma_l, sigma_r):
            return (sigma_r - sigma_l) * constant * (sc.gamma(2 / beta) / sc.gamma(1 / beta))
        
        beta = estimate_beta(data)
        sigma_l = estimate_sigma(data, beta, lambda z: z < 0)
        sigma_r = estimate_sigma(data, beta, lambda z: z >= 0)
        
        constant = np.sqrt(sc.gamma(1 / beta) / sc.gamma(3 / beta))
        mean = estimate_mean(beta, sigma_l, sigma_r)
        alpha = sigma_l/sigma_r
        sigma = sigma_r

        return alpha, beta, mean, sigma

    def _cdf(self, x, alpha, beta):
        c = 0.5 * np.sign(x)
        f = lambda x, a: (0.5 + c) - c * sc.gammaincc(1.0/beta, abs(x/a)**beta)
        return np.where(x < 0, f(x, alpha), f(x, 1.0))

    def _sf(self, x, alpha, beta):
        return self._cdf(-x, alpha, beta)

asgennorm = asgennorm_gen(name='asgennorm')

@cython.binding(True)
cpdef mscn(image, patch=7, trunc=3.0, debug=None):
    r"""
    Mean subtracted contrast normalized (MSCN) coefficients

    Compute the mean subtracted contrast normalized (MSCN) coefficients [1]_ for
    the 2D or 3D NumPy array `image`. The MSCN coefficients, :math:`\hat{I}`,
    are defined as: 

    .. math:: \hat{I} = \frac{I - \mu}{\sigma}

    where :math:`I` is the original image, :math:`\mu` is the local mean field,
    and :math:`\sigma` is the local variance field. The MSCN coefficients serve
    as input for the function :func:`~quantimpy.brisque.coeff`. When the `debug`
    parameter is set, images of the local mean field and local variance field
    are also returned.

    Parameters
    ----------
    image : ndarray, {int, uint, float}
        2D or 3D grayscale input image.
    patch : int, defaults to 7
        Size of the patch used to compute the local mean field and local
        variance field. Defaults to 7.
    trunc : float, defaults to 3.0
        Value at which to truncate the normal distribution used to calculate the
        local mean field and local variance field. Defaults to 3.0. 
    debug : str, defaults to "None"
        Output directory for debugging images. When this parameter is set,
        images of the local mean field and local variance field are are written
        to disk. Set to "./" to write to the working directory. The default is
        "None".
    
    Returns
    -------
    out : ndarray, float
        The 2D or 3D MSCN coefficients. The return data type is float and the
        image is normalized betweeen 0 and 1 or -1 and 1.

    See Also
    --------
    ~quantimpy.brisque.coeff

    Examples
    --------
    This example uses the NumPy package [2]_. The NumPy data file
    "`rock_2d.npy`_" is available on Github [6]_ [7]_.

    .. code-block:: python

        import numpy as np
        from quantimpy import brisque as bq

        # Load data
        image = np.load("rock_2d.npy")

        # Compute MSCN coefficients
        mscn = bq.mscn(image)

        print(bq.coeff(mscn))

    References
    ----------
    
    .. [1] Anish Mittal, Anush Moorthy, and Alan Bovik, "No-reference image
        quality assessment in the spatial domain", IEEE Transactions on image
        processing, vol. 21, no. 12, pp 4695-4708, 2012,
        doi:`10.1109/TIP.2012.2214050`_

    .. _10.1109/TIP.2012.2214050: https://doi.org/10.1109/TIP.2012.2214050
    
    .. [2] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt et al.,
        "Array programming with NumPy", Nature, vol. 585, pp 357-362, 2020,
        doi:`10.1038/s41586-020-2649-2`_

    .. _10.1038/s41586-020-2649-2: https://doi.org/10.1038/s41586-020-2649-2

    .. [3] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, et al., "SciPy
        1.0: Fundamental Algorithms for Scientific Computing in Python", Nature
        Methods, vol. 17, pp 261-272, 2020, doi:`10.1038/s41592-019-0686-2`_
    
    .. _10.1038/s41592-019-0686-2: https://doi.org/10.1038/s41592-019-0686-2

    .. [4] Nour-Eddine Lasmar, Youssef Stitou, and Yannick Berthoumieu,
        "Multiscale skewed heavy tailed model for texture analysis", 16th IEEE
        International Conference on Image Processing (ICIP), pp 2281-2284, 2009,
        doi:`10.1109/ICIP.2009.5414404`_

    .. _10.1109/ICIP.2009.5414404: https://doi.org/10.1109/ICIP.2009.5414404

    .. [5] John D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in
        Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
        doi:`10.1109/MCSE.2007.55`_

    .. _10.1109/MCSE.2007.55: https://doi.org/10.1109/MCSE.2007.55

    .. [6] Catherine Spurin, Tom Bultreys, Maja Rücker, et al., "The
        development of intermittent multiphase fluid flow pathways through a
        porous rock", Advances in Water Resources, vol. 150, 2021,
        doi:`10.1016/j.advwatres.2021.103868`_

    .. _10.1016/j.advwatres.2021.103868: https://doi.org/10.1016/j.advwatres.2021.103868

    .. [7] Catherine Spurin, Tom Bultreys, Maja Rücker, et al., "Real-Time
        Imaging Reveals Distinct Pore-Scale Dynamics During Transient and
        Equilibrium Subsurface Multiphase Flow", Water Resources Research, vol.
        56, no. 12, 2020, doi:`10.1029/2020WR028287`_

    .. _10.1029/2020WR028287: https://doi.org/10.1029/2020WR028287

    .. _rock_2d.npy: https://github.com/boeleman/quantimpy/raw/thresholding/test/rock_2d.npy

    .. _rock_3d.npy: https://github.com/boeleman/quantimpy/raw/thresholding/test/rock_3d.npy

    """
# Check that patch size is odd    
    if (patch % 2) == 0:
        raise ValueError("patch should be an odd number")

# Convert to float and normalize
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float64)/(np.iinfo(image.dtype).max)
    
    if (image.ndim == 2):
        return _mscn_2d(image, patch, trunc, debug)
    elif (image.ndim == 3):
        return _mscn_3d(image, patch, trunc, debug)
    else:
        raise ValueError('Cannot handle more than three dimensions')

@cython.boundscheck(False)
@cython.wraparound(False)
def _mscn_2d(np.ndarray[cython.floating, ndim=2] image, int patch, double trunc, str debug):

    cdef np.ndarray[cython.floating, ndim=2] mu
    cdef np.ndarray[cython.floating, ndim=2] sigma

# Apply Gaussian filter in patch of 7x7(x7) with sigma=7/6=1.166666667
    mu = gaussian_filter(image, patch/(2.0*trunc), truncate=trunc)
    sigma = gaussian_filter(np.asarray(image)*np.asarray(image), patch/(2.0*trunc), truncate=trunc)
    sigma = np.sqrt(sigma - mu*mu)
    sigma = sigma + 1.0e-16 # Avoid devision by zero

    if (debug is not None):
        name = debug + "mu.png"

        plt.gray()
        plt.imshow(mu)
        plt.axis("off")
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.clf()

        name = debug + "sigma.png"

        plt.gray()
        plt.imshow(sigma)
        plt.axis("off")
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.clf()

    return (image - mu)/sigma

@cython.boundscheck(False)
@cython.wraparound(False)
def _mscn_3d(np.ndarray[cython.floating, ndim=3] image, int patch, double trunc, str debug):

    cdef np.ndarray[cython.floating, ndim=3] mu
    cdef np.ndarray[cython.floating, ndim=3] sigma

# Apply Gaussian filter in patch of 7x7(x7) with sigma=7/6=1.166666667
    mu = gaussian_filter(image, patch/(2.0*trunc), truncate=trunc)
    sigma = gaussian_filter(np.asarray(image)*np.asarray(image), patch/(2.0*trunc), truncate=trunc)
    sigma = np.sqrt(sigma - mu*mu)
    sigma = sigma + 1.0e-16 # Avoid devision by zero

    if (debug is not None):
        name = debug + "mu.png"

        plt.gray()
        plt.imshow(mu[int(0.5*mu.shape[0]),:,:])
        plt.axis("off")
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.clf()

        name = debug + "sigma.png"

        plt.gray()
        plt.imshow(sigma[int(0.5*sigma.shape[0]),:,:])
        plt.axis("off")
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.clf()

    return (image - mu)/sigma

@cython.binding(True)
cpdef coeff(mscn, sample_size=500000, debug=None):
    r"""
    Coefficients of pairwise products of neighboring MSCN coefficients

    Compute the fitting coefficients of the MSCN coefficients and the pairwise
    products of neighboring MSCN coefficients. The MSCN coefficients are fitted
    with the generalized Gaussian distribution and the pairwise products with
    the asymetric generalized Gausian distribution. The input 2D or 3D MSCN
    coefficients can be computed using function :func:`~quantimpy.brisque.mscn`.
    When the `debug` parameter is set, images of the distributions and their fit
    are also returned.

    Parameters
    ----------
    mscn : ndarray, float
        2D or 3D MSCN coefficients.
    sample_size : int, defaults to 500000
        To reduce computational resources, the distributions can be fitted to a
        randomly selected subset of the  MSCN coefficients. `sample_size` gives
        the size of this subset. When -1 is passed, the whole dataset is used.
        Defaults to 500000. 
    debug : str, defaults to "None"
        Output directory for debugging images. When this parameter is set,
        images of the distributions and their fit are written to disk. Set to
        "./" to write to the working directory. The default is "None".
    
    Returns
    -------
    out : ndarray, float
        2D or 3D fitting parameters.

    See Also
    --------
    ~quantimpy.brisque.mscn

    Examples
    --------
    This example uses the NumPy [2]_, and Matplotlib [5]_ packages. The NumPy
    data file "`rock_2d.npy`_" is available on Github [6]_ [7]_.

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from quantimpy import brisque as bq

        # Load data
        image = np.load("rock_2d.npy")

        # Compute MSCN coefficients
        mscn = bq.mscn(image)

        # Show coefficients
        plt.gray()
        plt.imshow(mscn)
        plt.show()

    """
    if (mscn.ndim == 2):
        return _coeff_2d(mscn, sample_size, debug)
    elif (mscn.ndim == 3):
        return _coeff_3d(mscn, sample_size, debug)
    else:
        raise ValueError('Cannot handle more than three dimensions')

@cython.boundscheck(False)
def _coeff_2d(np.ndarray[cython.floating, ndim=2] mscn, int sample_size, str debug):

    cdef np.ndarray[cython.floating, ndim=2] pair

    coefficients = np.zeros(19)

    if (sample_size == -1):
        data = mscn.flatten()
    else:
# reduce dataset size for faster processing
        data = np.random.choice(mscn.flatten(), size=sample_size)

    coefficients[0:3] = np.asarray(gennorm.fit(data))

    if (debug is not None):
        name = debug + "mscn.png"

        beta = coefficients[0]
        loc = coefficients[1]
        scale = coefficients[2]

        x = np.linspace(gennorm.ppf(0.001, beta, loc=loc, scale=scale), 
            gennorm.ppf(0.999, beta, loc=loc, scale=scale), 101)
        plt.plot(x, gennorm.pdf(x, beta, loc=loc, scale=scale), 'r-')
        plt.hist(data, 50, density=True, facecolor='g')
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.clf()

# Edges
    pair = mscn[:,:-1]*mscn[:,1:]
    data0 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[3:7] = np.asarray(asgennorm.fit(data0))

    pair = mscn[:-1,:]*mscn[1:,:]
    data1 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[7:11] = np.asarray(asgennorm.fit(data1))
    
    if (debug is not None):
        name = debug + "edge_0.png"
        _plot(name, data0, coefficients[3:7])
    
        name = debug + "edge_1.png"
        _plot(name, data1, coefficients[7:11])

# Diagonals
    pair = mscn[:-1,:-1]*mscn[1:,1:]
    data0 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[11:15] = np.asarray(asgennorm.fit(data0))

    pair = mscn[1:,:-1]*mscn[:-1,1:]
    data1 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[15:19] = np.asarray(asgennorm.fit(data1))

    if (debug is not None):
        name = debug + "diagonal_0.png"
        _plot(name, data0, coefficients[11:15])

        name = debug + "diagonal_1.png"
        _plot(name, data1, coefficients[15:19])

    return coefficients

@cython.boundscheck(False)
def _coeff_3d(np.ndarray[cython.floating, ndim=3] mscn, int sample_size, str debug):

    cdef np.ndarray[cython.floating, ndim=3] pair

    coefficients = np.zeros(55)
    
    if (sample_size == -1):
        data = mscn.flatten()
    else:
# reduce dataset size for faster processing
        data = np.random.choice(mscn.flatten(), size=sample_size)

    coefficients[0:3] = np.asarray(gennorm.fit(data))

    if (debug is not None):
        name = debug + "mscn.png"

        beta = coefficients[0]
        loc = coefficients[1]
        scale = coefficients[2]

        x = np.linspace(gennorm.ppf(0.001, beta, loc=loc, scale=scale), 
            gennorm.ppf(0.999, beta, loc=loc, scale=scale), 101)
        plt.plot(x, gennorm.pdf(x, beta, loc=loc, scale=scale), 'r-')
        plt.hist(data, 50, density=True, facecolor='g')
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.clf()

# Edges
    pair = mscn[:-1,:,:]*mscn[1:,:,:]
    data0 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[3:7] = np.asarray(asgennorm.fit(data0))

    pair = mscn[:,:-1,:]*mscn[:,1:,:]
    data1 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[7:11] = np.asarray(asgennorm.fit(data1))

    pair = mscn[:,:,:-1]*mscn[:,:,1:]
    data2 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[11:15] = np.asarray(asgennorm.fit(data2))
    
    if (debug is not None):
        name = debug + "edge_0.png"
        _plot(name, data0, coefficients[3:7])
    
        name = debug + "edge_1.png"
        _plot(name, data1, coefficients[7:11])
    
        name = debug + "edge_2.png"
        _plot(name, data2, coefficients[11:15])

# Face diagonals
    pair = mscn[:-1,:-1,:]*mscn[1:,1:,:]
    data0 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[15:19] = np.asarray(asgennorm.fit(data0))

    pair = mscn[:-1,1:,:]*mscn[1:,:-1,:]
    data1 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[19:23] = np.asarray(asgennorm.fit(data1))

    pair = mscn[:-1,:,:-1]*mscn[1:,:,1:]
    data2 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[23:27] = np.asarray(asgennorm.fit(data2))

    pair = mscn[:-1,:,1:]*mscn[1:,:,:-1]
    data3 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[27:31] = np.asarray(asgennorm.fit(data3))

    pair = mscn[:,:-1,:-1]*mscn[:,1:,1:]
    data4 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[31:35] = np.asarray(asgennorm.fit(data4))

    pair = mscn[:,:-1,1:]*mscn[:,1:,:-1]
    data5 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[35:39] = np.asarray(asgennorm.fit(data5))

    if (debug is not None):
        name = debug + "face_diagonal_0.png"
        _plot(name, data0, coefficients[15:19])

        name = debug + "face_diagonal_1.png"
        _plot(name, data1, coefficients[19:23])

        name = debug + "face_diagonal_2.png"
        _plot(name, data2, coefficients[23:27])

        name = debug + "face_diagonal_3.png"
        _plot(name, data3, coefficients[27:31])

        name = debug + "face_diagonal_4.png"
        _plot(name, data4, coefficients[31:35])

        name = debug + "face_diagonal_5.png"
        _plot(name, data5, coefficients[35:39])

# Body diagonals
    pair = mscn[:-1,:-1,:-1]*mscn[1:,1:,1:]
    data0 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[39:43] = np.asarray(asgennorm.fit(data0))

    pair = mscn[:-1,:-1,1:]*mscn[1:,1:,:-1]
    data1 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[43:47] = np.asarray(asgennorm.fit(data1))

    pair = mscn[:-1,1:,1:]*mscn[1:,:-1,:-1]
    data2 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[47:51] = np.asarray(asgennorm.fit(data2))

    pair = mscn[:-1,1:,:-1]*mscn[1:,:-1,1:]
    data3 = np.random.choice(pair.flatten(), size=sample_size)
    coefficients[51:55] = np.asarray(asgennorm.fit(data3))

    if (debug is not None):
        name = debug + "body_diagonal_0.png"
        _plot(name, data0, coefficients[39:43])

        name = debug + "body_diagonal_1.png"
        _plot(name, data1, coefficients[43:47])

        name = debug + "body_diagonal_2.png"
        _plot(name, data2, coefficients[47:51])

        name = debug + "body_diagonal_3.png"
        _plot(name, data3, coefficients[51:55])

    return coefficients

# Help function for plotting fitting coefficients
cpdef _plot(name, data, coefficients):

    alpha = coefficients[0]
    beta = coefficients[1]
    loc = coefficients[2]
    scale = coefficients[3]

    x = np.linspace(asgennorm.ppf(0.001, alpha, beta, loc=loc, scale=scale), 
        asgennorm.ppf(0.999, alpha, beta, loc=loc, scale=scale), 101)
    plt.plot(x, asgennorm.pdf(x, alpha, beta, loc=loc, scale=scale), 'r-')
    plt.hist(data, 50, density=True, facecolor='g')
    plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.clf()

#@cython.binding(True)
#cpdef to_float32(image):
#    r"""
#
#
#    """
## Convert float to np.float32
#    if np.issubdtype(image.dtype, np.floating):
#        return image.astype(np.float32)
## Convert non-float to float32 and normalize
#    elif not np.issubdtype(image.dtype, np.floating):
#        return image.astype(np.float32)/(np.iinfo(image.dtype).max)

#@cython.binding(True)
#cpdef brisque(image, patch=7, trunc=3.0, debug=None):
#    r"""
#    Test
#    """
#
#    image_hat = mscn(image, patch=patch, trunc=trunc)
#    
#    coeff(image_hat, debug=debug)

# What is structure going to look like?

# End up with one function that gives "score" for both 2D AND 3D images

# Steps
# * Compute MSCN for both full resolution and half resolution
# * Compute pairwise products
# * Fit Asymetric Generalized Gaussian Distribution
# * Compute score from coefficients

# Compute pairwise products and compute fit in serial to reduce memory usage

# TODO 
# * Exactly define score
