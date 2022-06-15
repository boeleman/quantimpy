#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from quantimpy.brisque import asgennorm
from quantimpy import brisque as bq

data = asgennorm.rvs(2,1,size=10000)

alpha, beta, loc, scale = asgennorm.fit(data)

x = np.linspace(
    asgennorm.ppf(0.001, alpha, beta, loc=loc, scale=scale), 
    asgennorm.ppf(0.999, alpha, beta, loc=loc, scale=scale), 101)

plt.plot(x, asgennorm.pdf(x, alpha, beta, loc=loc, scale=scale), 'r-')
plt.hist(data, density=True, bins=51)
plt.show()
