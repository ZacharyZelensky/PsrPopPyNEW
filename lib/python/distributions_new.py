#!/usr/bin/python

import sys
import math
import random
import numpy as np

def drawlnorm(mean, sigma, ngen):
    """
    Draw ngen samples from a base-10 log-normal distribution
    """
    sigma = abs(sigma)
    return 10.0 ** np.random.normal(mean, sigma, size=ngen)

import numpy as np

def powerlaw(minval, maxval, power,ngen):
    """Draw a value randomly from the specified power law"""

    logmin = math.log10(minval)
    logmax = math.log10(maxval)

    c = -1.0 * logmax * power
    nmax = 10.0**(power*logmin + c)

    # slightly worried about inf loops here...
    
    log = np.random.uniform(logmin, logmax,size=ngen)
    n = 10.0**(power*log + c)

    return 10.0**log



def draw1d(dist):
    """Draw a bin number form a home-made distribution
        (dist is a list of numbers per bin)
    """
    # sum of distribution
    total = float(sum(dist))
    # cumulative distn
    cumulative = [sum(dist[:x+1])/total for x in range(len(dist))]

    rand_num = random.random()
    for i, c in enumerate(cumulative):
        if rand_num <= c:
            return i


def draw_double_sided_exp(scale, origin=0.0):
    """Exponential distribution around origin, with scale height scale."""
    if scale == 0.0:
        return origin

    rn = random.random()
    sign = random.choice([-1.0, 1.0])

    return origin + sign * scale * math.log(rn)

def uniform(low,high,ngen):
	"""Draw a random number from a uniform distribution between low and high"""
	return 10 ** np.random.uniform(np.log10(low),np.log10(high),size=ngen)
