#!/usr/bin/env python

import sys
import argparse
import math
import random

import inspect

import numpy as np 
import distributions_new as dists
import galacticops as go
import orbitalparams

import dist_gen as dg
from population import Population
from pulsar import Pulsar
from survey import Survey
import scipy.stats as stats

from progressbar import ProgressBar

if sys.version_info[0] < 3:
    import cPickle
else:
    import pickle as cPickle

class PopulateException(Exception):
    pass


def generate(ngen,
             surveyList=None,
             pDistType='lnorm',
             radialDistType='lfl06',
             radialDistPars=7.5,
             electronModel='ne2001',
             pDistPars=[2.7, -0.34],
             siDistPars=[-1.6, 0.35],
             lumDistType='lnorm',
             lumDistPars=[-1.1, 0.9],
             zscaleType='exp',
             zscale=0.33,
             duty_percent=6.,
             scindex=-3.86,
             gpsArgs=[-1, -1],
             doubleSpec=[-1, -1],
             nostdout=False,
             pattern='gaussian',
             orbits=False,
             dgf=None,
             singlepulse=False,
             dither=False,
             accelsearch=False,
             jerksearch=False,
             sig_factor=10.0,
             bns = False,
             orbparams = {'m': [1, 5], 'm1': [1.0, 2.4], 'm2': [0.2, 1e9], 'om': [0, 360.], 'inc': [0, 90], 'ec': [0., 1.], 'pod': [1e-3, 1e3]},
             brDistType='log_unif'):
    
    #set up population object
    pop = Population()
    
    
    if 'd_g' in (pDistType,lumDistType,zscaleType,radialDistType,brDistType) and dgf is None:
        print("Provide the distribution generation file")
        sys.exit()
    elif dgf != None:
        try:
            f = open(dgf, 'rb')
        except IOError:
            print("Could not open file {0}.".format(dgf))
            sys.exit()
        dgf_pop_load = cPickle.load(f)
        f.close()
    
    #draw periods
    pop.pDistType = pDistType
    pop.pmean= pDistPars[0]
    pop.psigma = abs(pDistPars[1])
    
    periods = get_periods(pop, ngen)

    pulsars = []
    for P in periods:
        p = Pulsar()
        p.period = P
        pulsars.append(p)
    
    
    return pulsars

 






def get_periods(pop, ngen):
    if pop.pDistType == 'lnorm':
        periods = dists.drawlnorm(pop.pmean, pop.psigma, ngen)

    elif pop.pDistType == 'unif':
        periods = np.random.uniform(
            pop.pmean - pop.psigma,
            pop.pmean + pop.psigma,
            size=ngen
        )

    elif pop.pDistType == 'norm':
        periods = np.random.normal(pop.pmean, pop.psigma, size=ngen)
    elif pop.pDistType == 'cc97':
        periods = np.array([_cc97() for _ in range(ngen)])

    elif pop.pDistType == 'gamma':
        #impliment good values for shape,scale
        shape, scale = pop.pmean, pop.psigma # mean and width
        periods = np.random.standard_gamma(shape, ngen)

    elif pop.pDistType == 'd_g':
        Pbin_num = dists.draw1d(
            dgf_pop_load['pHist'],
            size=ngen
        )
        Pmin = dgf_pop_load['pBins'][0]
        Pmax = dgf_pop_load['pBins'][-1]

        periods = (
            Pmin
            + (Pmax - Pmin)
            * (Pbin_num + np.random.random(size=ngen))
            / len(dgf_pop_load['pHist'])
        )

    elif pop.pDistType == 'lorimer12':
        periods = np.array(
            [_lorimer2012_msp_periods() for _ in range(ngen)]
        )


    else:
        raise ValueError(f"Unknown pDistType: {pop.pDistType}")

    return periods


def _cc97():
    """A model for MSP period distribution."""
    p = 0.0

    # pick p from the distribution, but cut off at 1 and 30 ms
    while p < 1.0 or p > 30.0:
        p = 0.65 / (1 - random.random())

    return p


def _lorimer2012_msp_periods():
    """Picks a period at random from Dunc's
       distribution as mentioned in IAU
       (China 2012) proceedings
    """
    # min max and n in distribution
    logpmin = 0.
    logpmax = 1.5
    dist = [1., 3., 5., 16., 9., 5., 5., 3., 2.]

    # calculate which bin to take value of
    #
    bin_num = dists.draw1d(dist)

    # assume linear distn inside the bins
    logp = logpmin + (logpmax-logpmin)*(bin_num+random.random())/len(dist)

    return 10.**logp