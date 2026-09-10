#!/usr/bin/python

import math
import numpy as np


def calcFlux(snr,
             beta,
             Trec,
             Tsky,
             gain,
             n_p,
             t_obs,
             bw,
             duty):

    """Calculate flux assuming radiometer equation"""

    signal = signalterm(beta,
                        Trec,
                        Tsky,
                        gain,
                        n_p,
                        t_obs,
                        bw,
                        duty)

    return snr * signal


def calcSNR(flux,
            beta,
            Trec,
            Tsky,
            gain,
            n_p,
            t_obs,
            bw,
            duty):

    """Calculate the S/N ratio assuming radiometer equation"""

    signal = signalterm(beta,
                        Trec,
                        Tsky,
                        gain,
                        n_p,
                        t_obs,
                        bw,
                        duty)

    return flux / signal

###########################
def calcSNR_vectorized(flux,
                       beta,
                       Trec,
                       Tsky,
                       gain,
                       n_p,
                       t_obs,
                       bw,
                       duty):
    """Calculate S/N ratio using the radiometer equation for arrays."""

    signal = signalterm_vectorized(
        beta,
        Trec,
        Tsky,
        gain,
        n_p,
        t_obs,
        bw,
        duty
    )

    return flux / signal
###########################


def signalterm(beta,
               Trec,
               Tsky,
               gain,
               n_p,
               t_obs,
               bw,
               duty):

    """Returns the rest of the radiometer equation (aside from
        SNR/Flux"""

    dterm = duty / (1.0 - duty)
    return beta * (Trec+Tsky) * math.sqrt(dterm) \
        / gain / math.sqrt(n_p * t_obs * bw)

def single_pulse_snr(n_p,bw,duty,T_sys,gain,S_max,beta):
    eta=0.868
    S_sys=T_sys/gain
    return(eta*math.sqrt(n_p*bw*duty)*S_max/(S_sys*beta))

def signalterm_vectorized(beta,
                          Trec,
                          Tsky,
                          gain,
                          n_p,
                          t_obs,
                          bw,
                          duty):

    dterm = duty / (1.0 - duty)

    signal = (
        beta *
        (Trec + Tsky) *
        np.sqrt(dterm)
        /
        gain
        /
        np.sqrt(n_p * t_obs * bw)
    )

    return signal

def single_pulse_snr_vectorized(n_p,
                                bw,
                                duty,
                                T_sys,
                                gain,
                                S_max,
                                beta):

    eta = 0.868

    S_sys = T_sys / gain

    return (
        eta *
        np.sqrt(n_p * bw * duty) *
        S_max
        /
        (S_sys * beta)
    )
