import sys
import argparse
import math
import random
import numpy as np

from population import Population
from pulsar import Pulsar
from survey_new import Survey
import orbital_degradation

import pandas as pd
import pickle

if sys.version_info[0] < 3:
    import cPickle
else:
    import pickle as cPickle


def loadModel(popfile='pulsarpop.pkl', popmodel=None):
    """Loads in either a model from disk (popfile, cPickle),
       or pass in a model from memory (popmodel)"""

    if popmodel is None:

        with open(popfile, "rb") as f:
            pop = pickle.load(f)

        print("Loaded", len(pop.population), "pulsars")
    #else:
       # pop = popmodel

    return pop

class Detections:
    """Just a simple object to store survey detection summary"""
    def __init__(self,
                 ndet=None,
                 ndisc=None,
                 nsmear=None,
                 nout=None,
                 nbr=None,
                 ntf=None):
        self.ndet = ndet
        self.ndisc = ndisc
        self.nsmear = nsmear
        self.nout = nout
        self.nbr = nbr
        self.nfaint = ntf

def run(pop,
        surveyList,
        nostdout=False,
        allsurveyfile=True,
        scint=False,
        compute_degfac=False,
        accelsearch=False,
        jerksearch=False,
        rratssearch=False,loadmodel=False):

    """Run the surveys and detect the pulsars."""
    if loadmodel:
        pop = loadModel(popfile='pulsarpop.pkl')
    else:
        pop=pop
    if not nostdout:
        print("Running doSurvey on population...")
        print(pop)

    surveyPops = []

    for surv in surveyList:

        s = Survey(surv)
        s.discoveries = 0

        if not nostdout:
            print("\nRunning survey {0}".format(surv))

        survpop = Population()

        # Counters
        nsmear = 0
        nout = 0
        ntf = 0
        ndet = 0
        nbr = 0
            
        #evo only gives you alive pulsars
        alive = [psr for psr in pop.population]

        t_obs = s.tobs
        # Calculate S/N for entire population
        snr = s.SNRcalc_new(
            alive,
            pop,
            accelsearch=accelsearch,
            jerksearch=jerksearch,
            rratssearch=rratssearch
        )

        snr_arr = np.asarray(snr, dtype=float).copy()

        valid = snr_arr > 0
        
        if compute_degfac:
        
            if accelsearch:
                _, gamma = compute_population_degradation(alive,tobs=t_obs,order=2)
        
            elif jerksearch:
                _, gamma = compute_population_degradation(alive,tobs=t_obs,order=3)
        
            else:
                _, gamma = compute_population_degradation(alive,tobs=t_obs,order=1)
                #print(gamma)
            
            gamma = np.asarray(gamma, dtype=float)
        
            snr_arr[valid] *= gamma[valid]**2

       

        for psr, snr_val in zip(alive, snr_arr):

            # Apply scintillation if enabled
            if scint and snr_val > 0:
                snr_val = s.scint(psr, snr_val)

            # Detected pulsar
            if snr_val > s.SNRlimit:

                ndet += 1

                psr.snr = snr_val
                survpop.population.append(psr)

                # Count discoveries only once
                if not psr.detected:
                    psr.detected = True
                    s.discoveries += 1

            # Not detected categories
            elif snr_val == -1.0:
                nsmear += 1

            elif snr_val == -2.0:
                nout += 1

            elif snr_val == -3.0:
                nbr += 1

            else:
                ntf += 1


        if not nostdout:

            print(
                "Total pulsars in model = {0}".format(
                    len(pop.population)
                )
            )

            print(
                "Number detected by survey {0} = {1}".format(
                    surv,
                    ndet
                )
            )

            print(
                "Of which are discoveries = {0}".format(
                    s.discoveries
                )
            )

            print(
                "Number too faint = {0}".format(
                    ntf
                )
            )

            print(
                "Number smeared = {0}".format(
                    nsmear
                )
            )

            print(
                "Number out = {0}".format(
                    nout
                )
            )

            if rratssearch:
                print(
                    "Number didn't burst = {0}".format(
                        nbr
                    )
                )

            print("\n")


        d = Detections(
            ndet=ndet,
            ntf=ntf,
            nsmear=nsmear,
            nout=nout,
            nbr=nbr,
            ndisc=s.discoveries
        )

        surveyPops.append(
            [surv, survpop, d]
        )


    if allsurveyfile:

        allsurvpop = Population()

        allsurvpop.population = [
            psr for psr in pop.population
            if psr.detected
        ]

        surveyPops.append(
            [None, allsurvpop, None]
        )


    return surveyPops


def compute_population_degradation(population, tobs=2100.0, order=1):
    population=population
    #get them into orbital deg
    vectorized_pulsars = {
        "m":     np.array([p.m for p in pop.population]),
        "m1":    np.array([p.m1 for p in pop.population]),
        "m2":    np.array([p.m2 for p in pop.population]),
        "ps":    np.array([p.period for p in pop.population])/ 1000.0, # p.period maps to ps
        "omega": np.array([p.om for p in pop.population]),     # p.om maps to omega
        "inc":   np.array([p.inc for p in pop.population]),
        "ecc":   np.array([p.ec for p in pop.population]),     # p.ec maps to ecc
        "pb":    np.array([p.pod for p in pop.population]),    # p.pod maps to pb
    }

    #Calculate degradation factors
    gamma_array = orbital_degradation.degradation_factor(tobs=tobs, order=order, **vectorized_pulsars)
    
    return pop, gamma_array


if __name__ == '__main__':
    """ 'Main' function; read in options, then survey the population"""
    # Parse command line arguments

    parser = argparse.ArgumentParser(
        description='Run a survey on your population model')
    parser.add_argument(
        '-f', metavar='fname', default='populate.model',
        help='file containing population model (def=populate.model')

    parser.add_argument(
        '-surveys', metavar='S', nargs='+', required=True,
        help='surveys to use to detect pulsars (required)')

    parser.add_argument(
        '--noresults', nargs='?', const=True, default=False,
        help='flag to switch off pickled .results file (def=False)')
    parser.add_argument(
        '--singlepulse', nargs='?', const=True, default=False,
        help='Rotating Radio Transients uses single pulse snr')

    parser.add_argument(
        '--asc', nargs='?', const=True, default=False,
        help='flag to create ascii population file (def=False)')

    parser.add_argument(
        '--summary', nargs='?', const=True, default=False,
        help='flag to create ascii summary file (def=False)')

    parser.add_argument(
        '--nostdout', nargs='?', const=True, default=False,
        help='flag to switch off std output (def=False)')

    parser.add_argument(
        '--allsurveys', nargs='?', const=True, default=False,
        help='write additional allsurv.results file (def=False)')

    parser.add_argument(
        '--scint', nargs='?', const=True, default=False,
        help='include model scintillation effects (def=False)')
    parser.add_argument('--orbital_deg',default=False, help='turn on the orbital degredation calculation for binary')

    parser.add_argument(
        '--accel', nargs='?', const=True, default=False,
        help='use accel search for MSPs (def=False)')

    parser.add_argument(
        '--jerk', nargs='?', const=True, default=False,
        help='use accel & jerk search for MSPs (def=False)')
    parser.add_argument(--'load_pop', default=False, help='load a population')

    args = parser.parse_args()

    # Load a model population
    population = loadModel(popfile=args.f)
    # run the population through the surveys
    surveyPopulations = run(population,
                            args.surveys,
                            nostdout=args.nostdout,
                            allsurveyfile=args.allsurveys,
                            scint=args.scint,
                            accelsearch=args.accel,
                            jerksearch=args.jerk,
                            rratssearch=args.singlepulse)

    # write the output files
    write(surveyPopulations,
          nores=args.noresults,
          asc=args.asc,
          summary=args.summary)
