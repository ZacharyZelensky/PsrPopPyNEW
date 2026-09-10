#!/usr/bin/python

import os
import subprocess
import math
import random
import numpy as np
import ctypes as C


# get the FORTRAN libraries
__dir__ = os.path.dirname(os.path.abspath(__file__))
__libdir__ = os.path.dirname(__dir__)
fortranpath = os.path.join(__libdir__, 'fortran').encode()
Cpath = os.path.join(__libdir__, 'C').encode()

ne2001lib = C.CDLL(os.path.join(fortranpath, 'libne2001.so'.encode()))
ne2001lib.dm_.restype = C.c_float

slalib = C.CDLL(os.path.join(fortranpath, 'libsla.so'.encode()))
vxyzlib = C.CDLL(os.path.join(fortranpath, 'libvxyz.so'.encode()))

yklib = C.CDLL(os.path.join(fortranpath, 'libykarea.so'.encode()))
yklib.ykr_.restype = C.c_float
yklib.llfr_.restype = C.c_float

# BEGIN FUNCTION DEFINITIONS


def vxyz(galCoords, vx, vy, vz, ages):
    """
    Evolve a population of pulsars through the Galactic potential.

    Parameters
    ----------
    galCoords : ndarray, shape (N, 3)
        Initial x, y, z positions in kpc.

    vx, vy, vz : ndarray, shape (N,)
        Initial velocities.

    ages : ndarray, shape (N,)
        Pulsar ages in years.

    Returns
    -------
    galCoords_new : ndarray, shape (N, 3)
        Final positions.

    vx_new, vy_new, vz_new : ndarray
        Final velocities.
    """

    N = len(ages)

    # Output arrays
    galCoords_new = np.empty((N, 3), dtype=np.float32)
    vx_new = np.empty(N, dtype=np.float32)
    vy_new = np.empty(N, dtype=np.float32)
    vz_new = np.empty(N, dtype=np.float32)

    for i in range(N):

        x = C.c_float(galCoords[i, 0])
        y = C.c_float(galCoords[i, 1])
        z = C.c_float(galCoords[i, 2])

        vx_i = C.c_float(vx[i])
        vy_i = C.c_float(vy[i])
        vz_i = C.c_float(vz[i])

        age_Myr = C.c_float(ages[i] / 1.0E6)

        bound = C.c_long(0)

        dt = C.c_float(0.005)

        vxyzlib.vxyz_(
            C.byref(dt),

            C.byref(x),
            C.byref(y),
            C.byref(z),

            C.byref(age_Myr),

            C.byref(vx_i),
            C.byref(vy_i),
            C.byref(vz_i),

            C.byref(x),
            C.byref(y),
            C.byref(z),

            C.byref(vx_i),
            C.byref(vy_i),
            C.byref(vz_i),

            C.byref(bound)
        )

        galCoords_new[i, 0] = x.value
        galCoords_new[i, 1] = y.value
        galCoords_new[i, 2] = z.value

        vx_new[i] = vx_i.value
        vy_new[i] = vy_i.value
        vz_new[i] = vz_i.value

    return galCoords_new, vx_new, vy_new, vz_new


def calc_dtrue(cart_pos):
    """Calculate true distance to pulsar from the sun. ip cart_pos should be tuple with (x,y,z)"""
    x, y, z = cart_pos
    rsun = 8.5  # kpc
    return math.sqrt(x*x + (y-rsun)*(y-rsun) + z*z)


def calcXY(r0):
    """Calculate the X, Y, Z alactic coords for the pulsar."""
    # calculate a random theta in a unifrom distribution
    theta = 2.0 * math.pi * random.random()

    # calc x and y
    x = r0 * math.cos(theta)
    y = r0 * math.sin(theta)

    return x, y


def ne2001_dist_to_dm(dist, gl, gb):
    """Use NE2001 distance model."""
    # expects -180 < l < 180
    dist = C.c_float(dist)
    gl = C.c_float(gl)
    gb = C.c_float(gb)
    inpath = C.create_string_buffer(fortranpath)
    linpath = C.c_int(len(fortranpath))
    return ne2001lib.dm_(C.byref(dist),
                         C.byref(gl),
                         C.byref(gb),
                         C.byref(C.c_int(4)),
                         C.byref(C.c_float(0.0)),
                         C.byref(inpath),
                         C.byref(linpath))


def lmt85_dist_to_dm(dist, gl, gb):
    """ Use Lyne, Manchester & Taylor distance model to compute DM."""
    dist = C.c_float(dist)
    gl = C.c_float(gl)
    gb = C.c_float(gb)
    # passing path to fortran dir and the length of
    # this path --- removes need to edit getpath.f
    # during installation
    inpath = C.create_string_buffer(fortranpath)
    linpath = C.c_int(len(fortranpath))

    return ne2001lib.dm_(C.byref(dist),
                         C.byref(gl),
                         C.byref(gb),
                         C.byref(C.c_int(0)),
                         C.byref(C.c_float(0.0)),
                         C.byref(inpath),
                         C.byref(linpath))

def ymw16_dist_to_dm(dist, gl, gb):
    """ Use Yao Manchester Wang 2016 electron density model to get DM."""
    """ Return distance in kpc"""
    cmd=Cpath+'/ymw16 -d '+Cpath+'/ Gal %f %f %f %i'  % (gl,gb,dist*1000,2)
    proc = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    output=[line.strip() for line in proc.stdout]
    output=output[0].split(':')
    dm=float(output[2].split('log')[0])
    return(dm)

    
def ne2001_get_smtau(dist, gl, gb):
    """Use NE2001 model to get the DISS scattering timescale"""
    dist = C.c_float(dist)

    # gl gb need to be in radians
    gl = C.c_float(math.radians(gl))
    gb = C.c_float(math.radians(gb))

    # call dmdsm and get the value out of smtau
    ndir = C.c_int(-1)
    sm = C.c_float(0.)
    smtau = C.c_float(0.)
    inpath = C.create_string_buffer(fortranpath)
    linpath = C.c_int(len(fortranpath))
    ne2001lib.dmdsm_(C.byref(gl),
                     C.byref(gb),
                     C.byref(ndir),
                     C.byref(C.c_float(0.0)),
                     C.byref(dist),
                     C.byref(C.create_string_buffer(' ')),
                     C.byref(sm),
                     C.byref(smtau),
                     C.byref(C.c_float(0.0)),
                     C.byref(C.c_float(0.0)),
                     C.byref(inpath),
                     C.byref(linpath)
                     )
    return sm.value, smtau.value


def ne2001_scint_time_bw(dist, gl, gb, freq):
    sm, smtau = ne2001_get_smtau(dist, gl, gb)
    if smtau <= 0.:
        scint_time = None
    else:
        # reference: eqn (46) of Cordes & Lazio 1991, ApJ, 376, 123
        # uses coefficient 3.3 instead of 2.3. They do this in the code
        # and mention it explicitly, so I trust it!
        scint_time = 3.3 * (freq/1000.)**1.2 * smtau**(-0.6)
    if sm <= 0.:
        scint_bw = None
    else:
        # and eq 48
        scint_bw = 223. * (freq/1000.)**4.4 * sm**(-1.2) / dist

    return scint_time, scint_bw


def lb_to_radec(l, b):
    """Convert l, b to RA, Dec using SLA fortran (should be faster)."""
    ra = C.c_float(0.)
    dec = C.c_float(0.)
    l = C.c_float(l)
    b = C.c_float(b)
    # call with final arg 1 to do conversion in right direction!
    slalib.galtfeq_(C.byref(l),
                    C.byref(b),
                    C.byref(ra),
                    C.byref(dec),
                    C.byref(C.c_int(1)))
    return ra.value, dec.value

def lb_to_radec_vectorized(l, b):

    l = np.deg2rad(l)
    b = np.deg2rad(b)

    # Galactic -> Equatorial rotation matrix
    R = np.array([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [ 0.4941094279, -0.4448296300,  0.7469822445],
        [-0.8676661490, -0.1980763734,  0.4559837762]
    ])


    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)


    xyz = np.vstack((x,y,z))

    xyz_eq = R @ xyz


    ra = np.arctan2(
        xyz_eq[1],
        xyz_eq[0]
    )

    dec = np.arcsin(
        xyz_eq[2]
    )


    ra = np.rad2deg(ra) % 360
    dec = np.rad2deg(dec)

    return ra, dec


def radec_to_lb(ra, dec):
    """Convert RA, Dec to l, b using SLA fortran.
    Be sure to return l in range -180 -> +180"""
    l = C.c_float(0.)
    b = C.c_float(0.)
    ra = C.c_float(ra)
    dec = C.c_float(dec)
    # call with arg = -1 to convert in reverse!
    slalib.galtfeq_(C.byref(l),
                    C.byref(b),
                    C.byref(ra),
                    C.byref(dec),
                    C.byref(C.c_int(-1)))
    if l.value > 180.:
        l.value -= 360.
    return l.value, b.value


import numpy as np

def xyz_to_lb(cart_pos):
    
    cart_pos = np.asarray(cart_pos)

    x = cart_pos[:, 0]
    y = cart_pos[:, 1]
    z = cart_pos[:, 2]

    rsun = 8.5  # kpc

    # distance to pulsar
    d = np.sqrt(x*x + (rsun - y)**2 + z*z)

    # galactic latitude
    b = np.arcsin(z / d)

    # projected distance
    dcb = d * np.cos(b)

    # avoid division warnings
    ratio = np.zeros_like(x)
    nonzero = dcb != 0
    ratio[nonzero] = x[nonzero] / dcb[nonzero]

    # clip to valid trig range
    ratio = np.clip(ratio, -1.0, 1.0)

    l = np.zeros_like(x)

    mask1 = (y <= rsun)
    mask2 = ~mask1

    # Case y <= rsun
    l[mask1] = np.arcsin(ratio[mask1])

    # Case y > rsun
    l[mask2] = np.arccos(ratio[mask2])
    l[mask2] += np.pi/2.0
    l[mask2 & (x < 0.0)] -= 2.0*np.pi

    # convert to degrees
    l = np.degrees(l)
    b = np.degrees(b)

    # enforce -180 < l < 180
    l = np.where(l > 180.0, l - 360.0, l)

    return l, b





def lb_to_xyz(gl, gb, dist):
    """Convert galactic coordinates (l, b, dist) to Galactic XYZ."""
    rsun = 8.5  # kpc

    l = np.radians(gl)
    b = np.radians(gb)

    x = dist * np.cos(b) * np.sin(l)
    y = rsun - dist * np.cos(b) * np.cos(l)
    z = dist * np.sin(b)

    return x, y, z


def scatter_bhat(dm,
                 scatterindex=-3.86,
                 freq_mhz=1400.0):
    """Calculate bhat et al 2004 scattering timescale for freq in MHz."""
    logtau = -6.46 + 0.154 * math.log10(dm)
    logtau += 1.07 * math.log10(dm)*math.log10(dm)
    logtau += scatterindex * math.log10(freq_mhz/1000.0)

    # return tau with power scattered with a gaussian, width 0.8
    return math.pow(10.0, random.gauss(logtau, 0.8))

#######################################################################################3
def scatter_bhat_vectorized(dm, scatterindex=-3.86, freq_mhz=1400.0):
    """Calculate Bhat et al. (2004) scattering timescale for arrays of DMs."""

    dm = np.asarray(dm)

    logdm = np.log10(dm)

    logtau = (
        -6.46
        + 0.154 * logdm
        + 1.07 * logdm**2
        + scatterindex * np.log10(freq_mhz/1000.0)
    )

    # Gaussian scatter with width 0.8 dex
    logtau += np.random.normal(
        0,
        0.8,
        size=len(dm)
    )

    return 10.0**logtau
###############################################################################################

def scale_bhat(timescale,
               frequency,
               scaling_power=3.86):
    """Scale the scattering timescale from 1.4 GHz to frequency"""

    return timescale * (frequency/1400.0)**scaling_power


def _glgboffset(gl1, gb1, gl2, gb2):
    """
    Calculate the angular distance (deg) between two
    points in galactic coordinates
    """
    # Angular offset in polar coordinates
    # taken brazenly from
    # http://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php

    # requires gb conversion from +90 -> -90 to 0 -> 180
    gb1 = 90.0 - gb1
    gb2 = 90.0 - gb2

    term1 = math.cos(math.radians(gb1)) * math.cos(math.radians(gb2))
    term2 = math.sin(math.radians(gb1)) * math.sin(math.radians(gb2))
    term3 = math.cos(math.radians(gl1) - math.radians(gl2))
    cosalpha = term1 + term2*term3

    return math.degrees(math.acos(cosalpha))


def seed():
    return C.c_int(random.randint(1, 9999))


def slabdist():
    x = -15.0 + random.random()*30.0
    y = -15.0 + random.random()*30.0
    z = -5.0 + random.random() * 10.0

    return (x, y, z)


def diskdist():
    x = -15.0 + random.random()*30.0
    y = -15.0 + random.random()*30.0
    return (x, y, 0.0)



def lfl06(N):
    """Vectorized wrapper around scalar llfr_"""
    return np.array([yklib.llfr_(C.byref(seed())) for _ in range(N)])


def ykr(N=None):
    """Y&K Model."""

    if N is None:
        return float(yklib.ykr_(C.byref(seed())))

    r0 = np.empty(N, dtype=np.float64)

    for i in range(N):
        r0[i] = yklib.ykr_(C.byref(seed()))

    return r0


def spiralize(r):
    """
    Make spiral arms, as seen in Faucher-Giguere & Kaspi 2006.

    Parameters
    ----------
    r : ndarray
        Galactocentric radial distances.

    Returns
    -------
    x, y : ndarray
        Cartesian galactic coordinates.
    """

    r = np.asarray(r)
    N = len(r)

    # Spiral arm parameters
    k_list = np.array([4.25, 4.25, 4.89, 4.89])
    r0_list = np.array([3.48, 3.48, 4.9, 4.9])
    theta0_list = np.array([1.57, 4.71, 4.09, 0.95])

    # Select one of four spiral arms for each pulsar
    arm = np.random.randint(0, 4, size=N)

    k = k_list[arm]
    r0 = r0_list[arm]
    theta0 = theta0_list[arm]

    # Spiral-arm angle
    theta = k * np.log(r / r0) + theta0

    # Angular blurring
    angle = (
        2.0
        * np.pi
        * np.random.random(N)
        * np.exp(-0.35 * r)
    )

    # Randomly choose positive/negative blur
    signs = np.where(
        np.random.random(N) < 0.5,
        -1.0,
        1.0
    )

    angle *= signs

    theta += angle

    # Radial blurring
    dr = np.abs(
        np.random.normal(
            0.0,
            0.5 * r
        )
    )

    angle = np.random.random(N) * 2.0 * np.pi

    dx = dr * np.cos(angle)
    dy = dr * np.sin(angle)

    # Convert to Cartesian coordinates
    x = r * np.cos(theta) + dx
    y = r * np.sin(theta) + dy

    return x, y


def _double_sided_exp(scale, origin=0.0, size=None):
    if scale == 0.0:
        if size is None:
            return origin
        return np.full(size, origin)

    return np.random.laplace(loc=origin, scale=scale, size=size)


def readtskyfile():
    """Read in tsky.ascii into a list from which temps can be retrieved"""

    tskypath = os.path.join(fortranpath, 'lookuptables/tsky.ascii'.encode())
    tskylist = []
    with open(tskypath) as f:
        for line in f:
            str_idx = 0
            while str_idx < len(line):
                # each temperature occupies space of 5 chars
                temp_string = line[str_idx:str_idx+5]
                try:
                    tskylist.append(float(temp_string))
                except:
                    pass
                str_idx += 5

    return tskylist
