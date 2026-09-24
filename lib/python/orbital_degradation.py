"""Vectorized translation of gamma{1,2,3}_real.f (Bagchi, Lorimer & Wolfe 2013).

Every numerical choice below (grids, tolerances, iteration caps, evaluation order)
mirrors the Fortran so that results agree to the level of floating-point summation order.
"""

from typing import NamedTuple

import numpy as np

C = 2.99792458e8
G = 6.67384e-11
MSUN = 1.98892e30
PHASES_DEG = np.arange(37) * 10.0
KEPLER_TOL, KEPLER_MAX_ITER = 1e-10, 50
SIMPSON_EPS, SIMPSON_MAX_DOUBLINGS = 1e-6, 60
COARSE_POINTS, FINE_POINTS, REFINE_UNTIL_STEP = 1001, 101, 1e-5
BLOCK_VALUES = 1 << 21


class Elements(NamedTuple):
    """Per-evaluation quantities defining the phase residual of one trial (orbit, epoch, alphas)."""

    omo: np.ndarray
    ecc: np.ndarray
    ap: np.ndarray
    omper: np.ndarray
    tp: np.ndarray
    rl0: np.ndarray
    m_omp: np.ndarray
    a0: np.ndarray
    a1: np.ndarray
    a2: np.ndarray


def degradation_factor(m, tobs, m1, m2, ps, omega, inc, ecc, pb, order=1):
    """Orbital degradation factor gamma_<order> averaged over orbital phase.

    m: harmonic number; tobs: integration time [s]; m1, m2: masses [Msun]; ps: spin period [s];
    omega: longitude of periastron [deg]; inc: inclination [deg]; ecc: eccentricity; pb: orbital
    period [days]. Inputs broadcast against each other; scalars return a scalar.
    """
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2 or 3")
    args = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (m, tobs, m1, m2, ps, omega, inc, ecc, pb)))
    shape = args[0].shape
    m, tobs, m1, m2, ps, omega, inc, ecc, pb = (a.reshape(-1, 1) for a in args)

    m1, m2 = m1 * MSUN, m2 * MSUN
    omper = omega * np.pi / 180.0
    omo = 2.0 * np.pi / (pb * 24.0 * 3600.0)
    omp = 2.0 * np.pi / ps
    a_rel = ((G * (m1 + m2)) / (omo * omo)) ** 0.33333333333333333333
    ap = (a_rel * np.sin(inc * np.pi / 180.0) * m2) / (m1 + m2)

    fc = PHASES_DEG * np.pi / 180.0
    prob = 1.0 / ((1.0 + ecc * np.cos(fc)) * (1.0 + ecc * np.cos(fc)))
    probn = prob / prob[:, 18:19]
    e0 = 2.0 * np.arctan(np.sqrt((1.0 - ecc) / (1.0 + ecc)) * np.tan(fc / 2.0))
    tp = -(e0 - ecc * np.sin(e0)) / omo

    orbit = Elements(*np.broadcast_arrays(omo, ecc, ap, omper, tp, 0.0, m * omp, 0.0, 0.0, 0.0))
    gamma = _max_gamma(orbit, tobs, order)
    result = (probn * gamma).sum(1) / probn.sum(1)
    return result.reshape(shape)[()]


def _max_gamma(orbit, tobs, order):
    """findalpha: coarse grid over the observation, then refine around the best sample."""
    f0 = _true_anomaly(orbit.ecc, -orbit.omo * orbit.tp)
    rl0 = ((orbit.ap * (1.0 - orbit.ecc * orbit.ecc)) / (1.0 + orbit.ecc * np.cos(f0))) * np.sin(f0 + orbit.omper)
    orbit = orbit._replace(rl0=rl0)
    tobs = np.broadcast_to(tobs, rl0.shape)

    t = (tobs / (COARSE_POINTS - 1))[..., None] * np.arange(COARSE_POINTS)
    t0, step = _refinement_window(t, _gamma_grid(orbit, tobs, t, order))
    best = np.zeros(t0.shape)
    active = np.ones(t0.shape, dtype=bool)
    while active.any():
        t = t0[..., None] + step[..., None] * np.arange(FINE_POINTS)
        gamma = _gamma_grid(orbit, tobs, t, order)
        best = np.where(active, gamma.max(-1), best)
        t0, step = _refinement_window(t, gamma)
        active &= step > REFINE_UNTIL_STEP
    return best


def _refinement_window(t, gamma):
    """Start and step of the next grid: [t(n-1), t(n+1)] around the maximum, one-sided at the ends."""
    n = gamma.argmax(-1)
    last = t.shape[-1] - 1
    t_prev, t_n, t_next = (np.take_along_axis(t, k[..., None], -1)[..., 0] for k in (np.maximum(n - 1, 0), n, np.minimum(n + 1, last)))
    step = np.where(n == 0, (t_next - t_n) / 50.0, np.where(n == last, (t_n - t_prev) / 50.0, (t_next - t_prev) / 100.0))
    return t_prev, step


def _gamma_grid(orbit, tobs, t, order):
    """gamma at every trial epoch t, with the acceleration terms of the requested order."""
    omo, ecc, ap, omper = (x[..., None] for x in (orbit.omo, orbit.ecc, orbit.ap, orbit.omper))
    f = _true_anomaly(ecc, omo * (t - orbit.tp[..., None]))
    one_pec = 1.0 + ecc * np.cos(f)
    a2 = (omo * (ap / np.sqrt(1.0 - ecc * ecc))) * (np.cos(f + omper) + ecc * np.cos(omper))
    a1 = a0 = np.zeros(t.shape)
    if order >= 2:
        a1 = (-(omo * omo) * (ap / ((1.0 - ecc * ecc) * (1.0 - ecc * ecc))) * np.sin(f + omper) * one_pec * one_pec) / 2.0
    if order == 3:
        fac3 = -(omo * omo * omo) * (ap / ((1.0 - ecc * ecc) ** 3.5))
        a0 = (fac3 * (one_pec**3.0) * (np.cos(f + omper) + ecc * np.cos(omper) - 3.0 * ecc * np.sin(f + omper) * np.sin(f))) / 6.0

    fields = (x[..., None] for x in (orbit.omo, orbit.ecc, orbit.ap, orbit.omper, orbit.tp, orbit.rl0, orbit.m_omp))
    trials = Elements(*(x.ravel() for x in np.broadcast_arrays(*fields, a0, a1, a2)))
    return _gamma(trials, np.broadcast_to(tobs[..., None], t.shape).ravel()).reshape(t.shape)


def _gamma(trials, tobs):
    """gam + qsimpmnj + trapzd: |integral of exp(i phase)| / tobs, with the cos and sin parts
    converging independently as in the Fortran but sharing each phase evaluation."""
    b = tobs[:, None]
    ost = os = np.full((tobs.size, 2), -1.0e30)
    st = 0.5 * b * _cos_sin(trials, np.stack([np.zeros(tobs.size), tobs], 1)).sum(1)
    s = (4.0 * st - ost) / 3.0
    done = np.abs(s - os) < SIMPSON_EPS * np.abs(os)
    os, ost = s.copy(), st.copy()
    for j in range(2, SIMPSON_MAX_DOUBLINGS + 1):
        idx = np.flatnonzero(~done.all(1))
        if idx.size == 0:
            break
        sub = Elements(*(x[idx] for x in trials))
        points = 2 ** (j - 2)
        width = b[idx] / points
        total = np.zeros((idx.size, 2))
        block = max(1, BLOCK_VALUES // idx.size)
        for start in range(0, points, block):
            k = np.arange(start, min(start + block, points))
            total += _cos_sin(sub, (k + 0.5) * width).sum(1)
        st_new = 0.5 * (st[idx] + b[idx] * total / points)
        s_new = (4.0 * st_new - ost[idx]) / 3.0
        converged = np.abs(s_new - os[idx]) < SIMPSON_EPS * np.abs(os[idx])
        update = ~done[idx]
        s[idx] = np.where(update, s_new, s[idx])
        os[idx] = np.where(update, s_new, os[idx])
        ost[idx] = np.where(update, st_new, ost[idx])
        st[idx] = np.where(update, st_new, st[idx])
        done[idx] |= converged
    return np.sqrt(s[:, 0] * s[:, 0] + s[:, 1] * s[:, 1]) / tobs


def _cos_sin(trials, z):
    """Integrands func1 and func2 at times z (shape (n, points)), stacked on a trailing axis."""
    omo, ecc, ap, omper, tp, rl0, m_omp, a0, a1, a2 = (x[:, None] for x in trials)
    f = _true_anomaly(ecc, omo * (z - tp))
    los = ((ap * (1.0 - ecc * ecc)) / (1.0 + ecc * np.cos(f))) * np.sin(f + omper)
    g2 = (m_omp * (los - rl0 - a0 * z * z * z - a1 * z * z - a2 * z)) / C
    return np.stack([np.cos(g2), np.sin(g2)], -1)


def _true_anomaly(ecc, mean_anom):
    """keplersolve1 + keplersolve2: Newton-Raphson for E (plus one extra step after exit), then f."""
    mean_anom = np.where(mean_anom >= 2.0 * np.pi, mean_anom - 2.0 * np.pi, mean_anom)

    def newton(ea):
        return (mean_anom + ecc * np.sin(ea) - ea * ecc * np.cos(ea)) / (1.0 - ecc * np.cos(ea))

    ea = mean_anom + ecc * np.sin(mean_anom) * (1.0 + ecc * np.cos(mean_anom))
    active = np.ones(ea.shape, dtype=bool)
    for _ in range(KEPLER_MAX_ITER):
        ea_new = newton(ea)
        converged = np.abs(ea - ea_new) < KEPLER_TOL
        ea = np.where(active, ea_new, ea)
        active &= ~converged
        if not active.any():
            break
    ea = newton(ea)
    return 2.0 * np.arctan(np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(ea / 2.0))
