"""Faster evaluation of gamma{1,2,3} (Bagchi, Lorimer & Wolfe 2013), self-contained, compiled with Numba.

Same algorithm and numerical choices as fortran_port. The Simpson loop of every orbit (one binary at
one orbital phase) runs in one compiled kernel: the line-of-sight position is computed once per
integration point and shared by that orbit's trial epochs, which are spread over all cores, and a
cos or sin integral that has converged is no longer evaluated. Sums accumulate sequentially per
trial as in the Fortran, so results differ from fortran_port by summation order only.
"""

import numpy as np
from numba import njit, prange

C = 2.99792458e8
G = 6.67384e-11
MSUN = 1.98892e30
PHASES_DEG = np.arange(37) * 10.0
KEPLER_TOL, KEPLER_MAX_ITER = 1e-10, 50
SIMPSON_EPS, SIMPSON_MAX_DOUBLINGS = 1e-6, 60
COARSE_POINTS, FINE_POINTS, REFINE_UNTIL_STEP = 1001, 101, 1e-5
BLOCK_POINTS = 4096


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

    orbit = tuple(np.ascontiguousarray(x.ravel()) for x in np.broadcast_arrays(omo, ecc, ap, omper, tp, m * omp, tobs))
    gamma = _max_gamma(orbit, order).reshape(probn.shape)
    result = (probn * gamma).sum(1) / probn.sum(1)
    return result.reshape(shape)[()]


def _max_gamma(orbit, order):
    """findalpha: coarse grid over the observation, then refine around the best sample."""
    tobs = orbit[6]
    t = (tobs / (COARSE_POINTS - 1))[:, None] * np.arange(COARSE_POINTS)
    t0, step = _refinement_window(t, _gamma_grid(orbit, t, order))
    best = np.zeros(t0.shape)
    active = np.ones(t0.shape, dtype=bool)
    while active.any():
        t = t0[:, None] + step[:, None] * np.arange(FINE_POINTS)
        gamma = _gamma_grid(orbit, t, order)
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


def _gamma_grid(orbit, t, order):
    return _gamma_kernel(*orbit, *_alphas(*orbit[:5], t, order))


@njit(cache=True)
def _true_anomaly(ecc, mean_anom):
    """keplersolve1 + keplersolve2: Newton-Raphson for E (plus one extra step after exit), then f."""
    if mean_anom >= 2.0 * np.pi:
        mean_anom -= 2.0 * np.pi
    ea = mean_anom + ecc * np.sin(mean_anom) * (1.0 + ecc * np.cos(mean_anom))
    for _ in range(KEPLER_MAX_ITER):
        ea_new = (mean_anom + ecc * np.sin(ea) - ea * ecc * np.cos(ea)) / (1.0 - ecc * np.cos(ea))
        converged = abs(ea - ea_new) < KEPLER_TOL
        ea = ea_new
        if converged:
            break
    ea = (mean_anom + ecc * np.sin(ea) - ea * ecc * np.cos(ea)) / (1.0 - ecc * np.cos(ea))
    return 2.0 * np.arctan(np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(ea / 2.0))


@njit(cache=True)
def _los(omo, ecc, ap, omper, tp, z):
    """Line-of-sight position of one orbit at time z."""
    f = _true_anomaly(ecc, omo * (z - tp))
    return ((ap * (1.0 - ecc * ecc)) / (1.0 + ecc * np.cos(f))) * np.sin(f + omper)


@njit(cache=True)
def _alphas(omo, ecc, ap, omper, tp, t, order):
    """Velocity, acceleration/2 and jerk/6 terms at the trial epochs t (n_orbits, n_trials)."""
    a0, a1, a2 = np.zeros(t.shape), np.zeros(t.shape), np.zeros(t.shape)
    for o in range(t.shape[0]):
        for i in range(t.shape[1]):
            f = _true_anomaly(ecc[o], omo[o] * (t[o, i] - tp[o]))
            one_pec = 1.0 + ecc[o] * np.cos(f)
            a2[o, i] = (omo[o] * (ap[o] / np.sqrt(1.0 - ecc[o] * ecc[o]))) * (np.cos(f + omper[o]) + ecc[o] * np.cos(omper[o]))
            if order >= 2:
                a1[o, i] = (-(omo[o] * omo[o]) * (ap[o] / ((1.0 - ecc[o] * ecc[o]) * (1.0 - ecc[o] * ecc[o]))) * np.sin(f + omper[o]) * one_pec * one_pec) / 2.0
            if order == 3:
                fac3 = -(omo[o] * omo[o] * omo[o]) * (ap[o] / ((1.0 - ecc[o] * ecc[o]) ** 3.5))
                a0[o, i] = (fac3 * (one_pec**3.0) * (np.cos(f + omper[o]) + ecc[o] * np.cos(omper[o]) - 3.0 * ecc[o] * np.sin(f + omper[o]) * np.sin(f))) / 6.0
    return a0, a1, a2


@njit(parallel=True, cache=True)
def _gamma_kernel(omo, ecc, ap, omper, tp, m_omp, tobs, a0, a1, a2):
    """gam + qsimpmnj + trapzd for every trial of every orbit: |integral of exp(i g2)| / tobs.

    Column 0 of the (n_trials, 2) work arrays is the cos integral, column 1 the sin integral; as in
    the Fortran each converges on its own."""
    n_orb, n_t = a0.shape
    out = np.empty((n_orb, n_t))
    z = np.empty(BLOCK_POINTS)
    los = np.empty(BLOCK_POINTS)
    for o in range(n_orb):
        b = tobs[o]
        rl0 = _los(omo[o], ecc[o], ap[o], omper[o], tp[o], 0.0)
        l_end = _los(omo[o], ecc[o], ap[o], omper[o], tp[o], b)
        s = np.empty((n_t, 2))
        os_ = np.full((n_t, 2), -1.0e30)
        st = np.empty((n_t, 2))
        ost = np.full((n_t, 2), -1.0e30)
        done = np.zeros((n_t, 2), dtype=np.bool_)
        total = np.zeros((n_t, 2))
        for i in range(n_t):
            g_start = (m_omp[o] * (rl0 - rl0 - a0[o, i] * 0.0 * 0.0 * 0.0 - a1[o, i] * 0.0 * 0.0 - a2[o, i] * 0.0)) / C
            g_end = (m_omp[o] * (l_end - rl0 - a0[o, i] * b * b * b - a1[o, i] * b * b - a2[o, i] * b)) / C
            st[i, 0] = 0.5 * b * (np.cos(g_start) + np.cos(g_end))
            st[i, 1] = 0.5 * b * (np.sin(g_start) + np.sin(g_end))
            for c in range(2):
                s[i, c] = (4.0 * st[i, c] - ost[i, c]) / 3.0
                done[i, c] = abs(s[i, c] - os_[i, c]) < SIMPSON_EPS * abs(os_[i, c])
                os_[i, c] = s[i, c]
                ost[i, c] = st[i, c]
        for j in range(2, SIMPSON_MAX_DOUBLINGS + 1):
            n_active = 0
            for i in range(n_t):
                if not (done[i, 0] and done[i, 1]):
                    n_active += 1
            if n_active == 0:
                break
            points = 2 ** (j - 2)
            width = b / points
            total[:, :] = 0.0
            for start in range(0, points, BLOCK_POINTS):
                n_block = min(BLOCK_POINTS, points - start)
                for k in range(n_block):
                    z[k] = (start + k + 0.5) * width
                    los[k] = _los(omo[o], ecc[o], ap[o], omper[o], tp[o], z[k])
                for i in prange(n_t):
                    need_cos, need_sin = not done[i, 0], not done[i, 1]
                    if not (need_cos or need_sin):
                        continue
                    sum_cos, sum_sin = 0.0, 0.0
                    for k in range(n_block):
                        g2 = (m_omp[o] * (los[k] - rl0 - a0[o, i] * z[k] * z[k] * z[k] - a1[o, i] * z[k] * z[k] - a2[o, i] * z[k])) / C
                        if need_cos:
                            sum_cos += np.cos(g2)
                        if need_sin:
                            sum_sin += np.sin(g2)
                    total[i, 0] += sum_cos
                    total[i, 1] += sum_sin
            for i in range(n_t):
                for c in range(2):
                    if done[i, c]:
                        continue
                    st_new = 0.5 * (st[i, c] + b * total[i, c] / points)
                    s_new = (4.0 * st_new - ost[i, c]) / 3.0
                    if abs(s_new - os_[i, c]) < SIMPSON_EPS * abs(os_[i, c]):
                        done[i, c] = True
                    s[i, c] = s_new
                    os_[i, c] = s_new
                    ost[i, c] = st_new
                    st[i, c] = st_new
        for i in range(n_t):
            out[o, i] = np.sqrt(s[i, 0] * s[i, 0] + s[i, 1] * s[i, 1]) / b
    return out
