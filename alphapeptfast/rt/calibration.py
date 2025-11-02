"""
Robust RT calibration using PCHIP interpolation with outlier removal.

This module provides production-grade iRT → RT calibration using:
- MAD-based robust outlier removal
- Fritsch-Carlson PCHIP with Hyman endpoint filter (monotone-preserving)
- Median-of-secants for stable tail slopes (prevents wild extrapolation)
- Dead-time offset estimation from early eluting peptides

All functions are Numba-compiled for performance.

Example
-------
>>> from alphapeptfast.rt.calibration import fit_pchip_irt_to_rt, predict_pchip_irt
>>> import numpy as np
>>>
>>> # Fit calibration from high-confidence peptides
>>> irt_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
>>> rt_observed = np.array([120.0, 350.0, 580.0, 810.0, 1040.0])
>>> model = fit_pchip_irt_to_rt(irt_values, rt_observed)
>>>
>>> # Predict RT for new peptides
>>> new_irt = np.array([0.2, 0.4, 0.6])
>>> predicted_rt = predict_pchip_irt(model, new_irt)
"""

import numpy as np
from numba import njit


# ========== Numba utilities ==========

@njit
def _median(a):
    """Compute median (Numba-optimized)."""
    b = a.copy()
    b.sort()
    n = b.size
    mid = n // 2
    if n % 2 == 1:
        return b[mid]
    else:
        return 0.5 * (b[mid-1] + b[mid])


@njit
def _mad(a):
    """Compute Median Absolute Deviation (MAD) with small epsilon."""
    med = _median(a)
    d = np.abs(a - med)
    return _median(d) + 1e-12


@njit
def _linear_fit(x, y):
    """
    Ordinary least squares: y = a*x + b

    Closed-form solution, stable for n ~ 1e3.
    """
    n = x.size
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        xi = x[i]
        yi = y[i]
        sx += xi
        sy += yi
        sxx += xi*xi
        sxy += xi*yi
    den = n*sxx - sx*sx
    if abs(den) < 1e-24:
        a = 0.0
        b = sy / n
    else:
        a = (n*sxy - sx*sy) / den
        b = (sy - a*sx) / n
    return a, b


@njit
def _sort_by_x(x, y):
    """Sort (x, y) pairs by x."""
    idx = np.argsort(x)
    return x[idx], y[idx]


@njit
def _dedup_mean_sorted(x, y, tol):
    """
    Collapse runs where consecutive x differences <= tol, using MEAN of y.

    x must be sorted.
    """
    n = x.size
    if n == 0:
        return x, y

    gx = np.empty(n, dtype=np.float64)
    gy = np.empty(n, dtype=np.float64)
    gcount = 0

    run_sum_x = x[0]
    run_sum_y = y[0]
    run_cnt = 1

    for i in range(1, n):
        if x[i] - x[i-1] <= tol:
            run_sum_x += x[i]
            run_sum_y += y[i]
            run_cnt += 1
        else:
            gx[gcount] = run_sum_x / run_cnt
            gy[gcount] = run_sum_y / run_cnt
            gcount += 1
            run_sum_x = x[i]
            run_sum_y = y[i]
            run_cnt = 1

    # Flush last run
    gx[gcount] = run_sum_x / run_cnt
    gy[gcount] = run_sum_y / run_cnt
    gcount += 1

    return gx[:gcount], gy[:gcount]


@njit
def _remove_nonincreasing(x, y):
    """Ensure strictly increasing x (drop duplicates/backsteps, keep first)."""
    n = x.size
    if n <= 1:
        return x, y

    rx = np.empty(n, dtype=np.float64)
    ry = np.empty(n, dtype=np.float64)
    m = 1
    rx[0] = x[0]
    ry[0] = y[0]
    last = x[0]

    for i in range(1, n):
        if x[i] > last:
            rx[m] = x[i]
            ry[m] = y[i]
            last = x[i]
            m += 1

    return rx[:m], ry[:m]


@njit
def _secants(x, y):
    """Compute secants (finite differences) between consecutive points."""
    m = x.size - 1
    out = np.empty(m, dtype=np.float64)
    for i in range(m):
        dx = x[i+1] - x[i]
        if dx <= 1e-18:
            out[i] = 0.0
        else:
            out[i] = (y[i+1] - y[i]) / dx
    return out


# ========== PCHIP slopes (Fritsch–Carlson with Hyman endpoint filter) ==========

@njit
def _pchip_slopes(x, y):
    """
    Compute PCHIP slopes using Fritsch-Carlson algorithm with Hyman filter.

    Guarantees monotonicity preservation.
    """
    n = x.size
    m = np.zeros(n, dtype=np.float64)

    if n == 1:
        m[0] = 0.0
        return m
    if n == 2:
        s = (y[1]-y[0]) / (x[1]-x[0])
        m[0] = s
        m[1] = s
        return m

    h = np.empty(n-1, dtype=np.float64)
    d = np.empty(n-1, dtype=np.float64)
    for i in range(n-1):
        h[i] = x[i+1]-x[i]
        d[i] = (y[i+1]-y[i]) / h[i]

    # Interior points
    for i in range(1, n-1):
        if d[i-1]*d[i] > 0.0:
            w1 = 2.0*h[i] + h[i-1]
            w2 = h[i] + 2.0*h[i-1]
            m[i] = (w1 + w2) / (w1/d[i-1] + w2/d[i])
        else:
            m[i] = 0.0

    # Endpoints (Hyman filter)
    m0 = ((2.0*h[0] + h[1])*d[0] - h[0]*d[1]) / (h[0] + h[1])
    if np.sign(m0) != np.sign(d[0]):
        m0 = 0.0
    elif (np.sign(d[0]) != np.sign(d[1])) and (abs(m0) > 3.0*abs(d[0])):
        m0 = 3.0*d[0]

    mn = ((2.0*h[n-2] + h[n-3])*d[n-2] - h[n-2]*d[n-3]) / (h[n-3] + h[n-2])
    if np.sign(mn) != np.sign(d[n-2]):
        mn = 0.0
    elif (np.sign(d[n-2]) != np.sign(d[n-3])) and (abs(mn) > 3.0*abs(d[n-2])):
        mn = 3.0*d[n-2]

    m[0] = m0
    m[n-1] = mn
    return m


# ========== evaluation ==========

@njit
def _binary_search_interval(x, z):
    """
    Find i such that x[i] <= z <= x[i+1].

    Assumes x strictly increasing and z within range.
    """
    lo = 0
    hi = x.size - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if z < x[mid]:
            hi = mid - 1
        elif z > x[mid+1]:
            lo = mid + 1
        else:
            return mid
    # Fallback
    if z <= x[1]:
        return 0
    return x.size - 2


@njit
def _eval_pchip_piece(x, y, m, z):
    """Evaluate PCHIP interpolation at point z."""
    i = _binary_search_interval(x, z)
    xi = x[i]
    xi1 = x[i+1]
    hi = xi1 - xi
    t = (z - xi) / hi
    t2 = t*t
    t3 = t2*t

    # Hermite basis functions
    h00 = (2.0*t3 - 3.0*t2 + 1.0)
    h10 = (t3 - 2.0*t2 + t) * hi
    h01 = (-2.0*t3 + 3.0*t2)
    h11 = (t3 - t2) * hi

    return h00*y[i] + h10*m[i] + h01*y[i+1] + h11*m[i+1]


@njit
def _median_of_slice(a, start, end):
    """Median of a[start:end] (end exclusive)."""
    b = a[start:end].copy()
    b.sort()
    n = b.size
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return b[mid]
    else:
        return 0.5*(b[mid-1] + b[mid])


# ========== Public API ==========

@njit
def fit_pchip_irt_to_rt(irt, rt, dedup_tol=1e-4, mad_k=3.5, tail_k=8, estimate_t0=True):
    """
    Fit robust PCHIP calibration: iRT → RT (seconds).

    Parameters
    ----------
    irt : ndarray
        Indexed retention time values (0-1 normalized scale)
    rt : ndarray
        Observed retention time values (seconds)
    dedup_tol : float, optional
        Tolerance for collapsing near-duplicate iRT values (default: 1e-4)
    mad_k : float, optional
        Outlier threshold in MAD units (default: 3.5)
    tail_k : int, optional
        Number of secants to use for tail slope estimation (default: 8)
    estimate_t0 : bool, optional
        Estimate dead-time offset from early eluting peptides (default: True)

    Returns
    -------
    model : tuple
        Calibration model: (x, y_shift, m, xmin, xmax, ymin, ymax, m_left, m_right, t0)
        Pass this to predict_pchip_irt() for RT prediction.

    Notes
    -----
    - Performs MAD-based outlier removal before fitting
    - Uses Fritsch-Carlson PCHIP with Hyman filter for monotonicity
    - Tail slopes computed from median of secants (robust to outliers)
    - Dead-time offset (t0) estimated from early eluting peptides

    Examples
    --------
    >>> irt_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> rt_observed = np.array([120.0, 350.0, 580.0, 810.0, 1040.0])
    >>> model = fit_pchip_irt_to_rt(irt_values, rt_observed)
    """
    # Sort
    x, y = _sort_by_x(irt.astype(np.float64), rt.astype(np.float64))

    # De-dup very close iRTs
    x, y = _dedup_mean_sorted(x, y, dedup_tol)

    n0 = x.size
    if n0 == 0:
        return (x, y, np.zeros(0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Robust first-pass linear fit → MAD trim
    a, b = _linear_fit(x, y)
    y_lin = a*x + b
    res = y - y_lin
    mad = _mad(res)
    keep = np.abs(res) <= mad_k * mad
    x = x[keep]
    y = y[keep]

    # Ensure strictly increasing x
    x, y = _remove_nonincreasing(x, y)
    n = x.size
    if n == 0:
        return (x, y, np.zeros(0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if n == 1:
        m = np.array([0.0])
        return (x, y, m, x[0], x[0], y[0], y[0], 0.0, 0.0, 0.0)

    # Optional dead-time offset
    if estimate_t0 and n >= 4:
        k0 = max(3, min(12, int(0.05 * n)))
        sec = _secants(x, y)
        if sec.size == 0:
            m_early = 0.0
        else:
            m_early = _median_of_slice(sec, 0, min(k0, sec.size))
        bvals = y[:min(k0+1, n)] - m_early * x[:min(k0+1, n)]
        t0 = _median(bvals)
        if t0 < 0.0:
            t0 = 0.0
    else:
        t0 = 0.0

    y_shift = y - t0

    # PCHIP slopes
    m = _pchip_slopes(x, y_shift)

    # Cache tails
    xmin = x[0]
    xmax = x[-1]
    ymin = y_shift[0]
    ymax = y_shift[-1]

    # Robust tail slopes from secants
    sec = _secants(x, y_shift)
    k = min(tail_k, sec.size)
    if k == 0:
        m_left = 0.0
        m_right = 0.0
    else:
        m_left = _median_of_slice(sec, 0, k)
        m_right = _median_of_slice(sec, sec.size - k, sec.size)
        if m_left < 1e-12:
            m_left = 1e-12
        if m_right < 1e-12:
            m_right = 1e-12

    return (x, y_shift, m, xmin, xmax, ymin, ymax, m_left, m_right, t0)


@njit
def predict_pchip_irt(model, irt_query):
    """
    Predict RT (seconds) from iRT using calibration model.

    Parameters
    ----------
    model : tuple
        Calibration model from fit_pchip_irt_to_rt()
    irt_query : ndarray
        iRT values to predict RT for (0-1 normalized scale)

    Returns
    -------
    rt_predicted : ndarray
        Predicted RT values (seconds)

    Notes
    -----
    - Uses PCHIP interpolation within calibration range
    - Linear extrapolation outside range (using robust tail slopes)
    - Returns NaN for empty model

    Examples
    --------
    >>> model = fit_pchip_irt_to_rt(irt_train, rt_train)
    >>> rt_pred = predict_pchip_irt(model, irt_test)
    """
    (x, y_shift, m, xmin, xmax, ymin, ymax, m_left, m_right, t0) = model
    z = irt_query.astype(np.float64)
    out = np.empty(z.size, dtype=np.float64)

    n = x.size
    if n == 0:
        for i in range(z.size):
            out[i] = np.nan
        return out
    if n == 1:
        for i in range(z.size):
            out[i] = y_shift[0] + t0
        return out

    for i in range(z.size):
        zi = z[i]
        if zi < xmin:
            # Linear extrapolation below
            out[i] = ymin + m_left * (zi - xmin) + t0
        elif zi > xmax:
            # Linear extrapolation above
            out[i] = ymax + m_right * (zi - xmax) + t0
        else:
            # PCHIP interpolation
            out[i] = _eval_pchip_piece(x, y_shift, m, zi) + t0

    return out
