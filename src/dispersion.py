"""
dispersion.py
=============
Linear surface-wave dispersion relation utilities.

    ω²  =  g · k · tanh(k · h)

All functions are pure (no I/O, no global state) and accept scalar or
numpy-array inputs unless stated otherwise.

References
----------
Dean & Dalrymple (1991) Water Wave Mechanics for Engineers and Scientists.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core dispersion
# ---------------------------------------------------------------------------

def angular_frequency(k, h, g=9.81):
    """
    Compute angular frequency ω [rad/s] from wavenumber k [rad/m]
    and water depth h [m] using the exact dispersion relation.

    Parameters
    ----------
    k : float or ndarray   Wavenumber [rad/m]
    h : float or ndarray   Water depth [m]  (positive downward)
    g : float              Gravitational acceleration [m/s²]

    Returns
    -------
    omega : float or ndarray   Angular frequency [rad/s]
    """
    k = np.asarray(k, dtype=float)
    h = np.asarray(h, dtype=float)
    return np.sqrt(g * k * np.tanh(k * h))


def phase_speed(k, h, g=9.81):
    """
    Phase speed c = ω/k  [m/s].

    Parameters
    ----------
    k : float or ndarray   Wavenumber [rad/m]
    h : float or ndarray   Water depth [m]
    g : float              Gravitational acceleration [m/s²]

    Returns
    -------
    c : float or ndarray   Phase speed [m/s]
    """
    k     = np.asarray(k, dtype=float)
    omega = angular_frequency(k, h, g)
    return omega / k


def group_speed(k, h, g=9.81):
    """
    Group speed c_g = dω/dk  [m/s]  (analytical derivative).

    c_g = (ω / 2k) · [1 + 2kh / sinh(2kh)]

    Parameters
    ----------
    k : float or ndarray   Wavenumber [rad/m]
    h : float or ndarray   Water depth [m]
    g : float              Gravitational acceleration [m/s²]

    Returns
    -------
    cg : float or ndarray  Group speed [m/s]
    """
    k     = np.asarray(k, dtype=float)
    h     = np.asarray(h, dtype=float)
    omega = angular_frequency(k, h, g)
    kh    = k * h
    # Avoid division by zero for very shallow or zero-k cases
    with np.errstate(invalid="ignore", divide="ignore"):
        n  = 0.5 * (1.0 + np.where(kh > 1e-6,
                                    2.0 * kh / np.sinh(2.0 * kh),
                                    1.0))
    return n * omega / k


# ---------------------------------------------------------------------------
# Wave period
# ---------------------------------------------------------------------------

def period_from_k(k, h, g=9.81):
    """
    Peak wave period T = 2π/ω  [s] from wavenumber and depth.

    Also diagnoses the wave regime:
        deep         kh > π        (tanh(kh) ≈ 1)
        shallow      kh < 0.1      (tanh(kh) ≈ kh)
        intermediate otherwise

    Parameters
    ----------
    k : float   Wavenumber [rad/m]
    h : float   Water depth [m]

    Returns
    -------
    T      : float   Wave period [s]
    regime : str     'deep' | 'intermediate' | 'shallow'
    """
    k  = float(k)
    h  = float(h)
    kh = k * h

    if kh > np.pi:
        regime = "deep"
    elif kh < 0.1:
        regime = "shallow"
    else:
        regime = "intermediate"

    omega = float(angular_frequency(k, h, g))
    T     = 2.0 * np.pi / omega
    return T, regime


def period_from_wavelength(lam, h, g=9.81):
    """
    Convenience wrapper: peak period from wavelength λ [m] and depth h [m].

    Parameters
    ----------
    lam : float   Wavelength [m]
    h   : float   Water depth [m]

    Returns
    -------
    T      : float   Wave period [s]
    regime : str
    """
    k = 2.0 * np.pi / lam
    return period_from_k(k, h, g)


# ---------------------------------------------------------------------------
# Wavenumber from period  (Newton–Raphson solve)
# ---------------------------------------------------------------------------

def k_from_period(T, h, g=9.81, tol=1e-8, max_iter=50):
    """
    Solve ω² = g·k·tanh(k·h) for k given period T and depth h.

    Uses the Hunt (1979) explicit approximation as the initial guess,
    then refines with Newton–Raphson.

    Parameters
    ----------
    T        : float   Wave period [s]
    h        : float   Water depth [m]
    g        : float   Gravitational acceleration [m/s²]
    tol      : float   Convergence tolerance on k
    max_iter : int     Maximum Newton iterations

    Returns
    -------
    k : float   Wavenumber [rad/m]
    """
    omega = 2.0 * np.pi / T
    omega2 = omega**2

    # Hunt (1979) initial guess
    x0    = omega2 * h / g
    y     = x0 / (1.0 - np.exp(-(x0**1.0557))) ** 0.9046
    k     = y / h

    # Newton–Raphson:  f(k) = g·k·tanh(k·h) - ω²  = 0
    for _ in range(max_iter):
        kh   = k * h
        th   = np.tanh(kh)
        f    = g * k * th - omega2
        df   = g * (th + kh * (1.0 - th**2))
        dk   = -f / df
        k   += dk
        if abs(dk) < tol * k:
            break

    return float(k)
