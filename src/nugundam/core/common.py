"""Shared numerical helpers used across the refactored code base."""
from __future__ import annotations

import numpy as np


def set_threads(t: int) -> int:
    """
    Set threads.
    
    Parameters
    ----------
    t : object
        Value for ``t``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    from multiprocessing import cpu_count

    maxt = cpu_count()
    if t <= 0:
        return maxt
    return min(t, maxt)


def makebins(nsep: int, sepmin: float, dsep: float, logsep: bool):
    """
    Construct bin edges, centers, and widths for linear or logarithmic separations.
    
    Parameters
    ----------
    nsep : object
        Value for ``nsep``.
    sepmin : object
        Value for ``sepmin``.
    dsep : object
        Value for ``dsep``.
    logsep : object
        Value for ``logsep``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    if logsep:
        sep = sepmin * 10.0 ** (np.arange(nsep + 1) * dsep)
        sepc = np.sqrt(sep[:-1] * sep[1:])
        ds = np.diff(np.log10(sep))
    else:
        sep = sepmin + np.arange(nsep + 1) * dsep
        sepc = 0.5 * (sep[:-1] + sep[1:])
        ds = np.diff(sep)
    return sep, (sep, sepc, ds)


def bound2d(dec_arrays, ra_arrays=None):
    """
    Compute common angular bounds spanning one or more catalogs.
    
    Parameters
    ----------
    dec_arrays : object
        Value for ``dec_arrays``.
    ra_arrays : object, optional
        Value for ``ra_arrays``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    dec_concat = np.concatenate([np.asarray(a) for a in dec_arrays])
    decmin = max(float(np.min(dec_concat)) - 0.001, -90.0)
    decmax = min(float(np.max(dec_concat)) + 0.001, 90.0)
    return (0.0, 360.0, decmin, decmax)

#def bound2d(dec_arrays, ra_arrays=None):
#    dec_concat = np.concatenate([np.asarray(a) for a in dec_arrays])
#    decmin = float(np.min(dec_concat))
#    decmax = float(np.max(dec_concat))
#    if ra_arrays is None:
#        return (0.0, 360.0, decmin, decmax)
#    ra_concat = np.concatenate([np.asarray(a) for a in ra_arrays])
#    return (float(np.min(ra_concat)), float(np.max(ra_concat)), decmin, decmax)


def cross0guess(ra):
    """
    Guess whether right-ascension wrapping across 0/360 degrees should be enabled.
    
    Parameters
    ----------
    ra : object
        Value for ``ra``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    ra = np.asarray(ra)
    s = np.sort(ra)
    if len(s) < 2:
        return False
    return bool(np.max(np.diff(s)) > 180.0)


#def radec2xyz(ra, dec, r=0.5):
#    x = r*np.cos(dec)*np.sin(ra)
#    y = r*np.cos(dec)*np.cos(ra)
#    z = r*np.sin(dec)
#    return x, y, z


def radec2xyz(ra_rad, dec_rad):
    """
    Convert angular coordinates in radians to Cartesian direction cosines.
    
    Parameters
    ----------
    ra_rad : object
        Value for ``ra_rad``.
    dec_rad : object
        Value for ``dec_rad``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return x, y, z
