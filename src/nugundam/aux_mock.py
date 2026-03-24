"""
MKONE (Mock Light Cone Utilities)
---------------------------------------------------

Generate mock light cones with filamentary structure or Gaussian blobs added.

The generated structures are intended for teaching, visualization, or testing
correlation-function pipelines. They are not meant to be physically realistic,
except for :func:`uniform_sky`, which produces uniform random points on the sky.

Main routines
-------------
- :func:`mcone_filam`: generate a mock light cone with filamentary structure.
- :func:`mcone_gaussblobs`: generate a mock light cone with Gaussian blobs or
  "clusters".
- :func:`filament_box`: generate random 3D points emulating filamentary
  structure in a box.
- :func:`fill_cone`: fill a light-cone volume with replicated point cubes.
- :func:`uniform_sky`: generate uniform random points (ra, dec) on the sky.

Notes
-----
This file is a Python 3.11+ modernization of an older Python 2 implementation.
The goal is to preserve the original public API and overall behavior while
updating syntax and replacing deprecated NumPy/SciPy usage.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import cartesian_to_spherical
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.spatial import cKDTree

try:
    import sampc  
except Exception:
    sampc = None


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _stack_columns(*cols: np.ndarray) -> np.ndarray:
    """Stack 1D arrays as columns into a 2D array."""
    return np.column_stack(cols)



def _tree_query_indices(tree: cKDTree, pts: np.ndarray) -> np.ndarray:
    """Return nearest-neighbour indices in a way compatible with old/new SciPy."""
    try:
        return tree.query(pts, workers=-1)[1]
    except TypeError:
        try:
            return tree.query(pts, n_jobs=-1)[1]
        except TypeError:
            return tree.query(pts)[1]


# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================

def ra_dec_to_xyz(ra, dec):
    """Convert (ra, dec) in degrees to Cartesian coordinates (x, y, z)."""
    ra = np.asarray(ra)
    dec = np.asarray(dec)

    sin_ra = np.sin(np.deg2rad(ra))
    cos_ra = np.cos(np.deg2rad(ra))
    sin_dec = np.sin(np.pi / 2.0 - np.deg2rad(dec))
    cos_dec = np.cos(np.pi / 2.0 - np.deg2rad(dec))
    return cos_ra * sin_dec, sin_ra * sin_dec, cos_dec



def rdz2xyz(ra, dec, reds, cosmo):
    """Convert (ra, dec, z) to Cartesian coordinates for a given cosmology."""
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    reds = np.asarray(reds)

    sin_ra = np.sin(np.deg2rad(ra))
    cos_ra = np.cos(np.deg2rad(ra))
    sin_dec = np.sin(np.pi / 2.0 - np.deg2rad(dec))
    cos_dec = np.cos(np.pi / 2.0 - np.deg2rad(dec))

    r = cosmo.comoving_distance(reds).value
    x = r * cos_ra * sin_dec
    y = r * sin_ra * sin_dec
    z = r * cos_dec
    return x, y, z



def ranshell3d(xc, yc, zc, r1, r2, n=500):
    """
    Generate ``n`` random points in a spherical shell centered at ``(xc, yc, zc)``.

    The shell has inner radius ``r1`` and outer radius ``r2``.
    """
    th = np.random.random(n) * (2.0 * np.pi)
    z0 = np.random.random(n) * 2.0 - 1.0
    x0 = np.sqrt(1.0 - z0**2) * np.cos(th)
    y0 = np.sqrt(1.0 - z0**2) * np.sin(th)
    t = np.random.random(n) * (r2**3 - r1**3) + r1**3
    r = t ** (1.0 / 3.0)
    x = x0 * r + xc
    y = y0 * r + yc
    z = z0 * r + zc
    return x, y, z



def raz2xy(ra, r):
    """Convert (ra, r) to planar polar coordinates for 2D cone plots."""
    theta = np.deg2rad(ra)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y



def uniform_sphere(n, R):
    """Draw ``n`` uniform random points inside a sphere of radius ``R``."""
    phi = np.random.random(n) * 2.0 * np.pi
    costheta = np.random.random(n) * 2.0 - 1.0
    h = np.random.random(n)

    theta = np.arccos(costheta)
    r = R * (h ** (1.0 / 3.0))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return _stack_columns(x, y, z)



def scatter3d(xyz, y=None, z=None, s=1.0, c="k"):
    """
    Basic 3D scatter plot.

    Parameters
    ----------
    xyz : array-like
        Either an array of shape ``(n, 3)`` or the x-coordinate vector.
    y, z : array-like, optional
        y and z coordinate vectors when ``xyz`` is not a ``(n, 3)`` array.
    s : float, optional
        Marker size.
    c : str or array-like, optional
        Marker color.

    Returns
    -------
    fig, ax
        Matplotlib figure and 3D axes.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    xyz_arr = np.asarray(xyz)
    if xyz_arr.ndim == 2 and xyz_arr.shape[1] == 3:
        x, y, z = xyz_arr[:, 0], xyz_arr[:, 1], xyz_arr[:, 2]
    else:
        x = xyz
        if y is None or z is None:
            raise ValueError("When xyz is not an (n, 3) array, y and z must be provided.")
    ax.scatter(x, y, z, s=s, c=c)
    return fig, ax



def send(dat, fname, cols=None, disc=True):
    """Send a 2D array or Astropy table to TOPCAT via SAMP."""
    if sampc is None:
        raise ImportError(
            "The optional dependency 'sampc' is not installed. "
            "Install it to use send()."
        )
    from sampc import Client

    client = Client()
    client.send(dat, fname, cols=cols, disc=disc)



def randomize(xyz, b):
    """
    Randomize 3D points in a box by mirroring axes and permuting coordinates.

    This preserves the internal structure of the box while reducing obvious
    repetition patterns when the same cube is stacked multiple times.

    Parameters
    ----------
    xyz : array-like of shape (n, 3)
        Input 3D points.
    b : float
        Box size.

    Returns
    -------
    numpy.ndarray
        Randomized 3D points with shape ``(n, 3)``.
    """
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy()

    mirror_map = {
        1: (False, False, False),
        2: (True, False, False),
        3: (False, True, False),
        4: (False, False, True),
        5: (True, True, False),
        6: (False, True, True),
        7: (True, False, True),
        8: (True, True, False),  # preserved from original code
    }
    mx, my, mz = mirror_map[np.random.randint(1, 9)]
    if mx:
        x = b - x
    if my:
        y = b - y
    if mz:
        z = b - z

    perm_map = {
        1: (x, y, z),
        2: (z, x, y),
        3: (y, z, x),
        4: (x, z, y),
        5: (y, x, z),
        6: (z, y, x),
    }
    px, py, pz = perm_map[np.random.randint(1, 7)]
    return _stack_columns(px, py, pz)


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def uniform_sky(ralim, declim, n=1):
    """Generate ``n`` uniform random points (ra, dec) in a sky rectangle."""
    zlim = np.sin(np.pi * np.asarray(declim) / 180.0)
    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(n)
    dec = np.rad2deg(np.arcsin(z))
    ra = ralim[0] + (ralim[1] - ralim[0]) * np.random.random(n)
    return ra, dec



def mcone_gaussblobs(
    ralim,
    declim,
    zlim,
    n=20000,
    ufrac=0.7,
    ncen=500,
    cradlim=None,
    rand_elong=False,
    fix_nmemb=False,
    random_state=None,
    doplot=False,
    colorize=False,
    cosmo=None,
    oformat="table",
):
    """
    Generate a mock light cone with random Gaussian blobs or "clusters".

    Parameters are kept intentionally close to the legacy implementation.
    """
    if random_state is not None:
        np.random.seed(seed=random_state)
    if cradlim is None:
        cradlim = [0.5, 25.0]
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

    nunif = int(ufrac * n)
    nnotunif = n - nunif

    # Randomly pick centers for the clusters.
    Cra, Cdec = uniform_sky(ralim, declim, n=ncen)
    Cred = (zlim[1] - zlim[0]) * np.random.random(ncen) + zlim[0]
    Cx, Cy, Cz = rdz2xyz(Cra, Cdec, Cred, cosmo)

    # Set fixed or random radii.
    if isinstance(cradlim, list):
        Crad = (cradlim[1] - cradlim[0]) * np.random.random_sample(ncen) + cradlim[0]
    else:
        Crad = np.asarray([cradlim] * ncen)
    Crad = Crad / 2.0  # legacy behavior: input radius corresponds to 2*sigma

    # Set number of members per cluster.
    if fix_nmemb:
        k = int(1.0 * nnotunif / ncen)
        Cmem = [k] * ncen
    else:
        uu = np.random.randint(4, nnotunif + 1, ncen - 1)
        tt = np.append(uu, [4, nnotunif + 1])
        tt.sort()
        Cmem = np.diff(tt)

    print("Generating clusters...")
    abc = None
    for i in range(ncen):
        print("  |--", i, Cra[i], Cdec[i], Cred[i], Crad[i], Cmem[i])
        mm = np.array([Cx[i], Cy[i], Cz[i]])

        if rand_elong:
            efac = np.random.random(3) * 3.0
        else:
            efac = np.ones(3)
        stdx = Crad[i] * efac[0]
        stdy = Crad[i] * efac[1]
        stdz = Crad[i] * efac[2]

        # Preserved from original implementation. NumPy expects a covariance
        # matrix here, so a more statistically literal implementation would use
        # std**2 on the diagonal. That change is intentionally not applied here
        # to avoid altering legacy behaviour silently.
        cv = np.array([[stdx, 0.0, 0.0], [0.0, stdy, 0.0], [0.0, 0.0, stdz]])

        nmem = Cmem[i]
        if nmem > 0:
            m = np.random.multivariate_normal(mm, cv, nmem)
            abc = m if abc is None else np.concatenate((abc, m), axis=0)

    if abc is None:
        abc = np.empty((0, 3), dtype=float)

    x1, y1, z1 = abc[:, 0], abc[:, 1], abc[:, 2]

    # Convert clusters from (x, y, z) to (ra, dec, redshift).
    ttt = cartesian_to_spherical(x1, y1, z1)
    comd1 = ttt[0].value
    dec1 = np.rad2deg(ttt[1].value)
    ra1 = np.rad2deg(ttt[2].value)

    tmpz = np.linspace(0.0, zlim[1] + 7.0, 500)
    tmpd = cosmo.comoving_distance(tmpz).value
    reds1 = np.interp(comd1, tmpd, tmpz, left=-1.0, right=-1.0)
    if (reds1 < 0).any():
        raise RuntimeError("Problem with redshifts!")

    # Cut clusters to desired window.
    idx, = np.where(
        (ra1 > ralim[0])
        & (ra1 < ralim[1])
        & (dec1 > declim[0])
        & (dec1 < declim[1])
        & (reds1 > zlim[0])
        & (reds1 < zlim[1])
    )
    ra1, dec1, reds1 = ra1[idx], dec1[idx], reds1[idx]
    ncpop = len(ra1)

    # Add uniform background points.
    nback = n - ncen - ncpop
    ra2, dec2 = uniform_sky(ralim, declim, n=nback)
    reds2 = (zlim[1] - zlim[0]) * np.random.random(nback) + zlim[0]
    x2, y2, z2 = rdz2xyz(ra2, dec2, reds2, cosmo)

    print("Results")
    print("  |--Nr of clusters              :", ncen)
    print("  |--Nr of cluster members       :", ncpop)
    print("  |--Nr of non-cluster objects   :", nback)
    print("  |--Total nr of objects in cone :", n)

    # Join everything and optionally plot.
    ra = np.concatenate([Cra, ra1, ra2])
    dec = np.concatenate([Cdec, dec1, dec2])
    reds = np.concatenate([Cred, reds1, reds2])
    distC = cosmo.comoving_distance(Cred).value
    dist1 = cosmo.comoving_distance(reds1).value
    dist2 = cosmo.comoving_distance(reds2).value
    dist = np.concatenate([distC, dist1, dist2])
    x = np.concatenate([Cx, x1, x2])
    y = np.concatenate([Cy, y1, y2])
    z = np.concatenate([Cz, z1, z2])

    if colorize:
        c2, c1, cc = "k", "b", "r"
        spts, scen = 0.2, 20
    else:
        c2, c1, cc = "k", "k", "k"
        spts, scen = 0.2, 0.2

    if doplot:
        x2p, y2p = raz2xy(ra2, reds2)
        x1p, y1p = raz2xy(ra1, reds1)
        xCp, yCp = raz2xy(Cra, Cred)
        plt.scatter(x2p, y2p, s=spts, color=c2)
        plt.scatter(x1p, y1p, s=spts, color=c1)
        plt.scatter(xCp, yCp, s=scen, color=cc)

        plt.figure()
        plt.scatter(ra2, dec2, s=spts, color=c2)
        plt.scatter(ra1, dec1, s=spts, color=c1)
        plt.scatter(Cra, Cdec, s=scen, color=cc)
        plt.axis("equal")

    kone = _stack_columns(ra, dec, reds, dist, x, y, z)
    if oformat == "table":
        cols = ["ra", "dec", "z", "comd", "px", "py", "pz"]
        kone = Table(data=kone, names=cols)

    return kone



def filament_box(npts=80000, nvoids=2000, nstep=100, rmin=None, rep=None, b=150.0):
    """
    Generate random 3D points emulating filamentary structure in a box.
    """
    print("-------------------------------------------------------------")
    print("Building cube")

    if rep is None:
        rep = [1, 1, 1]
    b = 1.0 * b

    ptsa = np.zeros((1, 3), dtype=float)
    for i in range(rep[0]):
        for j in range(rep[1]):
            for k in range(rep[2]):
                voidpts = uniform_sphere(nvoids, 1)
                pts = uniform_sphere(npts, 1)
                tree = cKDTree(voidpts)

                for _ in range(nstep):
                    idx = _tree_query_indices(tree, pts)
                    nv = voidpts[idx]
                    pts = 0.9975 * (pts + 0.01 * (pts - nv))

                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
                idx, = np.where(
                    (x > -0.5) & (x < 0.5) &
                    (y > -0.5) & (y < 0.5) &
                    (z > -0.5) & (z < 0.5)
                )
                pts = pts[idx] + [0.5, 0.5, 0.5]
                pts = pts + [1.0 * i, 1.0 * j, 1.0 * k]
                ptsa = np.vstack([ptsa, pts])

    ptsa = ptsa * [b, b, b]

    if rmin is not None:
        tree = cKDTree(ptsa)
        cp = tree.query_pairs(rmin)
        npairs = len(cp)
        if npairs > 0:
            cp = np.asarray(list(cp))
            todel = cp[:, 1]
            ptsa = np.delete(ptsa, todel, axis=0)
        print("  |-- Pairs below rmin :", npairs)

    print("  |-- Total objects in cube :", len(ptsa))
    print("-------------------------------------------------------------")
    return ptsa[1:, :]



def fill_cone(
    ralim,
    declim,
    zlim,
    xyz=None,
    b=None,
    repmethod="rotation",
    npts=80000,
    nvoids=2000,
    nstep=100,
    rmin=None,
    cosmo=None,
):
    """
    Fill the volume of an observation light cone with small cubes of 3D points.
    """
    print("-------------------------------------------------------------")
    print("Filling light cone with cubes")

    if b is None:
        raise ValueError("Parameter 'b' must be provided.")
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

    torad = np.pi / 180.0
    ra0, ra1 = ralim[0] * torad, ralim[1] * torad
    dec0, dec1 = declim[0] * torad, declim[1] * torad
    d0 = cosmo.comoving_distance(zlim[0]).value
    d1 = cosmo.comoving_distance(zlim[1]).value

    d_max = np.floor(1 + d1 / b) * b
    xyza = np.zeros((1, 3), dtype=float)

    nb = 0
    for x0 in np.arange(-1 * d_max, d_max, b):
        for y0 in np.arange(-1 * d_max, d_max, b):
            for z0 in np.arange(-1 * d_max, d_max, b):
                x1 = x0 + b
                y1 = y0 + b
                z1 = z0 + b

                if (x0 >= 0) and (y0 >= 0):
                    tra0 = np.arctan2(y0, x1)
                    tra1 = np.arctan2(y1, x0)
                    if z0 >= 0:
                        tdec0 = np.arctan2(z0, np.sqrt(x1**2 + y1**2))
                        tdec1 = np.arctan2(z1, np.sqrt(x0**2 + y0**2))
                    else:
                        tdec0 = np.arctan2(z0, np.sqrt(x0**2 + y0**2))
                        tdec1 = np.arctan2(z1, np.sqrt(x1**2 + y1**2))

                if (x0 < 0) and (y0 >= 0):
                    tra0 = np.arctan2(y1, x1)
                    tra1 = np.arctan2(y0, x0)
                    if z0 >= 0:
                        tdec0 = np.arctan2(z0, np.sqrt(x0**2 + y1**2))
                        tdec1 = np.arctan2(z1, np.sqrt(x1**2 + y0**2))
                    else:
                        tdec0 = np.arctan2(z0, np.sqrt(x1**2 + y0**2))
                        tdec1 = np.arctan2(z1, np.sqrt(x0**2 + y1**2))

                if (x0 < 0) and (y0 < 0):
                    tra0 = np.arctan2(-1 * y1, -1 * x0) + np.pi
                    tra1 = np.arctan2(-1 * y0, -1 * x1) + np.pi
                    if z0 >= 0:
                        tdec0 = np.arctan2(z0, np.sqrt(x0**2 + y0**2))
                        tdec1 = np.arctan2(z1, np.sqrt(x1**2 + y1**2))
                    else:
                        tdec0 = np.arctan2(z0, np.sqrt(x1**2 + y1**2))
                        tdec1 = np.arctan2(z1, np.sqrt(x0**2 + y0**2))

                if (x0 >= 0) and (y0 < 0):
                    tra0 = np.arctan2(-1 * y0, -1 * x0) + np.pi
                    tra1 = np.arctan2(-1 * y1, -1 * x1) + np.pi
                    if z0 >= 0:
                        tdec0 = np.arctan2(z0, np.sqrt(x1**2 + y0**2))
                        tdec1 = np.arctan2(z1, np.sqrt(x0**2 + y1**2))
                    else:
                        tdec0 = np.arctan2(z0, np.sqrt(x0**2 + y1**2))
                        tdec1 = np.arctan2(z1, np.sqrt(x1**2 + y0**2))

                x00 = x0 if x0 >= 0 else x1
                y00 = y0 if y0 >= 0 else y1
                z00 = z0 if z0 >= 0 else z1
                x11 = x1 if x0 >= 0 else x0
                y11 = y1 if y0 >= 0 else y0
                z11 = z1 if z0 >= 0 else z0

                td0 = np.sqrt(x00**2 + y00**2 + z00**2)
                td1 = np.sqrt(x11**2 + y11**2 + z11**2)

                if (
                    (td1 > d0)
                    and (td0 < d1)
                    and (tra0 < ra1)
                    and (tra1 > ra0)
                    and (tdec0 < dec1)
                    and (tdec1 > dec0)
                ):
                    print("  Cube", nb, "inside cone")
                    print("    |--- ", "(x0,y0,z0,d0) = ", (x00, y00, z00, td0))
                    print("    |--- ", "(x1,y1,z1,d1) = ", (x11, y11, z11, td1))

                    if repmethod == "copy":
                        xyz_use = xyz
                    elif repmethod == "rotation":
                        xyz_use = randomize(xyz, b)
                    elif repmethod == "fullrandom":
                        xyz_use = filament_box(
                            npts=npts,
                            nvoids=nvoids,
                            nstep=nstep,
                            rmin=rmin,
                            b=b,
                        )
                    else:
                        raise ValueError(
                            "repmethod must be one of {'copy', 'rotation', 'fullrandom'}."
                        )

                    xyza = np.vstack([xyza, xyz_use + [x0, y0, z0]])
                    nb += 1

    if rmin is not None:
        tree = cKDTree(xyza)
        cp = tree.query_pairs(rmin)
        npairs = len(cp)
        if npairs > 0:
            cp = np.asarray(list(cp))
            todel = cp[:, 1]
            xyza = np.delete(xyza, todel, axis=0)
        print("    |-- Pairs below rmin (along edges of cubes):", npairs)

    print("    |-- Nr of intersecting cubes :", nb)
    print("    |-- Objects inside all intersecting cubes :", len(xyza))
    print("-------------------------------------------------------------")
    return xyza[1:, :]



def mcone_filam(
    ralim,
    declim,
    zlim,
    npts=80000,
    nvoids=2000,
    nstep=100,
    rmin=None,
    b=150.0,
    repmethod="rotation",
    cosmo=None,
    oformat="table",
):
    """
    Generate a mock light cone with filamentary structure.
    """
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

    d0 = cosmo.comoving_distance(zlim[0]).value
    d1 = cosmo.comoving_distance(zlim[1]).value

    xyz = filament_box(npts=npts, nvoids=nvoids, nstep=nstep, rmin=rmin, b=b)
    xyza = fill_cone(
        ralim,
        declim,
        zlim,
        xyz=xyz,
        b=b,
        repmethod=repmethod,
        npts=npts,
        nvoids=nvoids,
        nstep=nstep,
        rmin=rmin,
        cosmo=cosmo,
    )

    x, y, z = xyza[:, 0], xyza[:, 1], xyza[:, 2]
    ttt = cartesian_to_spherical(x, y, z)
    comd = ttt[0].value
    dec = np.rad2deg(ttt[1].value)
    ra = np.rad2deg(ttt[2].value)

    idx, = np.where(
        (ra > ralim[0])
        & (ra < ralim[1])
        & (dec > declim[0])
        & (dec < declim[1])
        & (comd > d0)
        & (comd < d1)
    )
    nobj = len(idx)
    print("Results")
    print("  |-- Prunned objects outside intersecting cubes :", len(ra) - nobj)
    print("  |-- Total points inside light cone :", nobj)
    print("-------------------------------------------------------------")

    tmpz = np.linspace(0.0, zlim[1] + 7.0, 500)
    tmpd = cosmo.comoving_distance(tmpz).value
    redsh = np.interp(comd[idx], tmpd, tmpz, left=-1.0, right=-1.0)
    if (redsh < 0).any():
        raise RuntimeError("Problem with redshifts!")

    kone = _stack_columns(ra[idx], dec[idx], redsh, comd[idx], x[idx], y[idx], z[idx])
    if oformat == "table":
        cols = ["ra", "dec", "z", "comd", "px", "py", "pz"]
        kone = Table(data=kone, names=cols)

    return kone


def add_mock_weights(
    kone,
    wmin=1.0,
    wmax=5.0,
    nneigh=10,
    clip_percentile=(5.0, 95.0),
    density_column="mock_density",
    weight_column="weight",
    copy=True,
):
    """
    Add a mock weight column based on the local angular density on the sky.

    The function estimates a local projected density for each object from the
    angular distance to its ``nneigh``-th nearest neighbour on the celestial
    sphere. Objects in denser regions receive larger weights, while objects in
    sparse regions receive smaller weights. The final weights are linearly
    rescaled to the interval ``[wmin, wmax]`` after robust percentile clipping.

    This is intended as a simple mock of a real observational weighting scheme.
    It is useful for testing weighted correlation-function pipelines, but it is
    not meant to represent any specific survey selection effect.

    Parameters
    ----------
    kone : astropy.table.Table or numpy.ndarray
        Output of :func:`mcone_filam` or any compatible object containing at
        least ``ra`` and ``dec``. If a NumPy array is supplied, the first two
        columns are assumed to be ``ra`` and ``dec`` in degrees.
    wmin : float, optional
        Minimum output weight. Default is 1.0.
    wmax : float, optional
        Maximum output weight. Default is 5.0.
    nneigh : int, optional
        Nearest-neighbour index used to estimate the local angular density.
        Larger values produce smoother weights. Default is 10.
    clip_percentile : tuple of 2 floats, optional
        Lower and upper percentiles used to clip the density field before
        mapping it into ``[wmin, wmax]``. This prevents a few very dense or
        very sparse regions from dominating the weight scale. Set to ``None``
        to disable clipping. Default is ``(5.0, 95.0)``.
    density_column : str, optional
        Name of the density column added when ``kone`` is an Astropy table.
        Default is ``"mock_density"``.
    weight_column : str, optional
        Name of the weight column added when ``kone`` is an Astropy table.
        Default is ``"weight"``.
    copy : bool, optional
        If ``True``, return a modified copy of the input. If ``False`` and the
        input is an Astropy table, the table is modified in place. For NumPy
        arrays, ``copy=False`` is ignored and a new array is always returned.

    Returns
    -------
    astropy.table.Table or numpy.ndarray
        Object of the same general type as the input, with an added weight
        column. For Astropy tables, both the local density proxy and the weight
        column are added. For NumPy arrays, a new last column containing the
        weights is appended.

    Notes
    -----
    The local density proxy is computed as

    ``density ~ nneigh / (pi * theta_n^2)``

    where ``theta_n`` is the angular distance, in radians, to the
    ``nneigh``-th nearest neighbour. Since only the relative density matters
    for the final rescaling, the normalization of this proxy is arbitrary.

    Examples
    --------
    >>> kone = mcone_filam([0, 20], [0, 20], [0.01, 0.1])
    >>> kone_w = add_mock_weights(kone, wmin=1.0, wmax=5.0, nneigh=12)
    >>> kone_w.colnames[-2:]
    ['mock_density', 'weight']
    """
    if wmax < wmin:
        raise ValueError("wmax must be greater than or equal to wmin.")
    if nneigh < 1:
        raise ValueError("nneigh must be >= 1.")

    is_table = isinstance(kone, Table)
    if is_table:
        if "ra" not in kone.colnames or "dec" not in kone.colnames:
            raise ValueError("Input table must contain 'ra' and 'dec' columns.")
        out = kone.copy(copy_data=True) if copy else kone
        ra = np.asarray(out["ra"], dtype=float)
        dec = np.asarray(out["dec"], dtype=float)
    else:
        arr = np.asarray(kone, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                "Input array must be 2D with at least two columns: ra and dec."
            )
        out = arr.copy()
        ra = out[:, 0]
        dec = out[:, 1]

    nobj = len(ra)
    if nobj == 0:
        if is_table:
            out[density_column] = np.array([], dtype=float)
            out[weight_column] = np.array([], dtype=float)
            return out
        return np.column_stack([out, np.array([], dtype=float)])

    if nobj == 1:
        density = np.array([1.0], dtype=float)
        weights = np.array([0.5 * (wmin + wmax)], dtype=float)
    else:
        kquery = min(nneigh + 1, nobj)
        ux, uy, uz = ra_dec_to_xyz(ra, dec)
        unit_xyz = np.column_stack([ux, uy, uz])
        tree = cKDTree(unit_xyz)
        dists = tree.query(unit_xyz, k=kquery)[0]

        if kquery == 2:
            chord = np.asarray(dists[:, 1], dtype=float)
            nk = 1
        else:
            chord = np.asarray(dists[:, -1], dtype=float)
            nk = kquery - 1

        chord = np.clip(chord, 0.0, 2.0)
        theta = 2.0 * np.arcsin(0.5 * chord)
        theta = np.maximum(theta, 1.0e-12)
        area = np.pi * theta**2
        density = nk / area

        density_scaled = density.copy()
        if clip_percentile is not None:
            plo, phi = clip_percentile
            dlo, dhi = np.percentile(density_scaled, [plo, phi])
            density_scaled = np.clip(density_scaled, dlo, dhi)

        dmin = density_scaled.min()
        dmax = density_scaled.max()
        if dmax > dmin:
            frac = (density_scaled - dmin) / (dmax - dmin)
        else:
            frac = np.zeros_like(density_scaled)
        weights = wmin + frac * (wmax - wmin)

    if is_table:
        out[density_column] = density
        out[weight_column] = weights
        return out

    return np.column_stack([out, weights])


def make_random_sphere(N: int, seed: int = 0, ralim=None, declim=None):
    """
    Generate N random (ra, dec) points uniformly on the sphere, optionally
    restricted to a RA/Dec rectangle.

    Parameters
    ----------
    N : int
        Number of points.
    seed : int
        RNG seed.
    ralim : sequence of 2 floats, optional
        Right ascention boundaries in degrees. If None then stars are within [0,360]deg
    declim : sequence of 2 floats, optional
        Declination boundaries in degrees. If None then stars are within [-90,90]deg

    Returns
    -------
    ra, dec : np.ndarray
        Arrays of shape (N,) in degrees. ra in [0, 360), dec in [-90, 90].
    """
    rng = np.random.default_rng(seed)

    # RA sampling (uniform in angle)
    if ralim is None:
        ra = rng.uniform(0.0, 360.0, N)
    else:
        ra1, ra2 = float(ralim[0]), float(ralim[1])
        ra1 = ra1 % 360.0
        ra2 = ra2 % 360.0

        # If ra2 >= ra1: simple interval [ra1, ra2]
        # If ra2 <  ra1: wrap interval [ra1, 360) U [0, ra2]
        span = (ra2 - ra1) % 360.0  # in [0, 360)
        if span == 0.0:
            # Ambiguous: could mean full circle or empty. Here we treat as full circle.
            ra = rng.uniform(0.0, 360.0, N)
        else:
            ra = (ra1 + rng.uniform(0.0, span, N)) % 360.0

    # Dec sampling (uniform in sin(dec))
    if declim is None:
        u = rng.uniform(-1.0, 1.0, N)
    else:
        dec1, dec2 = float(declim[0]), float(declim[1])
        # allow to pass in any order
        dmin, dmax = (dec1, dec2) if dec1 <= dec2 else (dec2, dec1)

        # clip to physical range
        dmin = max(-90.0, dmin)
        dmax = min( 90.0, dmax)
        if dmax < dmin:
            raise ValueError("declim does not overlap [-90, 90] after clipping.")

        umin = np.sin(np.deg2rad(dmin))
        umax = np.sin(np.deg2rad(dmax))
        u = rng.uniform(umin, umax, N)

    dec = np.rad2deg(np.arcsin(u))

    return ra.astype(np.float64), dec.astype(np.float64)



if __name__ == "__main__":
    testsuite = "gaussblobs1"
    # testsuite = "fil1"

    if testsuite == "gaussblobs1":
        cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
        ralim = [10, 50]
        declim = [20, 40]
        zlim = [0.01, 0.15]
        cradlim = [0.5, 20]
        kone = mcone_gaussblobs(
            ralim,
            declim,
            zlim,
            n=20000,
            ufrac=0.7,
            ncen=500,
            cradlim=cradlim,
            rand_elong=False,
            random_state=222,
            doplot=False,
            colorize=True,
            cosmo=cosmo,
            oformat="table",
        )
        send(kone, "kone", cols=["ra", "dec", "z", "comd", "px", "py", "pz"])

    if testsuite == "fil1":
        cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
        ralim = [0, 60]
        declim = [0, 60]
        zlim = [0.01, 0.15]

        kone = mcone_filam(
            ralim,
            declim,
            zlim,
            npts=80000,
            nvoids=2000,
            nstep=100,
            rmin=None,
            b=150.0,
            repmethod="rotation",
            cosmo=cosmo,
            oformat="table",
        )
        send(kone, "kone", cols=["ra", "dec", "z", "comd", "px", "py", "pz"])
