"""
Atomic magnetic form factors for Magnetic-PyMS.

Implements the per-element atomic magnetic form factor a(k) that enters the
reciprocal-space construction of the projected magnetic vector potential,
following the "quasi-dipole" parametrization of

    P. M. Lyon and J. Rusz, "Parameterization of magnetic vector potentials
    and fields for efficient multislice calculations of electron scattering
    in magnetic materials," Ultramicroscopy 231, 113315 (2021).


Physical background
--------------------
Lyon & Rusz give, for a single atom with magnetic moment along unit vector
m_hat, a "quasi-dipole" real-space model of that atom's contribution to the
(periodic part of the) magnetic vector potential::

    A(r) = (m_hat x r) * f(r),      f(r) = sum_{i=0}^{4} a_i / (r**n_i + b_i)

with n_i = i/2 + 3 (their Table 1 gives fitted a_i, b_i per element,
normalized to a moment of 1 Bohr magneton). The b_i > 0 keep f(r) finite as
r -> 0.

The quantity this module computes, a(k), is defined so that the reciprocal-
space, in-plane-projected z-component of the vector potential sourced by an
areal magnetization density (m_x(r_xy), m_y(r_xy)) built from these atoms is

    A_z,p(k) = i * (kx * my(k) - ky * mx(k)) * a(k) / k        (ordinary k)

exactly mirroring how the validated Fortran reference (ak_mod.f90 + cms.f90)
combines it downstream, and structurally identical -- up to the a(k)/k vs
1/k**2 bookkeeping, see below -- to Lyon & Rusz's own closed reciprocal-space
result (their Eq. 10): Ap(k) = i*mu_0*(k x m(k)) / k**2.

Numerical implementation
-------------------------
a(k) is obtained by Hankel-transforming f(r) directly, rather than by
reusing the Fortran reference's own precomputed `ak_hankeltransform_z*.dat`
tables (see feature-01 Sec. 2.4 for why: the published Table 1 coefficients
are reproducible and independently citable; the Fortran's tables are neither
published nor available to this project). The exact transform relating f(r)
to a(k) is not given explicitly in Lyon & Rusz (2021) and was derived here
from first principles, starting from the standard result that multiplying a
real-space function by position is a k-gradient in reciprocal space::

    a(k) = 4*pi * d/dk[ I0(k) ],   I0(k) = integral_0^inf r**2 f(r) j0(k r) dr

(I0 is the ordinary 3-D Fourier transform of the radially symmetric scalar
f(r), consistent with the exp(-i k . r) convention; F[(m_hat x r) f(r)](k)
works out to i * d/dk[I0(k)] * (m_hat x k_hat), and dividing by k to project
onto the k=(kx,ky,0) plane and taking the z-component of that cross product
gives the a(k)/k form above). Using the Bessel relation j0'(x) = -j1(x), this
reduces to a single ordinary Hankel transform of order 3/2, which the
``hankel`` package (already a PyMS dependency -- see Ionization.py) can
evaluate directly::

    a(k) = 4*pi*sqrt(pi/(2*k)) * HankelTransform(nu=1.5).transform(g, k)
    g(r) = r**1.5 * f(r)

This derivation was validated numerically (not merely asserted) against
ak_mod.f90's own closed-form small-k asymptotic series -- itself transcribed
verbatim from the Fortran source and independent of this derivation -- for
Sc (Z=21), Fe (Z=26) and Ni (Z=28): agreement better than 0.5% throughout the
regime k*sqrt(b0) << 1 where that asymptotic series is expected to hold. See
``test_matches_fortran_small_k_asymptote`` in
tests/test_magnetic_form_factors.py for the regression test that reproduces
this check. The Hankel transform quadrature becomes numerically unreliable
(confirmed by comparing two independent quadrature resolutions and observing
disagreement, including sign flips, rather than smooth convergence) for
angular wavenumbers roughly above 15-20 rad/Angstrom. This is not a
practical limitation: by that point the true a(k) has decayed (as any
physical, finite-extent atomic form factor must) to many orders of magnitude
below any numerically meaningful contribution, so ``AtomicMagneticFormFactor``
clips a(k) to exactly zero beyond the point where the two quadrature
resolutions stop agreeing, rather than propagate quadrature noise. This is a
narrower validated range than the Fortran's own three-branch (small-k
series / spline / large-k series) scheme, which was fit to cover k -> infinity;
the difference is not expected to matter in practice since both
representations are effectively zero well before typical multislice
bandwidth limits (2/3 of the grid Nyquist frequency) are reached, but it is
recorded here as an honest, deliberate scope difference from the reference
implementation rather than a silent one.

Note on k conventions (see architecture-analysis.md Part D)
-------------------------------------------------------------
The Fortran's tables (and the closed-form asymptotic series above) are
expressed in terms of an ANGULAR wavenumber ``kappa = 2*pi*k``, consistent
with a Fourier convention ``exp(-i kappa . r)``. PyMS's own k-grids (e.g.
``structure_routines.calculate_scattering_factors``, via
``utils.numpy_utils.q_space_array``) are in the numpy ``fftfreq`` convention
(ORDINARY frequency, cycles per unit length, no factor of 2*pi).
``AtomicMagneticFormFactor.__call__`` takes ORDINARY k (the PyMS convention)
and performs the ``kappa = 2*pi*k`` conversion internally, so callers never
need to handle it explicitly -- unlike the Fortran, where every call site is
individually responsible for remembering to write ``ak_interp(2*pi*knorm)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

try:
    from hankel import HankelTransform
except ImportError as _e:  # pragma: no cover - import-time environment check
    raise ImportError(
        "magnetic_form_factors requires the 'hankel' package (already a "
        "PyMS dependency via Ionization.py). Install with `pip install hankel`."
    ) from _e


# ---------------------------------------------------------------------------
# Lyon & Rusz (2021) Table 1 quasi-dipole fit coefficients, transcribed
# verbatim from the validated Fortran reference's ak_mod.f90::init_mparams
# (independently spot-checked against the published table for Sc during
#
# Keys are atomic number Z. Values are
# (a0, b0, a1, b1, a2, b2, a3, b3, a4, b4).
# ---------------------------------------------------------------------------
_MPARAMS: dict[int, tuple[float, ...]] = {
    21: (0.96270000, 1.29800000, -0.01195000, 0.00188700, 0.00320700, 0.00072230, 0.17460000, 0.26240000, 0.02062000, 0.03740000),  # Sc
    22: (0.98650000, 2.00000000, -0.00256500, 0.00144200, 0.39060000, 0.25280000, -0.52380000, 6.22800000, -0.17310000, 0.38690000),  # Ti
    23: (6.59400000, 463.80000000, 2.00700000, 0.47610000, -41.02000000, 3962.00000000, -1.16500000, 2.72000000, -0.77930000, 0.49110000),  # V
    24: (1.20100000, 4.65500000, 1.30400000, 0.27830000, -5.86700000, 1.04500000, 3.57700000, 5.46200000, 3.88000000, 1.31200000),  # Cr
    25: (0.81820000, 4.15600000, 2.39000000, 0.26630000, -6.85800000, 3.05500000, 6.22200000, 4.96000000, -0.75400000, 0.24500000),  # Mn
    26: (6.96700000, 435.70000000, 1.79900000, 0.18940000, -44.85000000, 3955.00000000, -0.66650000, 1.29400000, -0.51910000, 0.16930000),  # Fe
    27: (5.32400000, 271.60000000, 1.89400000, 0.16380000, -31.58000000, 2386.00000000, -0.82530000, 0.95670000, -0.52110000, 0.12850000),  # Co
    28: (2.12300000, 0.21090000, 5.42300000, 6.53500000, -42.32000000, 0.95410000, 70.10000000, 1.13900000, -30.31000000, 1.32500000),  # Ni
    29: (1.68400000, 0.01976000, -1.09300000, 0.05604000, -0.07235000, 0.00205800, -0.11530000, 0.00449900, -0.44340000, 0.57740000),  # Cu
    39: (1.80800000, 0.00182400, -0.58160000, 0.09904000, -1.82400000, 0.00030880, 1.04500000, 0.00013210, -0.17140000, 0.00005709),  # Y
    40: (0.00295400, 0.00035820, 1.25100000, 20.98000000, 0.96150000, 5.49600000, -0.10080000, 0.24810000, 0.19460000, 0.57400000),  # Zr
    41: (0.98500000, 0.06480000, 0.33540000, 3.14600000, -0.31610000, 0.02979000, -1.07100000, 0.26020000, 0.66190000, 0.29390000),  # Nb
    42: (1.15000000, 0.70320000, 0.20980000, 0.01704000, -1.33100000, 0.03853000, 0.70770000, 0.03072000, 0.18150000, 0.17240000),  # Mo
    44: (1.14100000, 0.75940000, 0.25080000, 0.01774000, -1.32900000, 0.04251000, 0.58500000, 0.16240000, 0.38490000, 0.02449000),  # Ru
    45: (0.02283000, 0.00162500, 2.76900000, 1.72400000, -2.18700000, 2.57300000, -0.14440000, 0.04847000, 0.14020000, 0.06810000),  # Rh
    46: (0.00762300, 0.00040480, -3.55800000, 96.47000000, 9.94200000, 169.10000000, 2.20300000, 11.02000000, 0.23910000, 0.49860000),  # Pd
    47: (4.34900000, 39.61000000, -8.31200000, 153.70000000, 0.46110000, 0.26750000, -0.61890000, 0.34810000, 2.32000000, 21.11000000),  # Ag
    72: (0.86020000, 1.20100000, 0.26970000, 0.06587000, -0.19460000, 1.34500000, -0.17150000, 0.05218000, -0.11490000, 0.33840000),  # Hf
    73: (0.97820000, 1.50200000, 0.21600000, 0.05528000, -0.67570000, 0.11350000, 0.39000000, 1.21000000, 0.25660000, 0.12250000),  # Ta
    74: (0.87050000, 1.17100000, 0.95430000, 0.06933000, -1.54600000, 0.07871000, 0.27100000, 0.23820000, 0.31970000, 0.05784000),  # W
    75: (2.07000000, 669.70000000, 1.51800000, 0.09766000, 0.49920000, 13.97000000, -0.56300000, 0.04738000, -0.42180000, 0.23850000),  # Re
    76: (1.04200000, 1.46900000, 0.34990000, 0.05149000, -1.51400000, 0.11530000, 1.08600000, 0.94170000, 0.70260000, 0.11420000),  # Os
    77: (0.02514000, 0.02202000, 11.39000000, 101.10000000, -21.54000000, 299.20000000, 2.05000000, 6.56200000, -0.02924000, 0.04927000),  # Ir
    78: (2.00100000, 0.19250000, -3.30500000, 0.20580000, 1.18500000, 8.15800000, 3.71600000, 0.28570000, -2.08800000, 0.31530000),  # Pt
    79: (0.61000000, 1.74700000, 1.06900000, 2.51100000, 0.62580000, 0.27640000, -3.43300000, 0.63400000, 2.10200000, 0.77600000),  # Au
}

_ELEMENT_SYMBOLS: dict[int, str] = {
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au",
}


def has_magnetic_data(Z: int) -> bool:
    """
    Return True if element Z has a tabulated atomic magnetic form factor.

    This is the single source of truth for "is this element treated as
    magnetic" in Magnetic-PyMS, replacing the Fortran reference's two
    independent (and not necessarily consistent) gates: ``atoms.f90``'s
    hardcoded ``is_magnetic(Z)`` transition-metal range check, used only by
    its dead real-space code path, and ``ak_mod.f90``'s data-file-existence
    check, used by the active path. 

    Parameters
    ----------
    Z : int
        Atomic number.
    """
    return int(Z) in _MPARAMS


def magnetic_element_symbols() -> tuple[str, ...]:
    """Return the chemical symbols of all elements with magnetic form factor data."""
    return tuple(_ELEMENT_SYMBOLS[Z] for Z in sorted(_MPARAMS))


def _quasi_dipole_radial_profile(r: np.ndarray, p: tuple[float, ...]) -> np.ndarray:
    """
    Evaluate the Lyon & Rusz (2021) quasi-dipole radial shape function.

    f(r) = sum_{i=0}^{4} a_i / (r**n_i + b_i),  n_i = i/2 + 3

    Parameters
    ----------
    r : array_like
        Radial distance(s) in Angstrom (r >= 0).
    p : tuple of float
        (a0, b0, a1, b1, a2, b2, a3, b3, a4, b4) for one element.
    """
    r = np.asarray(r, dtype=np.float64)
    total = np.zeros_like(r)
    for i in range(5):
        a, b = p[2 * i], p[2 * i + 1]
        n = i / 2.0 + 3.0
        total = total + a / (np.power(r, n) + b)
    return total


def _hankel_a_of_kappa(
    kappa: np.ndarray, p: tuple[float, ...], N: int, h: float
) -> np.ndarray:
    """
    Evaluate a(kappa) at ANGULAR wavenumber(s) kappa via the order-3/2
    Hankel transform derived in the module docstring:

        a(kappa) = 4*pi*sqrt(pi/(2*kappa)) *
                   HankelTransform(nu=1.5).transform(g, kappa)
        g(r) = r**1.5 * f(r)

    Parameters
    ----------
    kappa : array_like
        Angular wavenumber(s), kappa > 0 (rad / Angstrom).
    p : tuple of float
        Quasi-dipole fit coefficients for one element (see _MPARAMS).
    N, h : int, float
        Quadrature resolution parameters passed to
        ``hankel.HankelTransform`` (number of nodes, node spacing).
    """

    def g(r: np.ndarray) -> np.ndarray:
        return np.power(r, 1.5) * _quasi_dipole_radial_profile(r, p)

    ht = HankelTransform(nu=1.5, N=N, h=h)
    kappa = np.atleast_1d(np.asarray(kappa, dtype=np.float64))
    transform = ht.transform(g, kappa, ret_err=False)
    return 4.0 * np.pi * np.sqrt(np.pi / (2.0 * kappa)) * transform


@dataclass
class AtomicMagneticFormFactor:
    """
    Per-element atomic magnetic form factor a(k), evaluated on demand.

    a(k) is precomputed once, on a radial grid of ANGULAR wavenumbers, via a
    validated Hankel transform of the Lyon & Rusz (2021) quasi-dipole
    real-space profile (see module docstring), then cheaply evaluated at
    arbitrary ORDINARY wavenumbers k -- the PyMS grid convention, see
    ``structure_routines.calculate_scattering_factors`` /
    ``utils.numpy_utils.q_space_array`` -- via cubic-spline interpolation.
    This mirrors the structure of the Fortran reference's own
    tabulate-then-interpolate approach (``ak_mod.f90``), but with the table
    computed fresh from published data rather than reused from an
    unpublished file, and with a single interpolation scheme (no separate
    small-k / large-k analytic branches are needed: the Hankel transform
    used here is accurate over the full range where a(k) is not already
    numerically negligible -- see module docstring).

    Parameters
    ----------
    Z : int
        Atomic number. Must satisfy ``has_magnetic_data(Z)``.
    kappa_min, kappa_max : float, optional
        Angular-wavenumber (rad/Angstrom) range over which the radial table
        is built. ``kappa_max`` should comfortably exceed
        ``2*pi*(the largest ordinary |k| any grid this is used with will
        need)``; a(k) is returned as exactly zero beyond the table's
        validated range (``kappa_max_reliable``), which is typically well
        inside [kappa_min, kappa_max] -- see ``_build_table``.
    n_table : int
        Number of (log-spaced) points in the radial precompute table.

    Attributes
    ----------
    kappa_max_reliable : float
        The largest angular wavenumber at which the underlying Hankel
        transform was found to be numerically converged (two independent
        quadrature resolutions agreeing to better than 1%). Set by
        ``_build_table``; exposed for diagnostics and testing.
    """

    Z: int
    kappa_min: float = 1e-6
    kappa_max: float = 60.0
    n_table: int = 400
    _p: tuple[float, ...] = field(init=False, repr=False)
    _kappa_table: np.ndarray = field(init=False, repr=False)
    _a_table: np.ndarray = field(init=False, repr=False)
    _spline: CubicSpline = field(init=False, repr=False)
    kappa_max_reliable: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if not has_magnetic_data(self.Z):
            raise ValueError(
                f"Z={self.Z} has no tabulated magnetic form factor "
                f"(available elements: {magnetic_element_symbols()})"
            )
        self._p = _MPARAMS[int(self.Z)]
        self._build_table()

    def _build_table(self) -> None:
        kappa = np.geomspace(self.kappa_min, self.kappa_max, self.n_table)

        # Evaluate at two independent quadrature resolutions. Disagreement
        # between them is the numerical-reliability signal: the underlying
        # Ogata quadrature becomes noisy (not just imprecise -- see the
        # module docstring) once the true a(kappa) has decayed into the
        # quadrature's own error floor.
        a_coarse = _hankel_a_of_kappa(kappa, self._p, N=3200, h=5e-4)
        a_fine = _hankel_a_of_kappa(kappa, self._p, N=12800, h=1e-4)

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_disagreement = np.abs(a_fine - a_coarse) / np.maximum(
                np.abs(a_fine), 1e-300
            )
        reliable = rel_disagreement < 1e-2

        # Once unreliable, treat as unreliable from there on: a(kappa) is
        # smooth and monotonically decaying in magnitude in the regime that
        # matters here, so an isolated "reliable" point past the first
        # sustained failure is quadrature noise crossing the tolerance by
        # chance, not a real recovery of accuracy.
        first_bad = int(np.argmax(~reliable)) if np.any(~reliable) else len(kappa)
        self.kappa_max_reliable = float(kappa[first_bad - 1]) if first_bad > 0 else 0.0

        a_table = a_fine.copy()
        a_table[first_bad:] = 0.0

        self._kappa_table = kappa
        self._a_table = a_table
        self._spline = CubicSpline(kappa, a_table, extrapolate=False)

    def __call__(self, k: np.ndarray) -> np.ndarray:
        """
        Evaluate a(k) at ORDINARY wavenumber(s) k (cycles / Angstrom).

        Converts internally to the angular wavenumber ``kappa = 2*pi*k``
        used by the underlying table (see module docstring). Returns exactly
        0 at k=0 -- the Ap(k=0)=0 gauge choice, matching both Lyon & Rusz's
        Eq. 10 convention and the Fortran reference's explicit k=0 special
        case in ``init_scattering_factors`` -- and for any |k| beyond the
        table's validated range.

        Parameters
        ----------
        k : array_like
            Ordinary wavenumber(s) (cycles / Angstrom); may be negative or
            zero (only |k| matters, since a(k) depends on k through kappa=2*pi*k
            and the table covers kappa > 0).

        Returns
        -------
        numpy.ndarray
            a(k), same shape as the (broadcast) input.
        """
        k = np.asarray(k, dtype=np.float64)
        kappa = 2.0 * np.pi * np.abs(k)

        out = np.zeros_like(kappa)
        in_range = (kappa > 0) & (kappa <= self._kappa_table[-1])

        # Values below kappa_table[0] are clamped to that endpoint rather
        # than extrapolated: a(kappa) diverges as kappa -> 0 (the expected
        # long-range dipole-field tail), so there is no meaningful
        # extrapolation below the smallest tabulated point; any k this
        # close to the grid's DC term is dominated by the explicit k=0
        # gauge-fix applied downstream (see feature-01 Sec. 2.6) in any case.
        kappa_clamped = np.clip(kappa, self._kappa_table[0], self._kappa_table[-1])
        spline_vals = self._spline(kappa_clamped)
        out[in_range] = spline_vals[in_range]
        return out


@lru_cache(maxsize=None)
def load_magnetic_form_factor(Z: int) -> AtomicMagneticFormFactor:
    """
    Load (and cache) the atomic magnetic form factor for element Z.

    Repeated calls with the same Z return the same cached instance (the
    underlying Hankel-transform table is only built once per element per
    process), mirroring how ``structure_routines.calculate_scattering_factors``
    is typically computed once per simulation and reused.

    Parameters
    ----------
    Z : int
        Atomic number. Must satisfy ``has_magnetic_data(Z)``.
    """
    return AtomicMagneticFormFactor(int(Z))
