"""
physics_engine.py
=================

Numerical physics for a 1D double-barrier resonant-tunneling demonstration.

Units
-----
Dimensionless units with  hbar = m = 1.
  * energy  E  : in units of hbar^2 / (m L0^2)
  * length  x  : in units of L0
  * time    t  : in units of m L0^2 / hbar
  * wavenumber k = sqrt(2 E),  group velocity v = k
Any consistent physical system can be mapped onto these units by choosing L0
(for example L0 = 1 nm and the electron mass), so the results are generic.

Model
-----
The potential is piecewise constant (x measured from the left edge of barrier 1)::

    V = 0     for x < 0                 (incoming region)
    V = V0    for 0   <= x < b          (barrier 1)
    V = 0     for b   <= x < b + w      (quantum well)
    V = V0    for b+w <= x < 2b + w     (barrier 2)
    V = 0     for x >= 2b + w           (transmitted region)

Two independent methods are implemented:

1. Stationary scattering (transfer matrix):  T(E), R(E) for a plane wave of
   fixed energy E incident from the left.
2. Time-dependent Schroedinger equation (Crank-Nicolson finite differences)
   for a Gaussian wave packet.

The two are connected: the fraction of a packet that is transmitted equals
the T(E)-weighted average over the packet's energy distribution.  That gives a
genuinely independent cross-check, implemented in `predicted_packet_transmission`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu

# --------------------------------------------------------------------------
# Defaults shared by the app and the tests
# --------------------------------------------------------------------------
DEFAULT_BARRIER_HEIGHT = 6.0
DEFAULT_BARRIER_WIDTH = 0.4
DEFAULT_WELL_WIDTH = 1.5

GRID_DX = 0.025  # spatial step of the TDSE solver (all widths are snapped to it)
GRID_DT = 0.02   # time step of the TDSE solver


# --------------------------------------------------------------------------
# Structure definition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Structure:
    """Symmetric double-barrier structure (all quantities dimensionless)."""

    barrier_height: float = DEFAULT_BARRIER_HEIGHT
    barrier_width: float = DEFAULT_BARRIER_WIDTH
    well_width: float = DEFAULT_WELL_WIDTH

    def __post_init__(self):
        for name in ("barrier_height", "barrier_width", "well_width"):
            val = getattr(self, name)
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be a positive finite number, got {val!r}")

    @property
    def total_width(self) -> float:
        return 2.0 * self.barrier_width + self.well_width

    @property
    def layers(self):
        """List of (potential, thickness) for the layers between x=0 and x=total_width."""
        return [
            (self.barrier_height, self.barrier_width),
            (0.0, self.well_width),
            (self.barrier_height, self.barrier_width),
        ]

    def snapped(self, dx: float = GRID_DX) -> "Structure":
        """Return the structure with all widths rounded to whole grid cells.

        The TDSE solver can only represent barriers whose edges sit on grid
        cells.  Using the snapped structure in the transfer-matrix calculation
        too means both methods describe *exactly the same* potential.
        """
        def snap(width):
            return max(1, int(round(width / dx))) * dx

        return Structure(self.barrier_height, snap(self.barrier_width), snap(self.well_width))


# --------------------------------------------------------------------------
# Method 1: stationary scattering with the transfer matrix
# --------------------------------------------------------------------------
def _layer_matrix(E: np.ndarray, V: float, d: float):
    """Transfer matrix of one constant-potential layer acting on (psi, psi').

    Valid for E > V (oscillating) and E < V (evanescent, via complex k) and
    also at E == V.  Returned as four arrays (m11, m12, m21, m22).
    """
    ksq = 2.0 * (E - V)                       # real
    k = np.sqrt(ksq.astype(complex))          # imaginary when E < V
    kd = k * d
    c = np.cos(kd).real                       # cos or cosh -> always real
    small = np.abs(k) < 1e-9
    k_safe = np.where(small, 1.0, k)
    s_over_k = np.where(small, d, np.sin(kd) / k_safe).real   # sin(kd)/k or sinh/kappa
    m11 = c
    m12 = s_over_k
    m21 = -ksq * s_over_k
    m22 = c
    return m11, m12, m21, m22


def _mat_mul(A, B):
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    return (
        a11 * b11 + a12 * b21,
        a11 * b12 + a12 * b22,
        a21 * b11 + a22 * b21,
        a21 * b12 + a22 * b22,
    )


def _total_matrix(E: np.ndarray, structure: Structure):
    M = None
    for V, d in structure.layers:
        P = _layer_matrix(E, V, d)
        M = P if M is None else _mat_mul(P, M)   # later layers multiply from the left
    return M


def resonance_function(E, structure: Structure):
    """Smooth real function h(E) with  T(E) = 1 / (1 + h(E)^2)  for a symmetric structure.

    h is the (real) amplitude of the backward-travelling wave behind the
    structure for unit incident amplitude.  T = 1 exactly where h = 0, and
    because h varies slowly its zeros can be located reliably even for
    extremely narrow resonances that a plain scan of T(E) could step over.
    """
    E = np.atleast_1d(np.asarray(E, dtype=float))
    m11, m12, m21, m22 = _total_matrix(E, structure)
    k = np.sqrt(2.0 * E)
    return 0.5 * (k * m12 + m21 / k)


def scattering(E, structure: Structure):
    """Transmission T(E), reflection R(E) and reflection amplitude r(E).

    A plane wave exp(ikx) arrives from the left.  Because V = 0 on both sides,
    the transmitted wave has the same wavenumber, so T = |t|^2 directly.
    """
    E = np.atleast_1d(np.asarray(E, dtype=float))
    if np.any(E <= 0):
        raise ValueError("Energies must be strictly positive.")
    m11, m12, m21, m22 = _total_matrix(E, structure)
    k = np.sqrt(2.0 * E)
    # Matching: psi(0)=1+r, psi'(0)=ik(1-r);  psi'(L) = ik psi(L)  (outgoing only)
    a = 1j * k * m11 - m21
    c = -(k ** 2) * m12 - 1j * k * m22
    r = -(a + c) / (a - c)
    u = m11 * (1 + r) + 1j * k * m12 * (1 - r)   # = t * exp(ikL)
    return np.abs(u) ** 2, np.abs(r) ** 2, r


def transmission(E, structure: Structure):
    """Stationary transmission probability T(E) (array in, array out)."""
    return scattering(E, structure)[0]


def transmission_diagnostics(structure: Structure, n: int = 4000):
    """Sanity checks of the transfer-matrix result over a wide energy range.

    * T must be finite and in [0, 1]
    * T + R must equal 1 (probability conservation of the *stationary* solution)
    """
    e = np.linspace(1e-3 * structure.barrier_height, 1.5 * structure.barrier_height, n)
    t, r, _ = scattering(e, structure)
    finite = bool(np.all(np.isfinite(t)) and np.all(np.isfinite(r)))
    return {
        "finite": finite,
        "t_min": float(np.min(t)),
        "t_max": float(np.max(t)),
        "bounded": bool(finite and np.all(t >= -1e-12) and np.all(t <= 1 + 1e-9)),
        "unitarity_error": float(np.max(np.abs(t + r - 1.0))),
    }


# --------------------------------------------------------------------------
# Resonance detection
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Resonance:
    energy: float   # energy of the transmission maximum
    t_peak: float   # T at the maximum (1.0 for a symmetric structure)
    fwhm: float     # full width at half maximum of the peak, in energy


def _peak_width(structure: Structure, e_peak: float, t_peak: float) -> float:
    """Full width at half maximum of a transmission peak."""
    half = 0.5 * t_peak

    def f(e):
        return float(transmission(e, structure)[0]) - half

    def edge(direction):
        step = max(1e-6, 1e-3 * e_peak)
        e_prev = e_peak
        for _ in range(60):
            e_new = e_prev + direction * step
            if e_new <= 1e-9:
                return None
            if f(e_new) < 0:
                return brentq(f, min(e_prev, e_new), max(e_prev, e_new), xtol=1e-12)
            e_prev = e_new
            step *= 1.6
        return None

    lo, hi = edge(-1), edge(+1)
    if lo is None and hi is None:
        return float("nan")
    if lo is None:
        return 2.0 * (hi - e_peak)
    if hi is None:
        return 2.0 * (e_peak - lo)
    return hi - lo


def find_resonances(structure: Structure, e_max: float | None = None, n_scan: int = 20000):
    """Find all transmission resonances below the barrier top (E < V0).

    For a symmetric structure T = 1/(1 + h^2) with h smooth (see
    `resonance_function`), so every resonance is a sign change of h.  We scan
    h on a fine grid, then polish each root with Brent's method.  Because h
    is smooth, even resonances far narrower than the grid spacing are found.
    """
    v0 = structure.barrier_height
    e_max = v0 if e_max is None else e_max
    e_min = 1e-4 * v0
    grid = np.linspace(e_min, e_max, n_scan)
    h = resonance_function(grid, structure)
    flips = np.where(np.sign(h[:-1]) * np.sign(h[1:]) < 0)[0]
    found = []
    for i in flips:
        e_r = brentq(lambda e: float(resonance_function(e, structure)[0]), grid[i], grid[i + 1],
                     xtol=1e-14, rtol=1e-14)
        t_r = float(transmission(e_r, structure)[0])
        found.append(Resonance(float(e_r), t_r, _peak_width(structure, e_r, t_r)))
    return found


def nearest_resonance(resonances, energy: float):
    if not resonances:
        return None
    return min(resonances, key=lambda r: abs(r.energy - energy))


def resonance_status(structure: Structure, energy: float, resonances=None):
    """Classify an energy as 'on', 'near' or 'off' resonance.

    Returns (label, nearest_resonance_or_None, T(E)).
    """
    if resonances is None:
        resonances = find_resonances(structure)
    t_val = float(transmission(energy, structure)[0])
    r = nearest_resonance(resonances, energy)
    if r is None or not np.isfinite(r.fwhm) or r.fwhm <= 0:
        return "off", r, t_val
    x = abs(energy - r.energy) / (0.5 * r.fwhm)
    if x <= 0.25:
        label = "on"
    elif x <= 1.0:
        label = "near"
    else:
        label = "off"
    return label, r, t_val


def pick_nonresonant_energy(structure: Structure, e_res: float, resonances=None, decimals: int = 3):
    """Choose a useful non-resonant comparison energy near `e_res`.

    The choice is the transmission minimum (valley) closest to `e_res`, which
    shows the strongest contrast.  If no valley is found we fall back to the
    smallest T in a window around the resonance.
    """
    v0 = structure.barrier_height
    lo, hi = max(0.05 * v0, 0.4 * e_res), min(0.98 * v0, 2.2 * e_res)
    grid = np.linspace(lo, hi, 4000)
    t = transmission(grid, structure)
    valleys, _ = find_peaks(-t)
    if len(valleys):
        # valleys must be clearly lower than the resonant peak
        good = [i for i in valleys if t[i] < 0.5]
        pool = good if good else list(valleys)
        i_best = min(pool, key=lambda i: abs(grid[i] - e_res))
    else:
        i_best = int(np.argmin(t))
    return round(float(grid[i_best]), decimals)


@dataclass(frozen=True)
class DemoEnergies:
    resonant: float
    non_resonant: float
    t_resonant: float
    t_non_resonant: float
    resonance: Resonance | None


def demo_energies(structure: Structure, resonances=None, reference_energy: float | None = None,
                  decimals: int = 3) -> DemoEnergies:
    """Resonant and non-resonant energies used for the comparison.

    By default the lowest resonance with a strong peak is used.  If
    `reference_energy` is given, the resonance nearest to it is used instead.
    """
    if resonances is None:
        resonances = find_resonances(structure)
    if not resonances:
        raise RuntimeError("No transmission resonance found for these parameters.")
    if reference_energy is None:
        strong = [r for r in resonances if r.t_peak > 0.9]
        res = strong[0] if strong else max(resonances, key=lambda r: r.t_peak)
    else:
        res = nearest_resonance(resonances, reference_energy)
    e_res = res.energy
    for d in range(decimals, 10):               # round only as far as the peak allows
        cand = round(res.energy, d)
        if cand > 0 and float(transmission(cand, structure)[0]) >= 0.98 * res.t_peak:
            e_res = cand
            break
    e_off = pick_nonresonant_energy(structure, res.energy, resonances, decimals)
    return DemoEnergies(
        e_res, e_off,
        float(transmission(e_res, structure)[0]),
        float(transmission(e_off, structure)[0]),
        res,
    )


# --------------------------------------------------------------------------
# Method 2: time-dependent Schroedinger equation (Crank-Nicolson)
# --------------------------------------------------------------------------
def design_packet_sigma(e0: float, fwhm_ref: float | None,
                        sigma_min: float = 4.0, sigma_max: float = 8.0) -> float:
    """Pick the spatial width (std. dev. of |psi|^2) of the Gaussian packet.

    Trade-off: a packet of spatial width sigma has an energy spread
        dE ~ k0 / (2 sigma)       (k0 = sqrt(2 E0)).
    To feel the resonance sharply, dE should be comparable to or smaller than
    the resonance width.  But narrow resonances need very long packets, which
    take long to simulate and no longer fit on screen.  We therefore aim for
    dE ~ 0.5 * FWHM and clamp sigma to [sigma_min, sigma_max].
    """
    k0 = np.sqrt(2.0 * e0)
    if fwhm_ref is None or not np.isfinite(fwhm_ref) or fwhm_ref <= 0:
        fwhm_ref = 0.25 * e0
    sigma = k0 / (2.0 * 0.5 * fwhm_ref)
    return float(np.clip(sigma, sigma_min, sigma_max))


def packet_energy_spread(e0: float, sigma: float) -> float:
    """Standard deviation of the packet's energy distribution (approx.)."""
    return float(np.sqrt(2.0 * e0) / (2.0 * sigma))


def build_potential(structure: Structure, dx: float, n_left: int, n_right: int):
    """Grid and potential.  Returns x, V, index of first cell of the structure, cells in it."""
    s = structure.snapped(dx)
    nb = int(round(s.barrier_width / dx))
    nw = int(round(s.well_width / dx))
    n_struct = 2 * nb + nw
    n = n_left + n_struct + n_right
    x = (np.arange(n) - n_left) * dx
    V = np.zeros(n)
    V[n_left:n_left + nb] = s.barrier_height
    V[n_left + nb + nw:n_left + n_struct] = s.barrier_height
    return x, V, n_left, n_struct


def gaussian_packet(x, x0: float, k0: float, sigma: float, dx: float):
    """Normalised Gaussian wave packet  psi ~ exp(-(x-x0)^2/(4 sigma^2)) exp(i k0 x)."""
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2) + 1j * k0 * x)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm


def predicted_packet_transmission(psi0: np.ndarray, dx: float, structure: Structure):
    """Transmitted fraction predicted from the stationary T(E).

    The packet is split into plane-wave components (FFT).  Each component with
    wavenumber k > 0 is transmitted with probability T(k^2/2), so
        T_packet = sum_k |phi(k)|^2 T(k^2/2)  /  sum_k |phi(k)|^2 .
    This uses only the transfer-matrix result and the initial packet, so it is
    independent of the time-stepping code.
    """
    phi = np.fft.fft(psi0)
    k = 2.0 * np.pi * np.fft.fftfreq(len(psi0), d=dx)
    w = np.abs(phi) ** 2
    w = w / w.sum()
    pos = k > 1e-6
    t_k = np.zeros_like(k)
    t_k[pos] = transmission(0.5 * k[pos] ** 2, structure)
    reflected_left_moving = 1.0 - w[pos].sum()   # components with k <= 0 (negligible)
    return float(np.sum(w * t_k)), float(reflected_left_moving)


@dataclass
class SimulationResult:
    structure: Structure                 # snapped structure actually simulated
    energy: float
    k0: float
    sigma: float
    energy_spread: float
    dx: float
    dt: float
    n_steps: int
    x: np.ndarray
    potential: np.ndarray
    i_struct0: int                       # first grid cell inside the structure
    n_struct: int                        # number of cells inside the structure
    times: np.ndarray                    # frame times
    frames: np.ndarray                   # complex psi, shape (n_frames, n_x)
    p_before: np.ndarray                 # probability left of structure, per frame
    p_inside: np.ndarray                 # probability inside structure, per frame
    p_after: np.ndarray                  # probability right of structure, per frame
    norm_initial: float
    norm_final: float
    norm_max_error: float
    stationary_T: float
    predicted_transmitted: float
    measured_transmitted: float          # P(after) at final time
    measured_reflected: float            # P(before) at final time
    remaining_inside: float              # P(inside) at final time
    warnings: list = field(default_factory=list)

    @property
    def conservation_error(self) -> float:
        return abs(self.norm_final - self.norm_initial)


def simulate(structure: Structure, energy: float, sigma: float | None = None,
             fwhm_ref: float | None = None, dx: float = GRID_DX, dt: float = GRID_DT,
             n_frames: int = 64) -> SimulationResult:
    """Run a wave-packet scattering simulation with the Crank-Nicolson scheme.

    Scheme:  (1 + i dt H/2) psi^{n+1} = (1 - i dt H/2) psi^n,   H = -1/2 d^2/dx^2 + V.
    It is unconditionally stable and exactly unitary for the discrete (real,
    symmetric) Hamiltonian, so total probability is conserved to round-off.
    Boundaries are hard walls (psi = 0); the box is made big enough that the
    packet never reaches them during the run.
    """
    if energy <= 0:
        raise ValueError("Energy must be positive.")
    s = structure.snapped(dx)
    k0 = float(np.sqrt(2.0 * energy))
    if sigma is None:
        sigma = design_packet_sigma(energy, fwhm_ref)
    ls = s.total_width

    # Dwell-time allowance: a resonance holds probability for about 1/FWHM; the
    # exponential tail needs a few lifetimes to drain out.
    tau = 0.0
    if fwhm_ref is not None and np.isfinite(fwhm_ref) and fwhm_ref > 0:
        tau = min(2.5 / fwhm_ref, 40.0)

    d0 = 4.0 * sigma + 3.0                              # start distance left of barrier 1
    travel = (d0 + ls + 4.0 * sigma + 3.0) / k0 + tau   # time to get everything past the structure
    t_final = travel
    n_steps = int(np.ceil(t_final / dt))
    n_steps = max(n_steps, n_frames)

    margin = 5.0 * sigma
    reach = k0 * t_final                                # farthest a packet centre can travel
    left_extent = d0 + max(0.0, reach - d0) + margin    # reflected packet leaves to the left
    right_extent = ls + (reach - d0) + margin
    n_left = int(np.ceil(max(left_extent, d0 + margin) / dx))
    n_right = int(np.ceil(max(right_extent, margin) / dx))
    x, V, i0, n_struct = build_potential(s, dx, n_left, n_right)
    n = len(x)

    psi = gaussian_packet(x, -d0, k0, sigma, dx).astype(complex)
    norm0 = float(np.sum(np.abs(psi) ** 2) * dx)

    # Hamiltonian (Dirichlet walls) and Crank-Nicolson matrices
    main = 1.0 / dx ** 2 + V
    off = -0.5 / dx ** 2 * np.ones(n - 1)
    H = diags([off, main, off], [-1, 0, 1], format="csc")
    ident = identity(n, format="csc", dtype=complex)
    A = (ident + 0.5j * dt * H).tocsc()
    B = (ident - 0.5j * dt * H).tocsr()
    lu = splu(A)

    frame_steps = np.unique(np.linspace(0, n_steps, n_frames).round().astype(int))
    frames, times = [], []
    p_before, p_inside, p_after = [], [], []
    norm_err = 0.0
    sl_in = slice(i0, i0 + n_struct)

    def record(psi_now, step):
        dens = np.abs(psi_now) ** 2
        frames.append(psi_now.copy())
        times.append(step * dt)
        p_before.append(float(dens[:i0].sum() * dx))
        p_inside.append(float(dens[sl_in].sum() * dx))
        p_after.append(float(dens[i0 + n_struct:].sum() * dx))

    record(psi, 0)
    frame_set = set(frame_steps[1:].tolist())
    for step in range(1, n_steps + 1):
        psi = lu.solve(B @ psi)
        if step in frame_set:
            record(psi, step)
            norm_err = max(norm_err, abs(float(np.sum(np.abs(psi) ** 2) * dx) - norm0))

    norm_final = float(np.sum(np.abs(psi) ** 2) * dx)
    t_stat = float(transmission(energy, s)[0])
    t_pred, neg_k = predicted_packet_transmission(gaussian_packet(x, -d0, k0, sigma, dx), dx, s)

    result = SimulationResult(
        structure=s, energy=float(energy), k0=k0, sigma=float(sigma),
        energy_spread=packet_energy_spread(energy, sigma), dx=dx, dt=dt, n_steps=n_steps,
        x=x, potential=V, i_struct0=i0, n_struct=n_struct,
        times=np.array(times), frames=np.array(frames),
        p_before=np.array(p_before), p_inside=np.array(p_inside), p_after=np.array(p_after),
        norm_initial=norm0, norm_final=norm_final, norm_max_error=float(norm_err),
        stationary_T=t_stat, predicted_transmitted=t_pred,
        measured_transmitted=p_after[-1], measured_reflected=p_before[-1],
        remaining_inside=p_inside[-1],
    )
    result.warnings = check_simulation(result, neg_k_weight=neg_k)
    return result


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def check_simulation(sim: SimulationResult, neg_k_weight: float = 0.0):
    """Return a list of human-readable warnings (empty list = all checks passed)."""
    msgs = []
    if not np.all(np.isfinite(sim.frames)):
        msgs.append("The solver produced non-finite values; the result cannot be trusted.")
    if abs(sim.norm_initial - 1.0) > 1e-6:
        msgs.append(f"The initial wave packet is not normalised (norm = {sim.norm_initial:.6f}).")
    if sim.norm_max_error > 1e-8:
        msgs.append(f"Probability was not conserved (max drift {sim.norm_max_error:.2e}).")
    if not (0.0 <= sim.stationary_T <= 1.0 + 1e-9) or not np.isfinite(sim.stationary_T):
        msgs.append("The stationary transmission is outside [0, 1]; the transfer-matrix result is invalid.")
    if sim.remaining_inside > 0.02:
        msgs.append(
            f"{sim.remaining_inside:.1%} of the packet is still inside the structure at the end "
            "of the run, so the transmitted fraction is slightly under-counted. "
            "(Narrow resonances trap probability for a long time.)"
        )
    if neg_k_weight > 1e-3:
        msgs.append("The packet is so broad in energy that part of it has k<=0; the comparison with T(E) is approximate.")
    gap = abs(sim.predicted_transmitted - sim.measured_transmitted)
    if gap > 0.03 + sim.remaining_inside:
        msgs.append(
            f"The time-dependent result ({sim.measured_transmitted:.3f}) and the stationary-theory "
            f"prediction ({sim.predicted_transmitted:.3f}) differ by {gap:.3f}; treat numbers with care."
        )
    return msgs


def settings_reliability_warnings(structure: Structure, energy: float, sigma: float,
                                  resonance: Resonance | None):
    """Warnings about the *physics setting* (not about numerical failure)."""
    msgs = []
    dE = packet_energy_spread(energy, sigma)
    if resonance is not None and np.isfinite(resonance.fwhm):
        if resonance.fwhm < 0.8 * dE:
            msgs.append(
                f"This resonance is very sharp (width ~ {resonance.fwhm:.3g}) compared with the energy "
                f"spread of the wave packet (~ {dE:.3g}). Only a small fraction of the packet can get "
                "through even on resonance. Try thinner barriers or a lower barrier height to see a "
                "stronger packet-level effect."
            )
    return msgs
