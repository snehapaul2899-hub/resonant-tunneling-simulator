"""
visuals.py
==========

Matplotlib figures and the embedded animation for the resonant-tunneling app.
No Streamlit code lives here, so everything can be tested without a browser.

All plotted probability densities are the *real* |psi|^2 produced by the solver.
Nothing is rescaled or amplified.  The only separate vertical scale is the
"zoom" panel of the animation, which is labelled as such.
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")  # no GUI needed; figures are rendered to images
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

import physics_engine as pe  # noqa: E402

# Colour palette (colour-blind friendly)
C_DENSITY = "#1f6fb4"
C_REAL = "#1f6fb4"
C_IMAG = "#d97706"
C_BARRIER = "#c2410c"
C_WELL = "#0f766e"
C_TRANSMIT = "#15803d"
C_REFLECT = "#6b7280"
C_RES = "#15803d"
C_OFF = "#b91c1c"

STAGE_NAMES = ["1  Incoming", "2  Barrier 1", "3  In the well", "4  Transmission", "5  Result"]

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "animation.embed_limit": 120.0,  # MB
})


# --------------------------------------------------------------------------
# Stage detection from the actual simulation data
# --------------------------------------------------------------------------
def detect_stages(sim: pe.SimulationResult):
    """Return the first frame index of stages 2..5, found from the real data.

    Stage 2 starts when probability first enters the structure region,
    stage 3 when the structure is substantially "loaded",
    stage 4 when a noticeable transmitted part has appeared,
    stage 5 once the probability inside the structure has mostly drained away.
    """
    n = len(sim.times)
    pin, paf = sim.p_inside, sim.p_after
    pin_max = max(float(pin.max()), 1e-12)
    i_peak = int(np.argmax(pin))

    def first(cond, start=0, default=None):
        idx = np.where(cond[start:])[0]
        return int(idx[0] + start) if len(idx) else default

    i2 = first(pin > 0.03 * pin_max, 0, n // 4)
    i3 = first(pin > 0.5 * pin_max, i2, i_peak)
    pa_final = max(float(paf[-1]), 1e-12)
    i4 = first(paf > 0.15 * pa_final, i3, min(n - 3, i3 + 1))
    i5 = first(pin < 0.15 * pin_max, max(i_peak, i4), None)
    if i5 is None:
        i5 = int(0.88 * n)
    # enforce strictly increasing order inside the available frames
    out = [i2]
    for cand in (i3, i4, i5):
        out.append(max(cand, out[-1] + 1))
    out = [min(v, n - 1) for v in out]
    for j in range(len(out) - 2, -1, -1):          # keep strictly increasing after clipping
        out[j] = min(out[j], out[j + 1] - 1)
    return out


def stage_caption(stage: int, sim: pe.SimulationResult, status: str) -> str:
    """Short plain-language caption for stage 0..4."""
    resonant = status in ("on", "near")
    if stage == 0:
        return ("A wave packet (a smooth bump of probability) travels toward the structure. "
                "Quantum particles behave like waves, so the packet is spread out in space.")
    if stage == 1:
        return ("The wave meets barrier 1. Most of it bounces back (reflection), but a small part "
                "leaks into and through the barrier. That leaking is quantum tunneling.")
    if stage == 2:
        if resonant:
            return ("RESONANCE: inside the well the wave bounces between the two barriers and the "
                    "bounces line up in phase (constructive interference), so probability builds up in the well.")
        return ("OFF RESONANCE: inside the well the bounces do not line up in phase, they partly cancel "
                "(destructive interference), so little probability gets into the well.")
    if stage == 3:
        if resonant:
            return ("Because the well is 'in tune' with this energy, the wave passes barrier 2 easily: "
                    "a strong transmitted wave appears on the right.")
        return ("Barrier 2 lets almost nothing out, because no resonance is helping: "
                "only a tiny transmitted wave appears on the right.")
    return (f"Result: {sim.measured_transmitted:.0%} of this packet got through and "
            f"{sim.measured_reflected:.0%} was reflected. "
            "A packet contains many energies, so this can differ from the single-energy value T(E).")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _view_limits(sim: pe.SimulationResult):
    """x-limits for the overview panel, wide enough to hold start and end of the run."""
    ls = sim.structure.total_width
    d0 = 4.0 * sim.sigma + 3.0
    reach = sim.k0 * sim.times[-1]
    half = max(d0, reach - d0)
    return -half - 3.0 * sim.sigma, ls + half + 3.0 * sim.sigma


def _zoom_limits(sim: pe.SimulationResult):
    ls = sim.structure.total_width
    pad = 4.5
    return -pad, ls + pad


def _decimate(x, arrs, target=2400):
    step = max(1, len(x) // target)
    return x[::step], [a[::step] for a in arrs]


def _shade_structure(ax, sim, ymax=None, alpha=0.30, label_bars=False):
    s = sim.structure
    b, w = s.barrier_width, s.well_width
    top = ax.get_ylim()[1] if ymax is None else ymax
    bottom = ax.get_ylim()[0]
    ax.fill_between([0, b], bottom, top, color=C_BARRIER, alpha=alpha, lw=0, zorder=0)
    ax.fill_between([b, b + w], bottom, top, color=C_WELL, alpha=0.10, lw=0, zorder=0)
    ax.fill_between([b + w, 2 * b + w], bottom, top, color=C_BARRIER, alpha=alpha, lw=0, zorder=0)


# --------------------------------------------------------------------------
# Main animation
# --------------------------------------------------------------------------
def _build_animation_figure(sim: pe.SimulationResult, status: str, view: str, dpi: int):
    """Build the animation figure and return (fig, draw_frame, n_frames)."""
    if view not in ("density", "reim"):
        raise ValueError("view must be 'density' or 'reim'")
    x = sim.x
    s = sim.structure
    ls = s.total_width
    n_frames = len(sim.times)
    stages = detect_stages(sim)
    ox0, ox1 = _view_limits(sim)
    zx0, zx1 = _zoom_limits(sim)

    mask_o = (x >= ox0) & (x <= ox1)
    mask_z = (x >= zx0) & (x <= zx1)
    xo, xz = x[mask_o], x[mask_z]
    xo_d = _decimate(xo, [xo])[0]
    step_o = max(1, len(xo) // 2400)

    inside_o = (xo >= -sim.dx) & (xo <= ls + sim.dx)      # structure interior: shown in the zoom panel
    dens = np.abs(sim.frames) ** 2
    if view == "density":
        y_over = 1.25 * dens[:, mask_o][:, ~inside_o].max()
        y_zoom = 1.18 * dens[:, mask_z].max()
        lo_over = lo_zoom = 0.0
    else:
        amp = np.abs(sim.frames)
        y_over = 1.25 * amp[:, mask_o][:, ~inside_o].max()
        y_zoom = 1.18 * amp[:, mask_z].max()
        lo_over, lo_zoom = -y_over, -y_zoom

    fig = plt.figure(figsize=(9.4, 8.7), dpi=dpi)
    fig.patch.set_facecolor("white")

    # --- header: stage strip + caption -----------------------------------
    ax_head = fig.add_axes([0.02, 0.905, 0.96, 0.088])
    ax_head.axis("off")
    stage_txt = []
    for i, name in enumerate(STAGE_NAMES):
        t = ax_head.text((i + 0.5) / 5.0, 0.80, name, ha="center", va="center", fontsize=9.5,
                         bbox=dict(boxstyle="round,pad=0.35", fc="#f3f4f6", ec="#d1d5db"),
                         color="#6b7280", transform=ax_head.transAxes)
        stage_txt.append(t)
    caption = ax_head.text(0.0, 0.40, "", ha="left", va="top", fontsize=10, color="#111827",
                           transform=ax_head.transAxes, linespacing=1.25)

    # --- overview ---------------------------------------------------------
    ax_o = fig.add_axes([0.075, 0.665, 0.90, 0.175])
    ax_o.set_xlim(ox0, ox1)
    ax_o.set_ylim(lo_over, y_over)
    ax_o.set_xlabel("position x   (overview; the structure interior is shown in the zoom panel below)", labelpad=1)
    ax_o.set_ylabel("|ψ|²" if view == "density" else "ψ")
    ax_o.axvspan(0, ls, color=C_BARRIER, alpha=0.35, lw=0, zorder=0)
    ax_o.text(ox0 + 0.01 * (ox1 - ox0), 1.04, "◀ INCOMING region (wave arrives from the left)",
              transform=ax_o.get_xaxis_transform(), ha="left", va="bottom", fontsize=8.8, color="#374151")
    ax_o.text(ox1 - 0.01 * (ox1 - ox0), 1.04, "TRANSMITTED region (wave that got through) ▶",
              transform=ax_o.get_xaxis_transform(), ha="right", va="bottom", fontsize=8.8, color=C_TRANSMIT)
    ax_o.annotate("double barrier", xy=(ls / 2, 1.0), xycoords=("data", "axes fraction"),
                  xytext=(ls / 2, 1.04), textcoords=("data", "axes fraction"),
                  ha="center", va="bottom", fontsize=8.8, color=C_BARRIER)
    if view == "density":
        line_o, = ax_o.plot([], [], color=C_DENSITY, lw=1.4)
        lines_o = [line_o]
    else:
        l1, = ax_o.plot([], [], color=C_REAL, lw=1.0, label="Re ψ")
        l2, = ax_o.plot([], [], color=C_IMAG, lw=1.0, label="Im ψ")
        lines_o = [l1, l2]
        ax_o.legend(loc="upper left", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.0, 0.93))
    fill_o = [None]

    # --- zoom -------------------------------------------------------------
    ax_z = fig.add_axes([0.075, 0.325, 0.90, 0.225])
    ax_z.set_xlim(zx0, zx1)
    ax_z.set_ylim(lo_zoom, y_zoom)
    ax_z.set_xlabel("position x   (ZOOM on the structure: same quantity, own vertical scale)", labelpad=1)
    ax_z.set_ylabel("|ψ|²" if view == "density" else "ψ")
    _shade_structure(ax_z, sim, ymax=y_zoom)
    b, w = s.barrier_width, s.well_width
    tr = ax_z.get_xaxis_transform()
    ax_z.text(zx0 + 0.02 * (zx1 - zx0), 1.05, "INCOMING\nregion", transform=tr, ha="left", va="bottom",
              fontsize=8.8, color="#374151")
    ax_z.text(zx1 - 0.02 * (zx1 - zx0), 1.05, "TRANSMITTED\nregion", transform=tr, ha="right",
              va="bottom", fontsize=8.8, color=C_TRANSMIT)
    for xt, xtext, txt, col in [(b / 2, b / 2 - 1.1, "BARRIER 1", C_BARRIER),
                                (b + w / 2, b + w / 2, "QUANTUM WELL", C_WELL),
                                (1.5 * b + w, 1.5 * b + w + 1.1, "BARRIER 2", C_BARRIER)]:
        ax_z.annotate(txt, xy=(xt, 1.0), xycoords=("data", "axes fraction"),
                      xytext=(xtext, 1.20), textcoords=("data", "axes fraction"), ha="center",
                      va="bottom", fontsize=8.8, color=col, fontweight="bold",
                      arrowprops=dict(arrowstyle="-", color=col, lw=0.8, shrinkA=0, shrinkB=0))
    if view == "density":
        line_z, = ax_z.plot([], [], color=C_DENSITY, lw=1.6)
        lines_z = [line_z]
    else:
        z1, = ax_z.plot([], [], color=C_REAL, lw=1.5)
        z2, = ax_z.plot([], [], color=C_IMAG, lw=1.5)
        lines_z = [z1, z2]
    fill_z = [None]

    # --- probability-by-region bars --------------------------------------
    ax_b = fig.add_axes([0.30, 0.065, 0.66, 0.125])
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(-0.6, 2.6)
    ax_b.set_yticks([0, 1, 2])
    ax_b.set_yticklabels(["TRANSMITTED region\n(after barrier 2)",
                          "INSIDE the structure\n(barriers + well)",
                          "INCOMING region\n(before barrier 1, incl. reflected)"], fontsize=8.6)
    ax_b.set_xlabel("probability of finding the particle in each region (the three always add up to 100 %)",
                    fontsize=8.8, labelpad=2)
    ax_b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    bars = ax_b.barh([2, 1, 0], [0, 0, 0], height=0.62, color=[C_REFLECT, C_BARRIER, C_TRANSMIT])
    bar_txt = [ax_b.text(0.0, yy, "", va="center", ha="left", fontsize=9, fontweight="bold")
               for yy in (2, 1, 0)]

    time_txt = fig.text(0.975, 0.228, "", ha="right", va="center", fontsize=9, color="#6b7280")

    def draw_frame(k):
        psi = sim.frames[k]
        if view == "density":
            d = np.abs(psi) ** 2
            d_o = np.where(inside_o, np.nan, d[mask_o])
            lines_o[0].set_data(xo[::step_o], d_o[::step_o])
            lines_z[0].set_data(xz, d[mask_z])
            for coll, ax_, xs, ys in ((fill_o, ax_o, xo[::step_o], d_o[::step_o]),
                                      (fill_z, ax_z, xz, d[mask_z])):
                if coll[0] is not None:
                    coll[0].remove()
                coll[0] = ax_.fill_between(xs, 0, ys, color=C_DENSITY, alpha=0.22, lw=0)
        else:
            po = np.where(inside_o, np.nan, psi[mask_o])[::step_o]
            pz = psi[mask_z]
            lines_o[0].set_data(xo[::step_o], po.real)
            lines_o[1].set_data(xo[::step_o], po.imag)
            lines_z[0].set_data(xz, pz.real)
            lines_z[1].set_data(xz, pz.imag)
        # bars are ordered [incoming(2), inside(1), transmitted(0)]
        for bar, txt, v in zip(bars, bar_txt, [sim.p_before[k], sim.p_inside[k], sim.p_after[k]]):
            bar.set_width(v)
            txt.set_text(f"{v:.1%}")
            txt.set_x(min(v + 0.01, 0.86))
        st = sum(1 for idx in stages if k >= idx)       # 0..4
        for i, t in enumerate(stage_txt):
            if i == st:
                t.set_color("white")
                t.get_bbox_patch().set(fc=C_DENSITY, ec=C_DENSITY)
            else:
                t.set_color("#6b7280" if i > st else "#374151")
                t.get_bbox_patch().set(fc="#f3f4f6" if i > st else "#e5e7eb", ec="#d1d5db")
        caption.set_text("\n".join(textwrap.wrap(stage_caption(st, sim, status), 118)))
        time_txt.set_text(f"t = {sim.times[k]:.1f}  (dimensionless units)")
        return []

    return fig, draw_frame, n_frames


def build_animation_html(sim: pe.SimulationResult, status: str, view: str = "density",
                         fps: int = 12, dpi: int = 66) -> str:
    """Create the stage-aware animation and return it as self-contained HTML.

    view = "density" : |psi|^2 (probability density)
    view = "reim"    : Re(psi) and Im(psi)  (shows the phase / interference)
    """
    fig, draw_frame, n_frames = _build_animation_figure(sim, status, view, dpi)
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 // fps, blit=False)
    html = anim.to_jshtml(fps=fps, default_mode="loop")
    plt.close(fig)
    return html


def render_frame_png(sim: pe.SimulationResult, status: str, view: str, frame: int, path: str, dpi: int = 100):
    """Save one animation frame as a PNG (used for visual inspection and tests)."""
    fig, draw_frame, n_frames = _build_animation_figure(sim, status, view, dpi)
    draw_frame(min(max(frame, 0), n_frames - 1))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------
# Transmission spectrum
# --------------------------------------------------------------------------
def figure_spectrum(structure: pe.Structure, energy: float, resonances, status: str,
                    energy_spread: float, e_res: float | None = None, e_off: float | None = None):
    """T(E) with resonance peaks, selected energy and the packet's energy range."""
    v0 = structure.barrier_height
    E = np.linspace(1e-3 * v0, v0, 6000)
    extra = [np.linspace(max(1e-4, r.energy - 4 * r.fwhm), min(v0, r.energy + 4 * r.fwhm), 160)
             for r in resonances if np.isfinite(r.fwhm)]       # draw even very narrow peaks faithfully
    E = np.unique(np.concatenate([E] + extra))
    T = pe.transmission(E, structure.snapped())
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=100)
    ax.plot(E, T, color="#111827", lw=1.8, label="T(E): stationary transmission")
    ax.fill_between(E, 0, T, color=C_DENSITY, alpha=0.10)
    for i, r in enumerate(resonances, 1):
        ax.plot(r.energy, r.t_peak, "o", color=C_RES, ms=7, zorder=5)
        ax.annotate(f"resonance {i}\nE={r.energy:.3f}", (r.energy, r.t_peak),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.3, color=C_RES)
    t_sel = float(pe.transmission(energy, structure.snapped())[0])
    col = {"on": C_RES, "near": "#ca8a04", "off": C_OFF}[status]
    ax.axvspan(energy - energy_spread, energy + energy_spread, color=col, alpha=0.15, lw=0,
               label="energy range of the wave packet (±1 spread)")
    ax.axvline(energy, color=col, lw=1.6, ls="--")
    ax.plot(energy, t_sel, "D", color=col, ms=8, zorder=6, label=f"selected energy: T = {t_sel:.3f}")
    if e_off is not None:
        ax.plot(e_off, float(pe.transmission(e_off, structure.snapped())[0]), "s", color=C_OFF,
                ms=6, zorder=5, label="non-resonant comparison energy")
    ax.set_xlim(0, v0)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("incident energy E   (barrier top is at E = %.1f)" % v0)
    ax.set_ylabel("transmission probability T(E)")
    ax.legend(loc="upper center", fontsize=8.2, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.20))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Resonant vs non-resonant comparison
# --------------------------------------------------------------------------
def figure_comparison(sim_res: pe.SimulationResult, sim_off: pe.SimulationResult):
    """Side-by-side final snapshots on identical axes plus transmitted-fraction bars."""
    fig = plt.figure(figsize=(11.0, 3.9), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.9], wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[2])

    x0a, x1a = _view_limits(sim_res)
    x0b, x1b = _view_limits(sim_off)
    x0, x1 = min(x0a, x0b), max(x1a, x1b)
    ymax = 1.15 * max(np.abs(sim_res.frames[-1]).max() ** 2, np.abs(sim_off.frames[-1]).max() ** 2)
    ls = sim_res.structure.total_width

    for ax, sim, title, col in ((ax1, sim_res, "ON resonance", C_RES), (ax2, sim_off, "OFF resonance", C_OFF)):
        d = np.abs(sim.frames[-1]) ** 2
        x = sim.x
        m = (x >= x0) & (x <= x1)
        xs, ds = x[m], d[m]
        ax.axvspan(0, ls, color=C_BARRIER, alpha=0.35, lw=0)
        ax.fill_between(xs, 0, ds, where=xs < 0, color=C_REFLECT, alpha=0.45, lw=0)
        ax.fill_between(xs, 0, ds, where=xs > ls, color=C_TRANSMIT, alpha=0.65, lw=0)
        ax.plot(xs, ds, color="#111827", lw=1.0)
        ax.set_title(f"{title}:  E = {sim.energy:.3f}", color=col, fontweight="bold", loc="left")
        ax.text(0.03, 0.93, f"reflected  {sim.measured_reflected:.1%}", transform=ax.transAxes,
                color="#374151", fontsize=9, va="top")
        ax.text(0.97, 0.93, f"transmitted  {sim.measured_transmitted:.1%}", transform=ax.transAxes,
                color=C_TRANSMIT, fontsize=9, fontweight="bold", va="top", ha="right")
        ax.set_xlabel("position x  (barriers in orange)")
    ax1.set_ylabel("|ψ|² at the end of the run")
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(0, ymax)
    plt.setp(ax2.get_yticklabels(), visible=False)

    cats = ["ON\nresonance", "OFF\nresonance"]
    t_stat = [sim_res.stationary_T, sim_off.stationary_T]
    t_pack = [sim_res.measured_transmitted, sim_off.measured_transmitted]
    xpos = np.arange(2)
    b1 = ax3.bar(xpos - 0.19, t_stat, 0.36, color="#9ca3af", label="stationary T(E)\n(one exact energy)")
    b2 = ax3.bar(xpos + 0.19, t_pack, 0.36, color=C_TRANSMIT, label="wave packet\n(simulation)")
    for bars_ in (b1, b2):
        for r in bars_:
            ax3.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02, f"{r.get_height():.2f}",
                     ha="center", fontsize=8.8)
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(cats)
    ax3.set_ylim(0, 1.45)
    ax3.set_ylabel("fraction transmitted")
    ax3.set_title("Same scale, side by side", loc="left")
    ax3.legend(fontsize=8, frameon=False, loc="upper center", ncol=1)
    return fig
