"""
app.py - Quantum resonant tunneling, explained for beginners.

Run with:   streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

import physics_engine as pe
import visuals as vz

st.set_page_config(page_title="Resonant tunneling explained", page_icon="🌊", layout="wide")

DEFAULTS = {
    "V0": pe.DEFAULT_BARRIER_HEIGHT,
    "b": pe.DEFAULT_BARRIER_WIDTH,
    "w": pe.DEFAULT_WELL_WIDTH,
}


# --------------------------------------------------------------------------
# Cached computations (sliders re-run the script; heavy work must not repeat)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=32)
def cached_resonances(v0: float, b: float, w: float):
    structure = pe.Structure(v0, b, w).snapped()
    return pe.find_resonances(structure)


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_simulation(v0: float, b: float, w: float, energy: float, sigma: float, fwhm: float):
    structure = pe.Structure(v0, b, w).snapped()
    return pe.simulate(structure, energy, sigma=sigma, fwhm_ref=fwhm)


@st.cache_resource(show_spinner=False, max_entries=6)
def cached_animation(v0: float, b: float, w: float, energy: float, sigma: float, fwhm: float,
                     status: str, view: str):
    sim = cached_simulation(v0, b, w, energy, sigma, fwhm)
    return vz.build_animation_html(sim, status, view)


# --------------------------------------------------------------------------
# Button callbacks (they run before the page is redrawn)
# --------------------------------------------------------------------------
def _structure_from_state():
    return pe.Structure(st.session_state["V0"], st.session_state["b"], st.session_state["w"]).snapped()


def reset_all():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.session_state.pop("geom_sig", None)  # forces the energy back to the first resonance


def snap_to_resonance():
    s = _structure_from_state()
    res = cached_resonances(st.session_state["V0"], st.session_state["b"], st.session_state["w"])
    if res:
        st.session_state["E"] = pe.demo_energies(s, res, reference_energy=st.session_state["E"]).resonant


def move_off_resonance():
    s = _structure_from_state()
    res = cached_resonances(st.session_state["V0"], st.session_state["b"], st.session_state["w"])
    if res:
        st.session_state["E"] = pe.demo_energies(s, res, reference_energy=st.session_state["E"]).non_resonant


# --------------------------------------------------------------------------
# Title and introduction
# --------------------------------------------------------------------------
st.title("🌊 Resonant tunneling: how two barriers can let a wave through")
st.markdown(
    """
**Quantum particles such as electrons behave like waves.** A wave that hits a barrier is mostly reflected,
but a little of it leaks through. That leak is called **tunneling**.

Now put **two** barriers close together, with a gap called a **quantum well** between them.
At certain special energies the wave passes through *almost perfectly*, even though each barrier alone
would block most of it. This is **resonant tunneling**. Below you can watch it happen and see why.
"""
)
st.caption(
    "Units: everything is dimensionless with ħ = m = 1, so energies, lengths and times are simply numbers "
    "(this keeps the maths clean; any real system can be mapped onto them by choosing a length scale)."
)

# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)

st.subheader("1 · Choose the structure and the energy")
c1, c2, c3, c4 = st.columns(4)
with c1:
    V0 = st.slider("Barrier height V₀", 2.0, 10.0, step=0.5, key="V0",
                   help="How strongly each barrier repels the wave.")
with c2:
    b = st.slider("Barrier width", 0.2, 1.2, step=0.05, key="b",
                  help="Thicker barriers leak less, which makes the resonances narrower.")
with c3:
    w = st.slider("Well width", 0.8, 3.0, step=0.1, key="w",
                  help="The gap between the barriers. It decides where the resonances sit.")

try:
    structure = pe.Structure(V0, b, w).snapped()
    resonances = cached_resonances(V0, b, w)
    if not resonances:
        raise RuntimeError("No transmission peak was found below the barrier top.")
    demo = pe.demo_energies(structure, resonances)
except Exception as exc:  # noqa: BLE001 - show any failure to the user instead of crashing
    st.error(f"Could not analyse this structure: {exc}. Try different barrier or well settings.")
    st.stop()

geom_sig = (V0, b, w)
if st.session_state.get("geom_sig") != geom_sig:       # new structure -> re-tune to the first resonance
    st.session_state["geom_sig"] = geom_sig
    st.session_state["E"] = demo.resonant
E_MIN, E_MAX = round(0.03 * V0, 3), round(0.98 * V0, 3)
st.session_state["E"] = float(min(max(st.session_state.get("E", demo.resonant), E_MIN), E_MAX))

with c4:
    energy = st.slider("Incident energy E", E_MIN, E_MAX, step=0.001, format="%.3f", key="E",
                       help="The energy of the incoming wave. The default is a detected resonance.")

bc1, bc2, bc3, _ = st.columns([1.2, 1.2, 1.2, 3])
bc1.button("🎯 Snap E to nearest resonance", on_click=snap_to_resonance)
bc2.button("↔ Move E between resonances", on_click=move_off_resonance)
bc3.button("⟲ Reset everything", on_click=reset_all)

status, ref_res, t_sel = pe.resonance_status(structure, energy, resonances)
if status == "on":
    st.success(f"● **On resonance.** At E = {energy:.3f} a steady wave of this exact energy would be transmitted "
               f"with probability **T = {t_sel:.3f}**.")
elif status == "near":
    st.warning(f"● **Close to a resonance.** At E = {energy:.3f} the stationary transmission is "
               f"**T = {t_sel:.3f}**. Use the button to snap to the peak.")
else:
    st.error(f"● **Between resonances.** At E = {energy:.3f} a steady wave of this energy is mostly reflected: "
             f"**T = {t_sel:.3f}**.")

# --------------------------------------------------------------------------
# Simulations (cached)
# --------------------------------------------------------------------------
ref = ref_res if ref_res is not None else demo.resonance
sigma = pe.design_packet_sigma(ref.energy, ref.fwhm)
demo_here = pe.demo_energies(structure, resonances, reference_energy=energy)

try:
    with st.spinner("Simulating the wave (a few seconds, cached afterwards)…"):
        sim = cached_simulation(V0, b, w, energy, sigma, ref.fwhm)
        sim_res = cached_simulation(V0, b, w, demo_here.resonant, sigma, ref.fwhm)
        sim_off = cached_simulation(V0, b, w, demo_here.non_resonant, sigma, ref.fwhm)
except Exception as exc:  # noqa: BLE001
    st.error(f"The simulation failed ({exc}). Try different settings or press Reset.")
    st.stop()

for msg in pe.settings_reliability_warnings(structure, energy, sigma, ref) + sim.warnings:
    st.warning(msg)

# --------------------------------------------------------------------------
# Main animation
# --------------------------------------------------------------------------
st.subheader("2 · Watch the wave hit the double barrier")
view_label = st.radio(
    "What to draw",
    ["Probability density |ψ|²", "The wave itself (real and imaginary parts of ψ)"],
    horizontal=True,
    help="ψ (the wavefunction) is a complex 'amplitude'. It is NOT a probability. "
         "The probability density is |ψ|², its squared size.",
)
view = "density" if view_label.startswith("Probability") else "reim"
if view == "density":
    st.caption("Height of the blue curve = how likely the particle is to be found at that position (|ψ|²). "
               "Ripples in front of barrier 1 are interference between the incoming and reflected waves.")
else:
    st.caption("ψ is a complex wave: blue = real part, orange = imaginary part. Their wiggles show the phase, "
               "which decides whether waves add up or cancel. Probability is not ψ itself but |ψ|².")

try:
    with st.spinner("Rendering the animation (about 10 s the first time for each setting)…"):
        html = cached_animation(V0, b, w, energy, sigma, ref.fwhm, status, view)
    components.html(html, height=int(8.7 * 66) + 95, scrolling=True)
except Exception as exc:  # noqa: BLE001
    st.error(f"The animation could not be drawn ({exc}).")
st.info("▶ Press **play** under the picture (or drag the slider). The five boxes at the top follow the story; "
        "the bars at the bottom show where the probability is at each moment.")

# --------------------------------------------------------------------------
# Spectrum
# --------------------------------------------------------------------------
st.subheader("3 · Which energies get through?")
left, right = st.columns([1.55, 1])
with left:
    fig = vz.figure_spectrum(structure, energy, resonances, status, sim.energy_spread,
                             demo_here.resonant, demo_here.non_resonant)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
with right:
    m1, m2 = st.columns(2)
    m1.metric("T(E) at selected energy", f"{t_sel:.3f}", help="Stationary transmission probability for a single exact energy.")
    m2.metric("Packet actually transmitted", f"{sim.measured_transmitted:.1%}",
              help="Fraction of the whole wave packet found beyond barrier 2 at the end of the run.")
    st.markdown(
        f"""
**Why peaks?** Inside the well the wave bounces back and forth between the barriers. At most energies the
bounces arrive out of step and partly **cancel**. At the resonance energies they arrive **in step** and
**add up** (constructive interference), so the wave builds up in the well and flows out through barrier 2.
The barriers never overlap: it is the interference *between* them that matters.

**Why does the packet get less than T(E)?** A real wave packet is a *mixture* of energies (the shaded band).
Only the part of the band that sits on the narrow peak gets through, so the packet's transmitted
fraction is usually below the peak value, even when T = 1.
"""
    )

# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
st.subheader("4 · Resonant versus non-resonant, same structure, same scale")
fig = vz.figure_comparison(sim_res, sim_off)
st.pyplot(fig, clear_figure=True)
plt.close(fig)
st.markdown(
    f"""
The same wave packet is sent at the same structure at two energies: **E = {sim_res.energy:.3f}** (a resonance, stationary
T = {sim_res.stationary_T:.3f}) and **E = {sim_off.energy:.3f}** (a valley between resonances, T = {sim_off.stationary_T:.3f}).
Grey = reflected part, green = transmitted part. On resonance the waves bouncing in the well are in step, so a large share
gets out the other side; off resonance they are out of step and nearly everything is reflected.
The resonant packet's **reflected part has a notch**: the energies sitting right on the peak went through instead of bouncing back.
"""
)

with st.expander("Four quantities that look similar but are NOT the same"):
    st.markdown(
        """
- **T(E)**, the stationary transmission probability: for one exact energy and an endless, steady wave.
- **|ψ(x,t)|²**, the probability density: how likely the particle is at position x at time t.
- **Probability in a region**: |ψ|² added up over a stretch of space (the bars in the animation).
- **Transmitted fraction of a packet**: the total probability found beyond barrier 2 after the whole packet has passed.
  It equals T(E) averaged over the packet's energies, so it is not generally equal to T(E) at the central energy.
"""
    )

# --------------------------------------------------------------------------
# Numerical checks
# --------------------------------------------------------------------------
with st.expander("Numerical checks (are these results trustworthy?)"):
    diag = pe.transmission_diagnostics(structure)
    gap = abs(sim.predicted_transmitted - sim.measured_transmitted)
    rows = [
        ("Initial total probability", f"{sim.norm_initial:.9f}", abs(sim.norm_initial - 1) < 1e-6),
        ("Final total probability", f"{sim.norm_final:.9f}", abs(sim.norm_final - 1) < 1e-6),
        ("Probability-conservation error (max over the run)", f"{sim.norm_max_error:.1e}", sim.norm_max_error < 1e-8),
        ("Stationary T(E) finite and between 0 and 1",
         f"min {diag['t_min']:.2e}, max {diag['t_max']:.6f}", diag["bounded"]),
        ("Stationary check: T + R = 1 (max deviation)", f"{diag['unitarity_error']:.1e}", diag["unitarity_error"] < 1e-9),
        ("Packet transmission: time-dependent solver vs. theory from T(E)",
         f"{sim.measured_transmitted:.4f} vs {sim.predicted_transmitted:.4f}  (difference {gap:.4f})",
         gap < 0.03 + sim.remaining_inside),
        ("Probability still inside the structure at the end", f"{sim.remaining_inside:.2%}", sim.remaining_inside < 0.02),
    ]
    for label, value, ok in rows:
        st.markdown(f"{'✅' if ok else '⚠️'} **{label}:** {value}")
    st.caption(
        "Conserved probability alone does not prove the physics is right. The last comparison matters more: it "
        "checks the wave-packet simulation (finite differences in time) against the transfer-matrix theory "
        "(a completely different calculation)."
    )

with st.expander("Assumptions and limitations"):
    st.markdown(
        f"""
- One dimension, square barriers, no friction or scattering, particle mass constant everywhere, ħ = m = 1.
- Wave-packet solver: Crank–Nicolson finite differences (grid step {pe.GRID_DX}, time step {pe.GRID_DT}), with hard walls
  at the edges of a box that is large enough that the packet never reaches them.
  Barrier and well widths are rounded to whole grid cells.
- Stationary transmission: exact transfer-matrix method (independent of the solver).
- A resonance is a **narrow** peak, so the packet must be long (narrow in energy) to feel it. Very long packets take
  long to simulate, so the app limits the packet length. For sharp resonances (thick, high barriers) only a small
  fraction of the packet can pass even on resonance; the app warns you when this happens.
- The animation shows the real |ψ|² or ψ from the solver. Only the *zoom* panel uses its own vertical scale (labelled).
  Nothing is multiplied or exaggerated.
"""
    )
