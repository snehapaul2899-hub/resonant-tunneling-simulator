# Quantum Resonant Tunneling Simulator

A one-page, beginner-friendly Streamlit app that shows **why a double-barrier structure transmits
strongly at certain energies** (resonant tunneling), using a real wave-packet simulation and an
independent transfer-matrix calculation.

## Setup and run (Windows PowerShell, VS Code terminal)

Open the project folder in VS Code (`File > Open Folder`), open a terminal (`Ctrl+``), then run these in order:

```powershell
# 1. Create a virtual environment (Python 3.12; use "python" instead of "py -3.12" if you only have one Python)
py -3.12 -m venv .venv

# 2. Activate it (if PowerShell blocks scripts, first run:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the automated tests (takes ~15 s; expect "OK")
python -m unittest -v

# 5. Launch the app (opens http://localhost:8501)
streamlit run app.py
```

* On the very first launch Streamlit may ask for an e-mail address in the terminal: just press **Enter**.
* **Stop** the app: click the terminal and press `Ctrl+C`.
* **Restart**: with the environment still active run `streamlit run app.py` again. In a new terminal first run
  `.\.venv\Scripts\Activate.ps1`. After editing code you can also press **R** in the browser to rerun.
* The first load takes ~15 s (three simulations + animation); results are cached afterwards, and each new
  setting costs a few seconds.

## Files

| File | Purpose |
|---|---|
| `app.py` | The single-page Streamlit interface |
| `physics_engine.py` | Transfer-matrix transmission, resonance detection, Crank-Nicolson wave-packet solver, diagnostics |
| `visuals.py` | Matplotlib animation (embedded as HTML), spectrum plot, comparison plot |
| `test_physics_engine.py`, `test_visuals.py` | Automated tests (`python -m unittest -v`) |
| `requirements.txt` | Dependencies |

## What you see

1. **Controls**: barrier height, barrier width, well width, incident energy (default = a detected resonance).
   Buttons: snap to resonance, move between resonances, reset.
2. **Animation** with five labelled stages (incoming, barrier 1, in the well, transmission, result), plain-language
   captions that follow the *actual* simulation data, region labels, and bars giving the probability in each region.
   A switch changes between |ψ|² and the real/imaginary parts of ψ (which show phase and interference).
3. **Transmission spectrum T(E)** with automatically detected resonances, the selected energy, and the energy
   range of the wave packet.
4. **Resonant vs. non-resonant** comparison on identical axes, with transmitted fractions side by side.
5. **Numerical checks** and **assumptions** (collapsed by default).

## The physics in brief

* Units: ħ = m = 1, so k = √(2E) and the group velocity is k. All lengths, energies and times are dimensionless.
* Potential: two rectangular barriers (height V₀, width b) separated by a well of width w; V = 0 elsewhere.
* **Stationary transmission T(E)**: exact transfer matrix for a plane wave of fixed energy.
  For a symmetric structure T = 1/(1 + h²) with h(E) smooth, so resonances (T = 1) are exact zeros of h and cannot
  be missed by a coarse energy grid, however narrow. Peak widths (FWHM) are computed from T directly.
* **Time-dependent solution**: Gaussian packet evolved with Crank–Nicolson,
  (1 + iΔt H/2)ψⁿ⁺¹ = (1 − iΔt H/2)ψⁿ, which is unconditionally stable and exactly unitary for the discrete
  Hamiltonian, so probability is conserved to round-off.
* **Independent cross-check**: the packet's transmitted fraction predicted from T(E) averaged over the packet's
  energy distribution (FFT of the initial state) is compared with the simulation.
* **Why resonance**: waves that tunnel into the well bounce between the barriers. At resonance energies the repeated
  round trips add in phase (constructive interference), probability builds up in the well and leaves through
  barrier 2; elsewhere the contributions are out of phase and cancel. The barriers do not overlap; the effect comes
  from interference in the region *between* them.
* **Four different quantities** (kept distinct in the app): T(E); the density |ψ(x,t)|²; probability integrated over
  a region; the transmitted fraction of a packet. A packet's fraction is T(E) averaged over its energies, so it is
  generally below the peak T = 1.

## Packet design trade-off

A packet of spatial width σ has energy spread ΔE ≈ k₀/(2σ). To feel a resonance of width Γ, ΔE should be of
order Γ or smaller, which needs σ ≈ k₀/Γ. Narrow resonances need long packets and long runs, so σ is limited to
4–8 and the app warns when a resonance is much sharper than the packet. With the defaults
(V₀ = 6, b = 0.4, w = 1.5; first resonance at E ≈ 1.068, width ≈ 0.14, σ = 8) about **57 %** of the packet is
transmitted on resonance versus **4 %** off resonance, although T(E) = 1.00 at the peak.

## Assumptions and limitations

* 1D, square barriers, constant mass, no dissipation or electron–electron interaction.
* Grid step 0.025 and time step 0.02; widths are rounded to whole grid cells (all slider steps are multiples of 0.025).
  The same rounded structure is used for the transfer-matrix calculation.
* Hard-wall box boundaries, sized so the packet never reaches them.
* Resonances are searched below the barrier top (E < V₀).
* The zoom panel of the animation uses its own vertical scale (labelled). No value is ever multiplied or exaggerated.
* Probability conservation by itself does not validate the physics; the transfer-matrix comparison and the
  analytic limit tests (single barrier, finite well, free particle, thick barrier) address that.
* Changing barrier/well sliders re-tunes E to the first resonance; changing only E keeps your energy.
