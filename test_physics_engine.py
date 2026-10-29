"""
Automated numerical tests for physics_engine.py

Run with:   python -m unittest -v          (or:  python -m pytest -v)

The tests cover normalisation, probability conservation, boundedness of the
stationary transmission, limiting cases with known analytic answers,
resonance detection against an independent finite-well calculation, and the
agreement of two independent methods (time-dependent solver vs transfer matrix).
"""

import unittest

import numpy as np
from scipy.optimize import brentq

import physics_engine as pe


def analytic_single_barrier(E, v0, width):
    """Textbook transmission through ONE rectangular barrier (E < V0)."""
    kappa = np.sqrt(2.0 * (v0 - E))
    return 1.0 / (1.0 + (v0 ** 2 * np.sinh(kappa * width) ** 2) / (4.0 * E * (v0 - E)))


class TestStationaryTransmission(unittest.TestCase):
    def test_bounded_and_finite(self):
        for v0, b, w in [(6, 0.4, 1.5), (2, 0.2, 0.8), (10, 1.2, 3.0), (4, 0.7, 2.2)]:
            s = pe.Structure(v0, b, w).snapped()
            E = np.linspace(1e-3, 1.6 * v0, 20000)
            T, R, _ = pe.scattering(E, s)
            self.assertTrue(np.all(np.isfinite(T)))
            self.assertTrue(np.all(T >= 0.0) and np.all(T <= 1.0 + 1e-9))
            # independent consistency: transmitted + reflected probability = 1
            self.assertLess(np.max(np.abs(T + R - 1.0)), 1e-9)

    def test_no_barrier_gives_full_transmission(self):
        s = pe.Structure(1e-9, 0.4, 1.5)
        E = np.linspace(0.1, 5.0, 50)
        self.assertTrue(np.allclose(pe.transmission(E, s), 1.0, atol=1e-6))

    def test_vanishing_well_equals_single_thick_barrier(self):
        # Well width -> 0 merges the two barriers into one barrier of width 2b.
        v0, b = 5.0, 0.35
        s = pe.Structure(v0, b, 1e-9)
        E = np.array([0.5, 1.0, 2.0, 3.5, 4.5])
        self.assertTrue(np.allclose(pe.transmission(E, s), analytic_single_barrier(E, v0, 2 * b), rtol=1e-6))

    def test_thick_barriers_block_transmission_off_resonance(self):
        s = pe.Structure(8.0, 2.0, 1.5)
        self.assertLess(float(pe.transmission(0.3, s)[0]), 1e-4)

    def test_energy_must_be_positive(self):
        with self.assertRaises(ValueError):
            pe.transmission(0.0, pe.Structure())
        with self.assertRaises(ValueError):
            pe.Structure(-1.0, 0.4, 1.5)

    def test_smooth_resonance_function_matches_transmission(self):
        # T = 1/(1+h^2) is an independent formula for a symmetric structure.
        s = pe.Structure().snapped()
        E = np.linspace(0.05, 5.9, 2000)
        h = pe.resonance_function(E, s)
        self.assertTrue(np.allclose(pe.transmission(E, s), 1.0 / (1.0 + h ** 2), atol=1e-10))


class TestResonanceDetection(unittest.TestCase):
    def test_default_configuration_has_known_resonance(self):
        s = pe.Structure().snapped()
        res = pe.find_resonances(s)
        self.assertGreaterEqual(len(res), 2)
        first = res[0]
        self.assertGreater(first.t_peak, 0.9999)             # symmetric structure -> T = 1 at the peak
        self.assertTrue(0.9 < first.energy < 1.25)           # known location for the defaults (~1.068)
        # independent brute-force check: the maximum of a fine T(E) scan sits at the same energy
        E = np.linspace(0.5, 1.6, 200001)
        e_scan = E[np.argmax(pe.transmission(E, s))]
        self.assertAlmostEqual(first.energy, e_scan, delta=2e-5)
        self.assertGreater(first.fwhm, 0.05)

    def test_resonance_matches_finite_well_bound_state(self):
        # With very thick barriers the first resonance approaches the ground state of a
        # single finite square well of width w and depth V0 (independent analytic condition).
        v0, w, b = 6.0, 1.5, 3.0
        f = lambda E: np.sqrt(2 * E) * np.tan(np.sqrt(2 * E) * w / 2) - np.sqrt(2 * (v0 - E))
        e_well = brentq(f, 1e-3, (np.pi / w) ** 2 / 2 - 1e-6)
        res = pe.find_resonances(pe.Structure(v0, b, w).snapped())
        self.assertAlmostEqual(res[0].energy, e_well, delta=0.01 * e_well)
        self.assertGreater(res[0].t_peak, 0.999)

    def test_very_narrow_resonance_is_not_missed(self):
        s = pe.Structure(10.0, 1.0, 3.0).snapped()
        res = pe.find_resonances(s)
        self.assertGreaterEqual(len(res), 3)
        self.assertLess(res[0].fwhm, 1e-3)
        self.assertAlmostEqual(float(pe.transmission(res[0].energy, s)[0]), 1.0, places=6)

    def test_demo_energies_are_useful(self):
        s = pe.Structure().snapped()
        d = pe.demo_energies(s)
        self.assertGreater(d.t_resonant, 0.99)
        self.assertLess(d.t_non_resonant, 0.2)
        self.assertGreater(d.t_resonant, 5 * d.t_non_resonant)
        label, _, _ = pe.resonance_status(s, d.resonant)
        self.assertEqual(label, "on")
        label, _, _ = pe.resonance_status(s, d.non_resonant)
        self.assertEqual(label, "off")


class TestWavePacketSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = pe.Structure().snapped()
        cls.demo = pe.demo_energies(cls.s)
        cls.fwhm = cls.demo.resonance.fwhm
        cls.sigma = pe.design_packet_sigma(cls.demo.resonant, cls.fwhm)
        cls.sim_res = pe.simulate(cls.s, cls.demo.resonant, cls.sigma, cls.fwhm)
        cls.sim_off = pe.simulate(cls.s, cls.demo.non_resonant, cls.sigma, cls.fwhm)

    def test_initial_packet_is_normalised(self):
        x = np.arange(-200, 200, pe.GRID_DX)
        psi = pe.gaussian_packet(x, -40.0, 1.5, 6.0, pe.GRID_DX)
        self.assertAlmostEqual(float(np.sum(np.abs(psi) ** 2) * pe.GRID_DX), 1.0, places=12)
        self.assertAlmostEqual(self.sim_res.norm_initial, 1.0, places=9)

    def test_probability_is_conserved(self):
        for sim in (self.sim_res, self.sim_off):
            self.assertLess(sim.norm_max_error, 1e-9)
            self.assertLess(sim.conservation_error, 1e-9)
            total = sim.p_before + sim.p_inside + sim.p_after
            self.assertTrue(np.allclose(total, sim.norm_initial, atol=1e-9))
            self.assertTrue(np.all(np.isfinite(sim.frames)))

    def test_regions_are_probabilities(self):
        for sim in (self.sim_res, self.sim_off):
            for arr in (sim.p_before, sim.p_inside, sim.p_after):
                self.assertTrue(np.all(arr >= -1e-12) and np.all(arr <= 1 + 1e-9))

    def test_two_independent_methods_agree(self):
        # Time-dependent solver vs. T(E) averaged over the packet's energy distribution.
        for sim in (self.sim_res, self.sim_off):
            total_out = sim.measured_transmitted + sim.remaining_inside   # leftover still draining
            self.assertAlmostEqual(sim.measured_transmitted, sim.predicted_transmitted,
                                   delta=0.02 + sim.remaining_inside)
            self.assertLess(abs(total_out - sim.predicted_transmitted), 0.02)

    def test_resonance_transmits_much_more_than_off_resonance(self):
        self.assertGreater(self.sim_res.measured_transmitted, 0.4)
        self.assertLess(self.sim_off.measured_transmitted, 0.1)
        self.assertGreater(self.sim_res.measured_transmitted, 8 * self.sim_off.measured_transmitted)

    def test_packet_is_not_perfectly_transmitted_even_on_resonance(self):
        # T(E) = 1 at the peak, but a packet with finite energy spread cannot transmit 100 %.
        self.assertGreater(self.sim_res.stationary_T, 0.99)
        self.assertLess(self.sim_res.measured_transmitted, 0.9)

    def test_free_particle_limit(self):
        s = pe.Structure(1e-6, 0.4, 1.5).snapped()
        sim = pe.simulate(s, 1.0, sigma=5.0)
        self.assertGreater(sim.measured_transmitted, 0.995)

    def test_thick_barrier_limit(self):
        s = pe.Structure(10.0, 1.2, 1.5).snapped()
        sim = pe.simulate(s, 1.0, sigma=5.0)
        self.assertLess(sim.measured_transmitted, 1e-3)
        self.assertGreater(sim.measured_reflected, 0.99)

    def test_no_warnings_for_default_demo(self):
        self.assertEqual(self.sim_res.warnings, [])
        self.assertEqual(self.sim_off.warnings, [])

    def test_sharp_resonance_triggers_reliability_warning(self):
        s = pe.Structure(10.0, 1.0, 3.0).snapped()
        res = pe.find_resonances(s)[0]
        sigma = pe.design_packet_sigma(res.energy, res.fwhm)
        msgs = pe.settings_reliability_warnings(s, res.energy, sigma, res)
        self.assertTrue(len(msgs) == 1)

    def test_invalid_energy_rejected(self):
        with self.assertRaises(ValueError):
            pe.simulate(self.s, -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
