"""Smoke tests for visuals.py (no browser needed).  Run with:  python -m unittest -v"""

import unittest

import matplotlib.pyplot as plt
import numpy as np

import physics_engine as pe
import visuals as vz


class TestVisuals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = pe.Structure().snapped()
        cls.demo = pe.demo_energies(cls.s)
        sigma = pe.design_packet_sigma(cls.demo.resonant, cls.demo.resonance.fwhm)
        cls.res = pe.simulate(cls.s, cls.demo.resonant, sigma, cls.demo.resonance.fwhm, n_frames=24)
        cls.off = pe.simulate(cls.s, cls.demo.non_resonant, sigma, cls.demo.resonance.fwhm, n_frames=24)

    def test_stages_are_ordered_and_inside_range(self):
        for sim in (self.res, self.off):
            st = vz.detect_stages(sim)
            self.assertEqual(len(st), 4)
            self.assertTrue(all(a < b for a, b in zip(st, st[1:])))
            self.assertTrue(0 < st[0] and st[-1] < len(sim.times))

    def test_captions_exist_for_every_stage(self):
        for stage in range(5):
            for status in ("on", "off"):
                self.assertGreater(len(vz.stage_caption(stage, self.res, status)), 20)

    def test_animation_html_is_generated(self):
        for view in ("density", "reim"):
            html = vz.build_animation_html(self.res, "on", view, fps=6, dpi=40)
            self.assertIn("<script", html)
            self.assertGreater(len(html), 10000)

    def test_unknown_view_rejected(self):
        with self.assertRaises(ValueError):
            vz.build_animation_html(self.res, "on", "nonsense")

    def test_static_figures(self):
        res = pe.find_resonances(self.s)
        fig = vz.figure_spectrum(self.s, self.demo.resonant, res, "on", self.res.energy_spread,
                                 self.demo.resonant, self.demo.non_resonant)
        self.assertEqual(len(fig.axes), 1)
        plt.close(fig)
        fig = vz.figure_comparison(self.res, self.off)
        self.assertEqual(len(fig.axes), 3)
        # both snapshot panels must share the same vertical scale
        self.assertEqual(fig.axes[0].get_ylim(), fig.axes[1].get_ylim())
        plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
