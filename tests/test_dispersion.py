"""
tests/test_dispersion.py
========================
Unit tests for src/dispersion.py

Run with:
    pytest tests/test_dispersion.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dispersion import (
    angular_frequency, phase_speed, group_speed,
    period_from_k, period_from_wavelength, k_from_period,
)

G  = 9.81
TOL = 1e-4   # relative tolerance for floating-point comparisons


# ---------------------------------------------------------------------------
# angular_frequency
# ---------------------------------------------------------------------------

class TestAngularFrequency:
    def test_deep_water(self):
        """Deep water: ω² ≈ g·k  (tanh → 1)."""
        k  = 0.1    # rad/m  → λ ≈ 63 m,  kh >> 1 for h = 1000 m
        h  = 1000.0
        om = angular_frequency(k, h, G)
        assert abs(om**2 - G * k) / (G * k) < 1e-6

    def test_shallow_water(self):
        """Shallow water: ω² ≈ g·k²·h  → c = sqrt(g·h)."""
        k  = 0.001   # rad/m  → λ ≈ 6 km,  kh << 1 for h = 2 m
        h  = 2.0
        om = angular_frequency(k, h, G)
        c_expected = np.sqrt(G * h)
        c_computed = om / k
        assert abs(c_computed - c_expected) / c_expected < 0.01

    def test_array_input(self):
        k  = np.array([0.05, 0.10, 0.20])
        h  = 50.0
        om = angular_frequency(k, h, G)
        assert om.shape == k.shape
        assert np.all(om > 0)


# ---------------------------------------------------------------------------
# phase_speed & group_speed
# ---------------------------------------------------------------------------

class TestSpeeds:
    def test_deep_water_phase_speed(self):
        """Deep water: c = g/ω = sqrt(g/k)."""
        k  = 0.1;  h = 5000.0
        c  = phase_speed(k, h, G)
        assert abs(c - np.sqrt(G / k)) / np.sqrt(G / k) < 1e-5

    def test_group_leq_phase(self):
        """Group speed ≤ phase speed everywhere (finite depth)."""
        for k in [0.01, 0.05, 0.1, 0.5]:
            for h in [5.0, 50.0, 500.0]:
                cg = group_speed(k, h, G)
                c  = phase_speed(k, h, G)
                assert cg <= c + 1e-10, f"cg > c for k={k}, h={h}"

    def test_deep_water_cg_half_c(self):
        """Deep water: c_g = c/2."""
        k  = 0.1;  h = 5000.0
        cg = group_speed(k, h, G)
        c  = phase_speed(k, h, G)
        assert abs(cg - 0.5 * c) / c < 1e-4


# ---------------------------------------------------------------------------
# period_from_k
# ---------------------------------------------------------------------------

class TestPeriodFromK:
    def test_deep_regime(self):
        k = 0.1;  h = 5000.0
        T, regime = period_from_k(k, h, G)
        assert regime == "deep"
        assert T > 0

    def test_shallow_regime(self):
        k = 0.001;  h = 2.0   # kh ≈ 0.002 << 0.1
        T, regime = period_from_k(k, h, G)
        assert regime == "shallow"

    def test_intermediate_regime(self):
        k = 0.05;  h = 38.0   # kh ≈ 1.9 — neither deep nor shallow
        T, regime = period_from_k(k, h, G)
        assert regime == "intermediate"

    def test_known_value(self):
        """100 m wavelength in 38 m water — manually verified."""
        lam = 100.0
        h   = 38.0
        k   = 2.0 * np.pi / lam
        T, _ = period_from_k(k, h, G)
        # Sanity: period should be ~8–9 s for 100 m wave in 38 m water
        assert 6.0 < T < 12.0


# ---------------------------------------------------------------------------
# period_from_wavelength
# ---------------------------------------------------------------------------

class TestPeriodFromWavelength:
    def test_equivalence(self):
        lam = 200.0;  h = 38.0
        T1, r1 = period_from_wavelength(lam, h, G)
        k      = 2.0 * np.pi / lam
        T2, r2 = period_from_k(k, h, G)
        assert abs(T1 - T2) < 1e-10
        assert r1 == r2


# ---------------------------------------------------------------------------
# k_from_period  (inverse solve)
# ---------------------------------------------------------------------------

class TestKFromPeriod:
    @pytest.mark.parametrize("T,h", [
        (8.0, 38.0),
        (12.0, 200.0),
        (5.0, 10.0),
        (20.0, 5000.0),
    ])
    def test_round_trip(self, T, h):
        """k_from_period followed by period_from_k should recover T."""
        k        = k_from_period(T, h, G)
        T_back, _ = period_from_k(k, h, G)
        assert abs(T_back - T) / T < TOL, (
            f"Round-trip failed for T={T}, h={h}: got {T_back:.4f}")

    def test_deep_water_limit(self):
        """Very deep water: k ≈ ω²/g."""
        T     = 10.0;  h = 50_000.0
        k     = k_from_period(T, h, G)
        omega = 2.0 * np.pi / T
        k_deep = omega**2 / G
        assert abs(k - k_deep) / k_deep < 1e-5
