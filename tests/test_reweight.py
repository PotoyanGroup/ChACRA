"""
Tests for chacra.reweight — MBAR-pooled response estimators.

Synthetic model: 1-D harmonic "REST2" ladder.  Configuration x is sampled
exactly from p_k(x) ∝ exp(-u_k(x)) with

    u_k(x) = a + b(x) * lam_k + c(x) * sqrt(lam_k),   b(x) = x^2/2,  c(x) = x

so the reduced potential is exactly of REST2 form and every state is a
Gaussian with known mean/variance.  The contact observable is phi(x) = 1[x > t]
whose ensemble average and lambda-derivatives are available in closed form
(Gaussian tail), giving an analytic reference for <phi>(lam), R1 and R2.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from chacra.reweight import (
    OccupancyStore,
    ResponseEstimator,
    mbar_free_energies,
    mbar_log_weights,
    pc_frame,
)

A_CONST = 3.0


def _gauss_params(lam):
    # u = a + lam x^2/2 + sqrt(lam) x  ->  x ~ N(mu, sigma^2), sigma^2 = 1/lam, mu = -1/sqrt(lam)
    return -1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)


def _phi_mean_analytic(lam, thresholds):
    mu, sig = _gauss_params(lam)
    return norm.sf((np.asarray(thresholds) - mu) / sig)


def _make_store(n_frames=6000, seed=0):
    rng = np.random.default_rng(seed)
    lambdas = np.array([1.0, 0.8, 0.64, 0.512, 0.41])
    temps = 300.0 / lambdas
    thresholds = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0])
    K, C = len(lambdas), len(thresholds)
    phi = np.zeros((K, n_frames, C), dtype=np.uint8)
    u = np.zeros((K, n_frames, K))
    for s, lam in enumerate(lambdas):
        mu, sig = _gauss_params(lam)
        x = rng.normal(mu, sig, n_frames)
        phi[s] = (x[:, None] > thresholds[None, :]).astype(np.uint8)
        u[s] = A_CONST + 0.5 * x[:, None] ** 2 * lambdas[None, :] + x[:, None] * np.sqrt(lambdas)[None, :]
    return OccupancyStore(phi=phi, u=u, contacts=[f"c{i}" for i in range(C)],
                          lambdas=lambdas, temps=temps), thresholds


class TestEnergyDecomposition:
    def test_recovers_a_b_c_exactly(self):
        store, _ = _make_store(n_frames=500)
        a, b, c, resid = store.energy_decomposition()
        assert resid < 1e-8
        np.testing.assert_allclose(a, A_CONST, atol=1e-8)
        # b = x^2/2, c = x  ->  b == c^2 / 2
        np.testing.assert_allclose(b, 0.5 * c ** 2, atol=1e-8)


class TestMBAR:
    def test_free_energies_match_gaussian_normalisation(self):
        store, _ = _make_store(n_frames=20000, seed=1)
        u_kn = store.pooled_u_kn()
        N_k = np.full(store.n_states, store.n_frames)
        f = mbar_free_energies(u_kn, N_k, use_pymbar=False)
        # Z_k = exp(-a) * sqrt(2 pi / lam) * exp(1/2)  ->  f_k - f_0 = 0.5 ln(lam_k)
        expected = 0.5 * np.log(store.lambdas)
        np.testing.assert_allclose(f, expected, atol=0.02)

    def test_weights_normalised_and_reweighted_mean(self):
        store, thresholds = _make_store(n_frames=20000, seed=2)
        u_kn = store.pooled_u_kn()
        N_k = np.full(store.n_states, store.n_frames)
        f = mbar_free_energies(u_kn, N_k, use_pymbar=False)
        lam = 0.7  # not a sampled state
        a, b, c, _ = store.energy_decomposition()
        logw = mbar_log_weights(a + b * lam + c * np.sqrt(lam), u_kn, f, N_k)
        assert abs(np.exp(logw).sum() - 1) < 1e-10
        phi_pool = store.phi.reshape(-1, store.n_contacts).astype(float)
        est = np.exp(logw) @ phi_pool
        np.testing.assert_allclose(est, _phi_mean_analytic(lam, thresholds), atol=0.01)


class TestResponseEstimator:
    @pytest.fixture(scope="class")
    def est(self):
        store, thresholds = _make_store(n_frames=40000, seed=3)
        return ResponseEstimator(store, use_pymbar=False), thresholds

    def test_R1_matches_analytic_derivative(self, est):
        e, thresholds = est
        lam = 0.75
        h = 1e-3
        analytic = (_phi_mean_analytic(lam + h, thresholds) - _phi_mean_analytic(lam - h, thresholds)) / (2 * h)
        r = e.at(lam)
        np.testing.assert_allclose(r.phi_mean, _phi_mean_analytic(lam, thresholds), atol=0.01)
        np.testing.assert_allclose(r.R1, analytic, atol=0.03)
        assert r.ess > 0.2 * e.store.n_states * e.store.n_frames

    def test_R2_matches_analytic_second_derivative(self, est):
        e, thresholds = est
        lam = 0.75
        h = 2e-2
        f = lambda l: _phi_mean_analytic(l, thresholds)
        analytic = (f(lam + h) - 2 * f(lam) + f(lam - h)) / h ** 2
        r = e.at(lam)
        # third-cumulant estimates are noisy; check sign/magnitude agreement
        assert np.all(np.abs(r.R2 - analytic) <= 0.25 + 0.1 * np.abs(analytic)), (r.R2, analytic)

    def test_pooled_beats_single_state(self, est):
        e, thresholds = est
        k = 2
        lam = e.store.lambdas[k]
        h = 1e-3
        analytic = (_phi_mean_analytic(lam + h, thresholds) - _phi_mean_analytic(lam - h, thresholds)) / (2 * h)
        pooled = e.at(lam).R1
        single = e.single_state(k).R1
        assert np.abs(pooled - analytic).max() <= np.abs(single - analytic).max() + 1e-3

    def test_bootstrap_errors_are_sane(self, est):
        e, thresholds = est
        r = e.bootstrap(0.75, n_boot=10, block=100)
        assert r.R1_err.shape == r.R1.shape
        assert np.all(r.R1_err > 0)
        # the error should be small relative to the largest response
        assert r.R1_err.max() < 0.2 * np.abs(r.R1).max() + 1e-3

    def test_ladder_slope_shape(self, est):
        e, _ = est
        assert e.ladder_slope(2).shape == (e.store.n_contacts,)
        assert e.ladder_slope(0).shape == (e.store.n_contacts,)


class TestStoreIO:
    def test_roundtrip(self, tmp_path):
        store, _ = _make_store(n_frames=100)
        store.meta = {"run": 1}
        p = tmp_path / "occ.npz"
        store.save(p)
        back = OccupancyStore.load(p)
        np.testing.assert_array_equal(back.phi, store.phi)
        np.testing.assert_allclose(back.u, store.u)
        assert back.contacts == store.contacts
        assert back.meta == {"run": 1}
        assert back.freqs().shape == (store.n_states, store.n_contacts)


def test_pc_frame_fraction():
    L = np.eye(5)[:, :3]
    v = np.array([1.0, 1.0, 0.0, 2.0, 0.0])
    out = pc_frame(v, L, n_pcs=3)
    assert out["frac_in_pc1_3"] == pytest.approx(2.0 / 6.0)
    assert out["cos_pc1"] == pytest.approx(1 / np.sqrt(6))
