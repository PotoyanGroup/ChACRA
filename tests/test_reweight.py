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
    # a modern store records the temperature its u was scored at (no legacy rescale on load)
    return OccupancyStore(phi=phi, u=u, contacts=[f"c{i}" for i in range(C)],
                          lambdas=lambdas, temps=temps,
                          meta={"u_temperature": float(temps[0]),
                                "u_source_temperature": float(temps[0])}), thresholds


class TestEnergyDecomposition:
    def test_recovers_a_b_c_exactly(self):
        store, _ = _make_store(n_frames=500)
        a, b, c, resid = store.energy_decomposition()
        assert resid < 1e-8
        np.testing.assert_allclose(a, A_CONST, atol=1e-8)
        # b = x^2/2, c = x  ->  b == c^2 / 2
        np.testing.assert_allclose(b, 0.5 * c ** 2, atol=1e-8)

    def test_uniform_rescale_scales_coefficients_and_keeps_residual(self):
        # the femto temperature correction multiplies u by a constant; the
        # REST2 decomposition is linear in u, so (a, b, c) scale and the
        # residual stays at machine precision.
        store, _ = _make_store(n_frames=500)
        a, b, c, resid = store.energy_decomposition()
        factor = 298.15 / 290.0
        scaled = OccupancyStore(phi=store.phi, u=store.u * factor, contacts=store.contacts,
                                lambdas=store.lambdas, temps=store.temps)
        a2, b2, c2, resid2 = scaled.energy_decomposition()
        assert resid2 < 1e-8
        np.testing.assert_allclose(a2, a * factor, atol=1e-8)
        np.testing.assert_allclose(b2, b * factor, atol=1e-8)
        np.testing.assert_allclose(c2, c * factor, atol=1e-8)


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
        # a store that records the temperature its u was scored at round-trips untouched
        store.meta = {"run": 1, "u_temperature": 300.0, "u_source_temperature": 300.0}
        p = tmp_path / "occ.npz"
        store.save(p)
        back = OccupancyStore.load(p)
        np.testing.assert_array_equal(back.phi, store.phi)
        np.testing.assert_allclose(back.u, store.u)
        assert back.contacts == store.contacts
        assert back.meta == {"run": 1, "u_temperature": 300.0, "u_source_temperature": 300.0}
        assert back.freqs().shape == (store.n_states, store.n_contacts)

    def test_legacy_meta_is_rescaled_on_load(self, tmp_path):
        # no u_temperature on record: femto scored u at 298.15 K while MD ran at temps[0]
        store, _ = _make_store(n_frames=100)
        store.meta = {"run": 1}
        p = tmp_path / "legacy.npz"
        store.save(p)
        back = OccupancyStore.load(p, verbose=False)
        np.testing.assert_allclose(back.u, store.u * (298.15 / 300.0))
        assert back.meta["legacy_rescaled"] is True
        raw = OccupancyStore.load(p, legacy_u_temperature=None)
        np.testing.assert_allclose(raw.u, store.u)


def test_pc_frame_fraction():
    L = np.eye(5)[:, :3]
    v = np.array([1.0, 1.0, 0.0, 2.0, 0.0])
    out = pc_frame(v, L, n_pcs=3)
    assert out["frac_in_pc1_3"] == pytest.approx(2.0 / 6.0)
    assert out["cos_pc1"] == pytest.approx(1 / np.sqrt(6))


class TestSingleStateAndPooling:
    def test_single_state_response_matches_estimator_and_analytic(self):
        from chacra.reweight import single_state_response

        store, thresholds = _make_store(n_frames=40000, seed=5)
        k = 2
        lam = store.lambdas[k]
        r = single_state_response(store, k, n_boot=5, block=100)
        e = ResponseEstimator(store, use_pymbar=False).single_state(k)
        np.testing.assert_allclose(r.R1, e.R1, atol=1e-10)
        np.testing.assert_allclose(r.R2, e.R2, atol=1e-10)
        np.testing.assert_allclose(r.phi_mean, e.phi_mean, atol=1e-10)
        h = 1e-3
        analytic = (_phi_mean_analytic(lam + h, thresholds) - _phi_mean_analytic(lam - h, thresholds)) / (2 * h)
        np.testing.assert_allclose(r.phi_mean, _phi_mean_analytic(lam, thresholds), atol=0.01)
        np.testing.assert_allclose(r.R1, analytic, atol=0.05)
        assert r.R1_err.shape == r.R1.shape and np.all(r.R1_err >= 0)
        assert r.R2_err.shape == r.R2.shape
        assert r.lam == pytest.approx(lam) and r.ess == store.n_frames

    def test_single_state_rejects_bad_state(self):
        from chacra.reweight import single_state_response

        store, _ = _make_store(n_frames=200)
        with pytest.raises(IndexError):
            single_state_response(store, store.n_states)

    def test_ladder_slope_module_function_matches_method(self):
        from chacra.reweight import ladder_slope

        store, _ = _make_store(n_frames=2000, seed=8)
        e = ResponseEstimator(store, use_pymbar=False)
        for k in range(store.n_states):
            np.testing.assert_allclose(ladder_slope(store, k), e.ladder_slope(k))

    def test_concat_union_frames_and_pooled_estimate(self):
        s1, thresholds = _make_store(n_frames=3000, seed=6)
        s2, _ = _make_store(n_frames=2000, seed=7)
        # second run lacks the last contact -> union with zero-fill
        s2b = OccupancyStore(phi=s2.phi[:, :, :-1], u=s2.u, contacts=s2.contacts[:-1],
                             lambdas=s2.lambdas, temps=s2.temps, meta=dict(s2.meta, run=2))
        s1.meta["run"] = 1
        pooled = OccupancyStore.concat([s1, s2b])
        assert pooled.contacts == s1.contacts
        assert pooled.n_frames == 5000 and pooled.n_states == s1.n_states
        assert pooled.meta["segments"] == [0, 3000, 5000]
        assert pooled.meta["runs"] == [1, 2]
        f1, f2, fp = s1.freqs().to_numpy(), s2b.freqs().to_numpy(), pooled.freqs().to_numpy()
        np.testing.assert_allclose(fp[:, :-1], (3000 * f1[:, :-1] + 2000 * f2) / 5000)
        np.testing.assert_allclose(fp[:, -1], 3000 * f1[:, -1] / 5000)
        np.testing.assert_allclose(pooled.u[:, :3000], s1.u)
        np.testing.assert_allclose(pooled.u[:, 3000:], s2b.u)
        # the pooled MBAR estimate of the shared contacts is still right
        e = ResponseEstimator(pooled, use_pymbar=False)
        lam = 0.75
        h = 1e-3
        analytic = (_phi_mean_analytic(lam + h, thresholds) - _phi_mean_analytic(lam - h, thresholds)) / (2 * h)
        r = e.at(lam)
        np.testing.assert_allclose(r.R1[:-1], analytic[:-1], atol=0.06)

    def test_concat_rejects_mismatched_ladders_or_temperatures(self):
        s1, _ = _make_store(n_frames=100)
        s2, _ = _make_store(n_frames=100)
        s2.lambdas = s2.lambdas * 0.9
        with pytest.raises(ValueError):
            OccupancyStore.concat([s1, s2])
        s3, _ = _make_store(n_frames=100)
        s3.meta["u_temperature"] = 290.0
        with pytest.raises(ValueError):
            OccupancyStore.concat([s1, s3])

    def test_parse_runs(self, tmp_path):
        from chacra.reweight import parse_runs

        assert parse_runs("4-7") == [4, 5, 6, 7]
        assert parse_runs("4,6,7") == [4, 6, 7]
        assert parse_runs("5") == [5]
        assert parse_runs([7, 4, 4]) == [4, 7]
        assert parse_runs(None) == []
        for r in (1, 3):
            d = tmp_path / "contact_output" / f"run_{r}"
            d.mkdir(parents=True)
            (d / "occupancy.npz").write_bytes(b"")
        assert parse_runs("all", tmp_path) == [1, 3]
        with pytest.raises(ValueError):
            parse_runs("7-4")


class TestPymbarAndBootstrapGuards:
    """pymbar stays off JAX unless asked (the CUDA plugin segfaulted on A100 nodes),
    and a bootstrap that collapses to one block says so."""

    def test_pymbar_runs_without_jax_by_default(self, monkeypatch):
        import os
        from chacra.reweight import _configure_pymbar_env

        monkeypatch.delenv("PYMBAR_DISABLE_JAX", raising=False)
        monkeypatch.delenv("CHACRA_PYMBAR_JAX", raising=False)
        _configure_pymbar_env()
        assert os.environ["PYMBAR_DISABLE_JAX"] == "1"

    def test_user_can_opt_into_jax(self, monkeypatch):
        import os
        from chacra.reweight import _configure_pymbar_env

        monkeypatch.delenv("PYMBAR_DISABLE_JAX", raising=False)
        monkeypatch.setenv("CHACRA_PYMBAR_JAX", "1")
        _configure_pymbar_env()
        assert "PYMBAR_DISABLE_JAX" not in os.environ

    def test_explicit_user_setting_is_kept(self, monkeypatch):
        import os
        from chacra.reweight import _configure_pymbar_env

        monkeypatch.setenv("PYMBAR_DISABLE_JAX", "0")
        monkeypatch.delenv("CHACRA_PYMBAR_JAX", raising=False)
        _configure_pymbar_env()
        assert os.environ["PYMBAR_DISABLE_JAX"] == "0"

    def test_single_block_bootstrap_warns(self):
        from chacra.reweight import ResponseEstimator, single_state_response

        store, _ = _make_store(n_frames=40)
        with pytest.warns(RuntimeWarning, match="single bootstrap block"):
            single_state_response(store, 0, n_boot=2, block=100)
        with pytest.warns(RuntimeWarning, match="single bootstrap block"):
            ResponseEstimator(store, use_pymbar=False).bootstrap(1.0, n_boot=2, block=100)

    def test_enough_blocks_do_not_warn(self):
        import warnings
        from chacra.reweight import single_state_response

        store, _ = _make_store(n_frames=400)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            r = single_state_response(store, 0, n_boot=2, block=50)
        assert r.R1_err is not None and np.all(np.isfinite(r.R1_err))


def test_residue_pairs_from_tsv_matches_unique(tmp_path):
    """The streaming parser gives np.unique's sorted names and inverse codes."""
    from chacra.reweight import _residue_pairs_from_tsv

    tsv = tmp_path / "c.tsv"
    tsv.write_text(
        "# total_frames:3 beg:0 end:2 stride:1\n# Columns: frame, type, atom_1, atom_2, d\n"
        "0\thp\tA:LEU:5:CD1\tA:VAL:9:CG1\t3.9\n"
        "0\tsb\tA:GLU:2:OE1\tA:ARG:7:NH1\t3.1\n"
        "1\thp\tA:VAL:9:CG2\tA:LEU:5:CD2\t4.0\n"
        "\n"
        "2\thb\tA:ARG:7:NE\tA:GLU:2:OE2\t2.9\n"
    )
    frames, uniq, inv = _residue_pairs_from_tsv(tsv)
    assert frames.tolist() == [0, 0, 1, 2]
    assert uniq.tolist() == ["A:ARG:7-A:GLU:2", "A:LEU:5-A:VAL:9"]
    assert inv.tolist() == [1, 0, 1, 0] and inv.dtype == np.int32
