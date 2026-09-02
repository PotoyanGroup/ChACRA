"""
Tests for femto frame/state alignment and the union alignment in
compare_to_pca (Phase 1 fixes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chacra.trajectories.process_hremd import saved_frame_rows, sort_replica_trajectories


def _fake_femto_df(first_cycle: int, n_cycles: int, n_states: int = 4,
                   steps_per_cycle: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mimic femto's samples table: one row per cycle, absolute step counter."""
    rng = np.random.default_rng(seed)
    steps = (np.arange(first_cycle, first_cycle + n_cycles) * steps_per_cycle)
    r2s = [rng.permutation(n_states).astype(np.int16) for _ in range(n_cycles)]
    u_kn = [np.zeros((n_states, n_states)) for _ in range(n_cycles)]
    return pd.DataFrame({"step": steps, "replica_to_state_idx": r2s, "u_kn": u_kn})


class TestSavedFrameRows:
    def test_aligned_start_matches_legacy_slicing(self):
        df = _fake_femto_df(first_cycle=0, n_cycles=50)
        rows = saved_frame_rows(df, save_interval=10)
        np.testing.assert_array_equal(rows, np.arange(0, 50, 10))

    def test_offset_start(self):
        # femto saves at absolute cycle % 10 == 0 → cycles 23010, 23020, ...
        df = _fake_femto_df(first_cycle=23001, n_cycles=40)
        rows = saved_frame_rows(df, save_interval=10)
        np.testing.assert_array_equal(rows, [9, 19, 29, 39])
        cycles = df["step"].to_numpy() // 1000
        assert np.all(cycles[rows] % 10 == 0)

    def test_truncates_to_traj_len(self):
        df = _fake_femto_df(first_cycle=0, n_cycles=100)
        rows = saved_frame_rows(df, save_interval=5, traj_len=7)
        assert len(rows) == 7

    def test_sort_replica_trajectories_uses_offset_rows(self):
        df = _fake_femto_df(first_cycle=23001, n_cycles=40, n_states=3)
        out = sort_replica_trajectories(df, save_interval=10, traj_len=4)
        # Reconstruct the (frame, replica) -> state table from the output and
        # compare against the rows femto actually saved.
        rows = [9, 19, 29, 39]
        r2s = np.vstack(df["replica_to_state_idx"].values[rows])
        for replica in range(3):
            for state in range(3):
                expected = np.where(r2s[:, replica] == state)[0]
                np.testing.assert_array_equal(out[replica][state], expected)


class TestCompareToPcaUnion:
    def test_pca_only_contacts_count_as_zero_response(self):
        from chacra.response import compare_to_pca

        class _CPCA:
            pass

        class _CF:
            pass

        pca_contacts = ["a", "b", "c", "d"]
        loadings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.8, -0.2]])  # (n_contacts, n_pcs)
        cf = _CF()
        cf.cpca = _CPCA()
        cf.cpca.loadings = loadings
        cf.freqs = pd.DataFrame(np.zeros((2, 4)), columns=pca_contacts)

        # R1 known only for a, b (c, d never form at this state → R1 == 0)
        out = compare_to_pca(np.array([1.0, 0.0]), cf, ["a", "b"])
        assert out["n_aligned"] == 4
        assert out["n_only_in_pca"] == 2
        np.testing.assert_array_equal(out["R1_aligned"], [1.0, 0.0, 0.0, 0.0])
        # cos(PC1, R1) with the union = 1 / |PC1| ; intersection would give 1.0
        pc1 = loadings[:, 0]
        assert out["cos_pc1_R1"] == pytest.approx(1.0 / np.linalg.norm(pc1))
