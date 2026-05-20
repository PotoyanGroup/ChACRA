"""
Tests for chacra.response — single-state cumulant estimators.

The math is pinned with synthetic data:

  - Linear-Gaussian model:  phi_k = alpha_k * dU + iid noise.
    Then Cov(phi_k, dU) = alpha_k * Var(dU) and the third cumulant
    <delta phi_k * (delta dU)^2>_c is zero (Isserlis), so R^(2) = 0.

  - Skewed model:  dU = X^2 - 1 with X ~ N(0,1).  Third cumulant of dU is 8.
    For phi_k = alpha_k * dU we have <delta phi_k * (delta dU)^2>_c =
    alpha_k * 8, giving R^(2)_k = beta^2 * 8 * alpha_k.

The loader for getcontacts per-frame TSVs is exercised on a tiny
hand-written TSV — no MD data required.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from chacra.response import (
    cumulant_R1,
    cumulant_R2,
    cos_align,
    load_per_frame_contacts_getcontacts,
)


# ------------------------------------------------------------------ #
# cumulant_R1                                                         #
# ------------------------------------------------------------------ #


class TestCumulantR1:
    def test_recovers_linear_response_gaussian(self):
        """phi_k = alpha_k * dU + noise  →  R^(1)_k ≈ -beta * alpha_k * Var(dU)."""
        rng = np.random.default_rng(42)
        n_frames, n_contacts = 50_000, 30
        beta = 1.5

        dU = rng.normal(loc=2.0, scale=1.7, size=n_frames)
        alpha = rng.normal(size=n_contacts)
        noise = rng.normal(scale=0.3, size=(n_frames, n_contacts))
        phi = alpha[None, :] * dU[:, None] + noise

        R1 = cumulant_R1(phi, dU, beta=beta)
        expected = -beta * alpha * dU.var(ddof=1)

        # ~0.7% MC error on each component for n=50k
        np.testing.assert_allclose(R1, expected, rtol=0.05, atol=0.02)

    def test_zero_when_phi_independent_of_dU(self):
        rng = np.random.default_rng(0)
        n = 20_000
        dU = rng.normal(size=n)
        phi = rng.normal(size=(n, 10))  # independent
        R1 = cumulant_R1(phi, dU, beta=1.0)
        # All components should be statistically zero
        np.testing.assert_allclose(R1, np.zeros_like(R1), atol=0.05)

    def test_frame_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="frames"):
            cumulant_R1(np.zeros((5, 3)), np.zeros(7), beta=1.0)

    def test_signs_track_correlation(self):
        """Cov > 0 on a contact ⇒ R^(1) < 0; the minus-beta is the FDT sign."""
        rng = np.random.default_rng(1)
        n = 10_000
        dU = rng.normal(size=n)
        phi_pos = (dU + 0.1 * rng.normal(size=n)).reshape(-1, 1)  # tracks dU
        phi_neg = -phi_pos
        beta = 2.0
        assert cumulant_R1(phi_pos, dU, beta)[0] < 0
        assert cumulant_R1(phi_neg, dU, beta)[0] > 0


# ------------------------------------------------------------------ #
# cumulant_R2                                                         #
# ------------------------------------------------------------------ #


class TestCumulantR2:
    def test_zero_for_gaussian(self):
        """Isserlis: third connected moment vanishes for jointly Gaussian phi, dU."""
        rng = np.random.default_rng(7)
        n = 100_000
        dU = rng.normal(size=n)
        alpha = rng.normal(size=15)
        phi = alpha[None, :] * dU[:, None] + rng.normal(scale=0.5, size=(n, 15))

        R2 = cumulant_R2(phi, dU, beta=1.0)
        # Statistical zero — for n=100k the MC noise on each component is small
        np.testing.assert_allclose(R2, np.zeros_like(R2), atol=0.08)

    def test_recovers_third_cumulant_for_skewed_dU(self):
        """dU = X^2 - 1 with X ~ N(0,1) has <(delta dU)^3>_c = 8."""
        rng = np.random.default_rng(13)
        n = 200_000
        x = rng.normal(size=n)
        dU = x ** 2 - 1.0  # mean 0; var 2; third cumulant 8

        beta = 1.0
        alpha = np.array([1.0, -0.5, 2.3])
        phi = alpha[None, :] * dU[:, None]  # exact linear response in dU

        # <delta phi_k * (delta dU)^2>_c = alpha_k * <(delta dU)^3>_c = 8 alpha_k
        R2 = cumulant_R2(phi, dU, beta=beta)
        expected = (beta ** 2) * 8.0 * alpha
        # Third cumulant is high-variance — allow ~5% MC error at n=200k
        np.testing.assert_allclose(R2, expected, rtol=0.07)


# ------------------------------------------------------------------ #
# cos_align                                                            #
# ------------------------------------------------------------------ #


class TestCosAlign:
    def test_parallel(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cos_align(v, v) == pytest.approx(1.0)
        assert cos_align(v, -v) == pytest.approx(1.0)  # absolute value

    def test_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cos_align(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert cos_align(np.zeros(5), np.ones(5)) == 0.0


# ------------------------------------------------------------------ #
# getcontacts loader                                                  #
# ------------------------------------------------------------------ #


class TestGetcontactsLoader:
    @pytest.fixture
    def tiny_tsv(self, tmp_path):
        # 4 frames, 2 unique residue pairs, mix of interaction types.
        # Contacts:
        #   frame 0: A:LEU:10 -- A:VAL:20  (sb)
        #   frame 1: A:LEU:10 -- A:VAL:20  (vdw),  B:GLY:5 -- B:ASN:9 (hb)
        #   frame 2: (none)
        #   frame 3: B:GLY:5 -- B:ASN:9 (hb)
        content = textwrap.dedent("""\
            # total_frames:4    interaction_types:sb,vdw,hb
            # Columns: frame  itype  atom_1  atom_2
            0\tsb\tA:LEU:10:CA\tA:VAL:20:CB
            1\tvdw\tA:LEU:10:CA\tA:VAL:20:CB
            1\thb\tB:GLY:5:N\tB:ASN:9:O
            3\thb\tB:GLY:5:N\tB:ASN:9:O
        """)
        p = tmp_path / "tiny_contacts.tsv"
        p.write_text(content)
        return p

    def test_basic_load(self, tiny_tsv):
        phi, names = load_per_frame_contacts_getcontacts(tiny_tsv)
        assert phi.shape == (4, 2)
        # Contact set, canonicalised + sorted
        assert "A:LEU:10-A:VAL:20" in names
        assert "B:ASN:9-B:GLY:5" in names

        # Frame-wise occupancies
        idx = {n: i for i, n in enumerate(names)}
        leu_val = idx["A:LEU:10-A:VAL:20"]
        gly_asn = idx["B:ASN:9-B:GLY:5"]
        np.testing.assert_array_equal(
            phi[:, leu_val], [1, 1, 0, 0]
        )
        np.testing.assert_array_equal(
            phi[:, gly_asn], [0, 1, 0, 1]
        )

    def test_interaction_filter(self, tiny_tsv):
        # Restrict to vdw only — only frame 1 has it on the LEU/VAL pair.
        phi, names = load_per_frame_contacts_getcontacts(
            tiny_tsv, interaction_types=["vdw"],
        )
        assert phi.shape == (4, 1)
        assert names == ["A:LEU:10-A:VAL:20"]
        np.testing.assert_array_equal(phi[:, 0], [0, 1, 0, 0])

    def test_n_frames_override(self, tiny_tsv):
        phi, _ = load_per_frame_contacts_getcontacts(tiny_tsv, n_frames=10)
        assert phi.shape[0] == 10
        # Frames 4..9 should be all zeros
        np.testing.assert_array_equal(phi[4:].sum(axis=0), np.zeros(phi.shape[1]))
