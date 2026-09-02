"""
Tests for chacra.perturb.

- residue_incidence parses ChACRA contact names and builds the incidence matrix.
- contact_basis_scan equals -chi @ M with the MBAR-weighted covariance and the
  ranking excludes a residue's own contacts.
- ResidueEnergyGenerator: on a tiny two-residue system (no PME) the
  finite-difference generator dU/ds_i matches the analytic derivative of the
  Coulomb + LJ energy under scaling of residue i's charges and epsilons.
"""

from __future__ import annotations

import numpy as np
import pytest

from chacra.perturb import (
    contact_basis_scan,
    rank_perturbation_sites,
    residue_incidence,
    weighted_contact_covariance,
)
from chacra.reweight import OccupancyStore, ResponseEstimator


def test_residue_incidence():
    contacts = ["A:ASP:118-A:LYS:136", "A:ARG:119-A:ASP:118", "B:GLY:2-B:LYS:10"]
    residues, M = residue_incidence(contacts)
    assert residues == ["A:ASP:118", "A:ARG:119", "A:LYS:136", "B:GLY:2", "B:LYS:10"]
    assert M.shape == (3, 5)
    assert M.sum() == 6
    assert M[0, residues.index("A:ASP:118")] == 1 and M[0, residues.index("A:LYS:136")] == 1


def _toy_store(seed=0, n_frames=3000):
    """3 states, 4 contacts among residues 1..4 with correlated occupancies."""
    rng = np.random.default_rng(seed)
    contacts = ["A:ALA:1-A:GLY:2", "A:GLY:2-A:VAL:3", "A:ALA:1-A:VAL:3", "A:LEU:4-A:VAL:3"]
    lambdas = np.array([1.0, 0.9, 0.8])
    K, C = 3, 4
    phi = np.zeros((K, n_frames, C), dtype=np.uint8)
    u = np.zeros((K, n_frames, K))
    for s in range(K):
        z = rng.normal(size=n_frames)
        # contacts 0 and 1 co-vary with z; contact 3 anti-varies; contact 2 independent
        phi[s, :, 0] = z > 0
        phi[s, :, 1] = (z + 0.5 * rng.normal(size=n_frames)) > 0
        phi[s, :, 2] = rng.random(n_frames) > 0.5
        phi[s, :, 3] = z < -0.3
        u[s] = 1.0 + 0.5 * z[:, None] ** 2 * lambdas[None, :] + z[:, None] * np.sqrt(lambdas)[None, :]
    return OccupancyStore(phi=phi, u=u, contacts=contacts, lambdas=lambdas, temps=300 / lambdas)


class TestContactBasis:
    def test_scan_is_minus_chi_M(self):
        store = _toy_store()
        est = ResponseEstimator(store, use_pymbar=False)
        scan = contact_basis_scan(est, lam=1.0)
        w = np.exp(est.log_weights(1.0))
        chi, _ = weighted_contact_covariance(store, w)
        residues, M = residue_incidence(store.contacts)
        np.testing.assert_allclose(scan.response, -chi @ M, atol=1e-10)
        np.testing.assert_allclose(scan.coupling, M.T @ scan.response, atol=1e-10)
        # covariance sanity: contacts 0 and 1 positively correlated, 0 and 3 negatively
        assert chi[0, 1] > 0 and chi[0, 3] < 0
        # symmetric coupling map for the contact generator
        np.testing.assert_allclose(scan.coupling, scan.coupling.T, atol=1e-10)

    def test_ranking_excludes_own_contacts(self):
        store = _toy_store()
        est = ResponseEstimator(store, use_pymbar=False)
        scan = contact_basis_scan(est, lam=1.0)
        # target: contact 3 (LEU4-VAL3). Perturbing ALA1 (in contacts 0, 2) should
        # score through its covariance with contact 3; perturbing LEU4 is "self".
        rk = rank_perturbation_sites(scan, target_contacts=["A:LEU:4-A:VAL:3"], exclude_self=True)
        leu = rk.set_index("residue").loc["A:LEU:4"]
        assert leu["score"] == 0.0  # its only contact is the target itself -> excluded
        ala = rk.set_index("residue").loc["A:ALA:1"]
        assert ala["abs_score"] > 0
        rk_all = rank_perturbation_sites(scan, target_contacts=["A:LEU:4-A:VAL:3"], exclude_self=False)
        assert rk_all.set_index("residue").loc["A:LEU:4"]["abs_score"] > 0


class TestEnergyGenerator:
    @pytest.fixture
    def tiny(self, tmp_path):
        """Two 'residues' of two atoms each, Coulomb + LJ, no cutoff, no PME."""
        openmm = pytest.importorskip("openmm")
        from openmm import app, unit

        system = openmm.System()
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        params = [(0.5, 0.3, 0.6), (-0.5, 0.3, 0.6), (0.3, 0.32, 0.4), (-0.2, 0.31, 0.5)]
        for q, sig, eps in params:
            system.addParticle(12.0)
            nb.addParticle(q, sig, eps)
        # one 1-4-like exception inside residue 0 and one across residues
        nb.addException(0, 1, 0.5 * 0.5 * -0.5, 0.3, 0.3)
        nb.addException(1, 2, 0.5 * -0.5 * 0.3, 0.31, 0.2)
        system.addForce(nb)
        xml = tmp_path / "sys.xml"
        xml.write_text(openmm.XmlSerializer.serialize(system))
        # PDB with two residues (chain A) so MDAnalysis groups atoms
        top = app.Topology()
        ch = top.addChain("A")
        r1 = top.addResidue("ALA", ch, id="1"); r2 = top.addResidue("GLY", ch, id="2")
        el = app.element.carbon
        top.addAtom("C1", el, r1); top.addAtom("C2", el, r1)
        top.addAtom("C3", el, r2); top.addAtom("C4", el, r2)
        pos = np.array([[0, 0, 0], [0.35, 0, 0], [0.1, 0.5, 0], [0.5, 0.45, 0.1]])  # nm
        pdb = tmp_path / "sys.pdb"
        with open(pdb, "w") as f:
            app.PDBFile.writeFile(top, pos * unit.nanometer, f)
        return str(xml), str(pdb), pos, params, openmm

    def _energy_scaled(self, params, pos, s, openmm):
        """Analytic Coulomb+LJ (kJ/mol) with residue scaling s = (s0, s1)."""
        ONE_4PI_EPS0 = 138.935456
        res_of = [0, 0, 1, 1]
        q = [p[0] * s[res_of[i]] for i, p in enumerate(params)]
        eps = [p[2] * s[res_of[i]] for i, p in enumerate(params)]
        sig = [p[1] for p in params]
        E = 0.0
        excl = {(0, 1): (0.5 * 0.5 * -0.5, 0.3, 0.3), (1, 2): (0.5 * -0.5 * 0.3, 0.31, 0.2)}
        for i in range(4):
            for j in range(i + 1, 4):
                r = np.linalg.norm(pos[i] - pos[j])
                if (i, j) in excl:
                    qq, sg, ep = excl[(i, j)]
                    # exception scaled like the generator: s^2 if same residue, s_i*s_j... the
                    # generator uses first-order weights (2 for same residue, 1 each otherwise)
                    if res_of[i] == res_of[j]:
                        f = 1 + 2 * (s[res_of[i]] - 1)
                    else:
                        f = 1 + (s[res_of[i]] - 1) + (s[res_of[j]] - 1)
                    E += f * (ONE_4PI_EPS0 * qq / r + 4 * ep * ((sg / r) ** 12 - (sg / r) ** 6))
                else:
                    sg = 0.5 * (sig[i] + sig[j]); ep = np.sqrt(eps[i] * eps[j])
                    E += ONE_4PI_EPS0 * q[i] * q[j] / r + 4 * ep * ((sg / r) ** 12 - (sg / r) ** 6)
        return E

    def test_generator_matches_analytic_derivative(self, tiny):
        from chacra.perturb import ResidueEnergyGenerator

        xml, pdb, pos, params, openmm = tiny
        gen = ResidueEnergyGenerator(xml, pdb, selection="all", platform="Reference", delta=1e-4)
        assert gen.labels == ["A:ALA:1", "A:GLY:2"]
        # unperturbed energy must equal the original system's energy
        e_gen = gen.generators_for_frame(pos)
        e0 = self._energy_scaled(params, pos, (1.0, 1.0), openmm)
        gen.context.setPositions(pos * openmm.unit.nanometer)
        assert gen.energy() == pytest.approx(e0, rel=1e-6)
        h = 1e-5
        for i in range(2):
            s_p = [1.0, 1.0]; s_m = [1.0, 1.0]
            s_p[i] += h; s_m[i] -= h
            analytic = (self._energy_scaled(params, pos, s_p, openmm)
                        - self._energy_scaled(params, pos, s_m, openmm)) / (2 * h)
            assert e_gen[i] == pytest.approx(analytic, rel=2e-3, abs=1e-3)
