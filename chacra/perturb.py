"""
Perturbation scanning: predicted response of the contact network to local
perturbations, from the same HREMD ensemble that gives R^(1).

Two generators for "perturb residue i":

1. **Contact basis** (cheap, no new energies).  With the contact-energy
   Hamiltonian picture of the AAlostery framework, perturbing residue i by a
   field h on every contact it participates in has generator
   ``J_i = sum_{l in contacts(i)} phi_l``.  The linear response of every
   contact is then

       R_{k,i} = -beta * Cov(phi_k, J_i) = -(chi J)_{k,i},   chi_kl = Cov_w(phi_k, phi_l)

   computed with MBAR weights at a chosen lambda (lambda = 1 is the physical
   Hamiltonian).  This is the projected susceptibility of Paper 1 in the
   contact basis and costs one (C x C) weighted covariance.

2. **Energy basis** (chemically explicit, needs an OpenMM pass over frames).
   Residue i's non-bonded parameters (charges, LJ epsilons, and the 1-4
   exceptions they enter) are scaled by a global parameter ``s_i``.  The
   generator is ``G_i = dU/ds_i`` at ``s_i = 1``, estimated by a one-sided
   finite difference, so that ``R_{k,i} = -beta Cov_w(phi_k, G_i)`` is the
   response of contact k to *softening* residue i's interactions — the
   first-order effect of a conservative mutation at i.  One OpenMM energy
   evaluation per residue per frame; run on a frame subsample on a GPU.

Both give a (C x n_res) response matrix and a residue x residue coupling map
``A_{j,i} = sum_{k in contacts(j)} R_{k,i}``; :func:`rank_perturbation_sites`
scores sites by their projected effect on a target contact set (for example
the top-loaded contacts of a chacra).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from chacra.reweight import OccupancyStore, ResponseEstimator


# --------------------------------------------------------------------------- #
# Contact <-> residue bookkeeping                                              #
# --------------------------------------------------------------------------- #


def contact_residues(contact: str) -> tuple[str, str]:
    """'A:ASP:118-A:LYS:136' -> ('A:ASP:118', 'A:LYS:136')."""
    a, b = contact.split("-", 1) if contact.count("-") == 1 else _split_contact(contact)
    return a, b


def _split_contact(contact: str) -> tuple[str, str]:
    # residue tokens are chain:resname:resid; resid may be negative in odd PDBs
    m = re.match(r"^([^:]+:[^:]+:-?\d+[A-Za-z]?)-([^:]+:[^:]+:-?\d+[A-Za-z]?)$", contact)
    if not m:
        raise ValueError(f"cannot parse contact name {contact!r}")
    return m.group(1), m.group(2)


def residue_incidence(contacts: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """
    Residues (sorted by chain, resid) and the (C x n_res) 0/1 incidence matrix
    M with M[k, i] = 1 if contact k involves residue i.
    """
    res_set = set()
    pairs = []
    for c in contacts:
        a, b = contact_residues(c)
        res_set.update((a, b))
        pairs.append((a, b))

    def _key(r):
        ch, _, num = r.split(":")
        return (ch, int(re.match(r"-?\d+", num).group()), num)

    residues = sorted(res_set, key=_key)
    ridx = {r: i for i, r in enumerate(residues)}
    M = np.zeros((len(contacts), len(residues)), dtype=np.float64)
    for k, (a, b) in enumerate(pairs):
        M[k, ridx[a]] = 1.0
        M[k, ridx[b]] = 1.0
    return residues, M


# --------------------------------------------------------------------------- #
# Contact-basis susceptibility                                                 #
# --------------------------------------------------------------------------- #


def weighted_contact_covariance(store: OccupancyStore, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    chi_kl = Cov_w(phi_k, phi_l) over the pooled frames with weights w
    (pooled, state-major order; normalised internally).  Returns (chi, phi_mean).
    """
    K, n_fr, C = store.phi.shape
    w = w / w.sum()
    m = np.zeros(C)
    S = np.zeros((C, C))
    for s in range(K):
        ws = w[s * n_fr:(s + 1) * n_fr]
        if not np.any(ws):
            continue
        phi = store.phi[s].astype(np.float32)
        m += ws @ phi
        S += (phi * ws[:, None]).T @ phi
    return S - np.outer(m, m), m


@dataclass
class PerturbationScan:
    contacts: list[str]
    residues: list[str]
    response: np.ndarray        # (C, n_res)  R_{k,i}: d<phi_k>/d(perturbation at i)
    coupling: np.ndarray        # (n_res, n_res) A_{j,i}
    generator: str              # "contact" | "energy"
    lam: float
    ess: float
    phi_mean: np.ndarray

    def response_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.response, index=self.contacts, columns=self.residues)

    def coupling_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.coupling, index=self.residues, columns=self.residues)


def contact_basis_scan(est: ResponseEstimator, lam: float = 1.0) -> PerturbationScan:
    """
    Response of every contact to a unit contact-energy perturbation on every
    residue, R = -chi M (chi: weighted contact covariance at lambda; M: incidence).
    """
    store = est.store
    w = np.exp(est.log_weights(lam))
    chi, m = weighted_contact_covariance(store, w)
    residues, M = residue_incidence(store.contacts)
    R = -chi @ M                                   # (C, n_res)
    A = M.T @ R                                    # (n_res, n_res): sum over contacts of j
    return PerturbationScan(contacts=list(store.contacts), residues=residues, response=R,
                            coupling=A, generator="contact", lam=float(lam),
                            ess=est.ess(lam), phi_mean=m)


# --------------------------------------------------------------------------- #
# Energy-basis generator (OpenMM finite differences)                          #
# --------------------------------------------------------------------------- #


def _protein_residue_groups(structure_pdb: str, selection: str = "protein") -> tuple[list[str], list[np.ndarray]]:
    """Residue labels (chain:resname:resid, ChACRA convention) and their atom indices."""
    import MDAnalysis as mda

    u = mda.Universe(structure_pdb)
    sel = u.select_atoms(selection)
    labels, groups = [], []
    for res in sel.residues:
        chain = getattr(res, "segid", "") or getattr(res.atoms[0], "chainID", "") or "A"
        if hasattr(res.atoms[0], "chainID") and res.atoms[0].chainID:
            chain = res.atoms[0].chainID
        labels.append(f"{chain}:{res.resname}:{res.resid}")
        groups.append(res.atoms.ix.copy())
    return labels, groups


class ResidueEnergyGenerator:
    """
    Evaluate G_i = dU/ds_i for every protein residue on given configurations.

    The system's NonbondedForce is rewired so that residue i's charges and
    LJ epsilons (and the 1-4 exception parameters they enter) are multiplied by
    a global parameter ``s_i`` (via parameter offsets); at s_i = 1 the energy is
    unchanged.  G_i is a one-sided finite difference with step ``delta``.

    Parameters
    ----------
    system_xml : path
        The *unmodified* OpenMM system used to build the simulation (before
        femto's REST2 rewiring) — the generator is defined at the physical
        Hamiltonian.
    structure_pdb : path
        Full-system structure (topology for the trajectory frames).
    platform : str
        OpenMM platform name ("CUDA", "OpenCL", "CPU").
    """

    def __init__(self, system_xml: str, structure_pdb: str, selection: str = "protein",
                 platform: str = "CUDA", delta: float = 0.02):
        import openmm
        from openmm import XmlSerializer
        from openmm.app import PDBFile

        self.delta = float(delta)
        with open(system_xml) as f:
            system = XmlSerializer.deserialize(f.read())
        self.labels, groups = _protein_residue_groups(structure_pdb, selection)
        nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
        if len(nb) != 1:
            raise ValueError("expected exactly one NonbondedForce in the system")
        nb = nb[0]
        atom_to_res = {}
        for i, g in enumerate(groups):
            for a in g:
                atom_to_res[int(a)] = i
        self.param_names = [f"s_res_{i}" for i in range(len(groups))]
        for name in self.param_names:
            nb.addGlobalParameter(name, 1.0)
        self._const = "s_res_const"
        nb.addGlobalParameter(self._const, 1.0)
        # particles: charge*s, eps*s ; base parameters zeroed
        for a, i in atom_to_res.items():
            q, sig, eps = nb.getParticleParameters(a)
            nb.setParticleParameters(a, 0.0, sig, 0.0)
            nb.addParticleParameterOffset(self.param_names[i], a, q, 0.0, eps)
        # exceptions (1-4s and exclusions with non-zero terms)
        for e in range(nb.getNumExceptions()):
            p1, p2, qq, sig, eps = nb.getExceptionParameters(e)
            r1, r2 = atom_to_res.get(p1), atom_to_res.get(p2)
            if r1 is None and r2 is None:
                continue
            if qq._value == 0.0 and eps._value == 0.0:
                continue
            # Offsets are linear in s: value = base + sum_k coeff_k * s_k.  The true
            # scaling is s^2 (both atoms in residue i) or s_i*s_j (different
            # residues); its first-order expansion at s = 1 is reproduced with
            # coefficients 2*qq (same residue) or qq + qq (two residues) plus a
            # constant -qq attached to a fixed unit parameter (OpenMM rejects a
            # negative base epsilon), so U(s=1) is unchanged and dU/ds_i is exact.
            nb.setExceptionParameters(e, p1, p2, 0.0, sig, 0.0)
            if r1 == r2:
                nb.addExceptionParameterOffset(self.param_names[r1], e, 2.0 * qq, 0.0, 2.0 * eps)
                nb.addExceptionParameterOffset(self._const, e, -qq, 0.0, -eps)
            elif r1 is not None and r2 is not None:
                for r in (r1, r2):
                    nb.addExceptionParameterOffset(self.param_names[r], e, qq, 0.0, eps)
                nb.addExceptionParameterOffset(self._const, e, -qq, 0.0, -eps)
            else:  # one atom outside the protein: plain s scaling
                nb.addExceptionParameterOffset(self.param_names[r1 if r1 is not None else r2], e, qq, 0.0, eps)
        # Speed: put every other force in group 1 and only evaluate the nonbonded group
        for f in system.getForces():
            f.setForceGroup(1)
        nb.setForceGroup(0)
        integrator = openmm.VerletIntegrator(0.001)
        plat = openmm.Platform.getPlatformByName(platform)
        self.context = openmm.Context(system, integrator, plat)
        self.pdb = PDBFile(structure_pdb)
        self._nb = nb

    def energy(self) -> float:
        import openmm.unit as unit

        return self.context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    def generators_for_frame(self, positions_nm: np.ndarray, box_nm: np.ndarray | None = None) -> np.ndarray:
        """G (n_res,) in kJ/mol per unit s for one configuration."""
        import openmm
        import openmm.unit as unit

        if box_nm is not None:
            box = np.asarray(box_nm, dtype=float)
            self.context.setPeriodicBoxVectors(
                *[openmm.Vec3(*row) * unit.nanometer for row in box])
        self.context.setPositions(positions_nm * unit.nanometer)
        e0 = self.energy()
        G = np.empty(len(self.param_names))
        for i, name in enumerate(self.param_names):
            self.context.setParameter(name, 1.0 - self.delta)
            G[i] = (e0 - self.energy()) / self.delta
            self.context.setParameter(name, 1.0)
        return G


def energy_generators_for_store(
    store: OccupancyStore,
    project_dir: str | os.PathLike,
    run: int,
    system_xml: str,
    structure_pdb: str,
    frame_stride: int = 10,
    platform: str = "CUDA",
    delta: float = 0.02,
    selection: str = "protein",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute G_i for a strided subsample of the store's pooled frames by reading
    the corresponding full-system replica frames.

    Returns (pooled_indices, G, residue_labels) with G of shape (n_sub, n_res).
    """
    import MDAnalysis as mda

    from chacra.reweight import load_femto_reduced_potentials

    project_dir = Path(project_dir)
    samples = project_dir / "replica_trajectories" / f"run_{run}" / "samples.arrow"
    _, r2s = load_femto_reduced_potentials(samples, int(store.meta.get("save_interval", 1)))
    r2s = r2s[:store.n_frames]
    K, n_fr = store.n_states, store.n_frames

    gen = ResidueEnergyGenerator(system_xml, structure_pdb, selection, platform, delta)
    sub_frames = np.arange(0, n_fr, frame_stride)
    # For every (state, frame) we need replica r = argmax(r2s[j] == s)
    jobs: dict[int, list[tuple[int, int]]] = {r: [] for r in range(K)}
    for s in range(K):
        for j in sub_frames:
            r = int(np.argmax(r2s[j] == s))
            jobs[r].append((j, s * n_fr + j))
    pooled_idx, G = [], []
    for r in range(K):
        if not jobs[r]:
            continue
        u = mda.Universe(structure_pdb, str(project_dir / "replica_trajectories" / f"run_{run}"
                                           / "trajectories" / f"r{r}.dcd"))
        for j, n in sorted(jobs[r]):
            ts = u.trajectory[j]
            box = mda.lib.mdamath.triclinic_vectors(ts.dimensions) / 10.0 if ts.dimensions is not None else None
            G.append(gen.generators_for_frame(ts.positions / 10.0, box))
            pooled_idx.append(n)
        if verbose:
            print(f"  [perturb] replica {r}: {len(jobs[r])} frames done")
    order = np.argsort(pooled_idx)
    return np.asarray(pooled_idx)[order], np.asarray(G)[order], gen.labels


def energy_basis_scan(
    est: ResponseEstimator,
    pooled_idx: np.ndarray,
    G: np.ndarray,
    residue_labels: list[str],
    lam: float = 1.0,
    temperature_K: float | None = None,
) -> PerturbationScan:
    """
    R_{k,i} = -beta Cov_w(phi_k, G_i) on the subsample with MBAR weights at
    lambda renormalised over the subsample.  beta uses the reference (solvent)
    temperature of the ladder unless given.
    """
    store = est.store
    K, n_fr, C = store.phi.shape
    T = float(temperature_K if temperature_K is not None else store.temps[0])
    beta = 1.0 / (0.0083144626 * T)  # kJ/mol
    logw = est.log_weights(lam)[pooled_idx]
    w = np.exp(logw - logw.max()); w /= w.sum()
    phi = np.stack([store.phi[n // n_fr, n % n_fr] for n in pooled_idx]).astype(np.float64)
    m_phi = w @ phi
    m_G = w @ G
    cov = (phi - m_phi).T @ ((G - m_G) * w[:, None])      # (C, n_res)
    R = -beta * cov
    residues, M = residue_incidence(store.contacts)
    # align residue axes: energy generator residues may include residues with no contacts
    lab_idx = {r: i for i, r in enumerate(residue_labels)}
    cols = [lab_idx.get(r) for r in residues]
    if any(c is None for c in cols):
        missing = [r for r, c in zip(residues, cols) if c is None]
        raise ValueError(f"residue labels from the structure do not match contact names, e.g. {missing[:3]}")
    R_aligned = R[:, cols]
    A = M.T @ R_aligned
    ess = float(1.0 / np.sum(w ** 2))
    return PerturbationScan(contacts=list(store.contacts), residues=residues, response=R_aligned,
                            coupling=A, generator="energy", lam=float(lam), ess=ess, phi_mean=m_phi)


# --------------------------------------------------------------------------- #
# Ranking                                                                      #
# --------------------------------------------------------------------------- #


def rank_perturbation_sites(
    scan: PerturbationScan,
    target_contacts: Sequence[str] | None = None,
    target_weights: np.ndarray | None = None,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    Score each residue i by its projected effect on a target contact set:

        score_i = sum_k t_k R_{k,i}   (signed)      and   |score| for ranking,

    with t the target indicator (or given weights, e.g. PC1 loadings).  With
    ``exclude_self`` the direct contacts of residue i are removed from the
    target so the score measures *allosteric* (non-local) response only.
    """
    C = len(scan.contacts)
    if target_weights is not None:
        t = np.asarray(target_weights, float)
    else:
        t = np.zeros(C)
        if target_contacts is None:
            t[:] = 1.0
        else:
            cidx = {c: i for i, c in enumerate(scan.contacts)}
            for c in target_contacts:
                if c in cidx:
                    t[cidx[c]] = 1.0
    _, M = residue_incidence(scan.contacts)
    rows = []
    for i, res in enumerate(scan.residues):
        mask = np.ones(C, bool)
        if exclude_self:
            mask &= M[:, i] == 0
        r = scan.response[:, i]
        signed = float((t * r)[mask].sum())
        rows.append(dict(residue=res, score=signed, abs_score=abs(signed),
                         n_contacts=int(M[:, i].sum()),
                         total_abs_response=float(np.abs(r[mask]).sum())))
    return pd.DataFrame(rows).sort_values("abs_score", ascending=False).reset_index(drop=True)
