"""
Tests for the replica-first processing path (Phase 3):

- build_occupancy_store in "replica" mode assigns each replica frame to the
  state given by the exchange record and reproduces the "state" mode result.
- strip_replica_trajectory / demux_state_trajectories produce per-state
  trajectories whose frames match the exchange record, chronologically.
- the convergence loader can build _StateContacts from an occupancy store.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow
import pytest

from chacra.reweight import build_occupancy_store, OccupancyStore


K, N_FR, STEPS = 3, 12, 1000
RESIDUES = ["A:ALA:1", "A:GLY:2", "A:VAL:3", "A:LEU:4"]


def _write_samples_arrow(path, first_cycle, n_cycles, r2s_per_cycle, u_per_cycle):
    schema = pyarrow.schema([
        ("step", pyarrow.int64()),
        ("u_kn", pyarrow.list_(pyarrow.list_(pyarrow.float64(), K), K)),
        ("replica_to_state_idx", pyarrow.list_(pyarrow.int16(), K)),
        ("n_proposed_swaps", pyarrow.list_(pyarrow.list_(pyarrow.int64(), K), K)),
        ("n_accepted_swaps", pyarrow.list_(pyarrow.list_(pyarrow.int64(), K), K)),
    ])
    with pyarrow.OSFile(str(path), "wb") as f, pyarrow.RecordBatchStreamWriter(f, schema) as w:
        for c in range(n_cycles):
            rec = pyarrow.record_batch([
                ((first_cycle + c) * STEPS,),
                (u_per_cycle[c].tolist(),),
                (r2s_per_cycle[c].astype(np.int16),),
                (np.zeros((K, K), int).tolist(),),
                (np.zeros((K, K), int).tolist(),),
            ], schema=schema)
            w.write_batch(rec)


def _write_long_parquet(path, frames, pairs):
    """ultracontacts long format with atom names (residue + atom)."""
    rows = []
    for fr, (r1, r2) in zip(frames, pairs):
        rows.append((int(fr), "hb", f"{r1}:CA", f"{r2}:CB"))
    pd.DataFrame(rows, columns=["frame", "itype", "atom1", "atom2"]).to_parquet(path)


@pytest.fixture
def project(tmp_path):
    """
    A fake ChACRA project with K replicas, save_interval 2, first cycle 7
    (so the first saved frame is samples row 1), and known per-replica
    contacts: replica r has contact (RES[r], RES[r+1]) on even frames only.
    """
    rng = np.random.default_rng(0)
    save_interval, first_cycle = 2, 7
    n_cycles = 2 * N_FR + 1
    r2s = np.array([rng.permutation(K) for _ in range(n_cycles)])
    lam = np.array([1.0, 0.9, 0.8])
    # reduced potentials of REST2 form for the replica at each state
    u = np.zeros((n_cycles, K, K))
    for c in range(n_cycles):
        b = rng.normal(size=K); cc = rng.normal(size=K)
        for m in range(K):  # replica m
            u[c, :, m] = 5.0 + b[m] * lam + cc[m] * np.sqrt(lam)
    (tmp_path / "replica_trajectories" / "run_1").mkdir(parents=True)
    _write_samples_arrow(tmp_path / "replica_trajectories/run_1/samples.arrow",
                         first_cycle, n_cycles, r2s, u)
    json.dump({"temps": (290.0 / lam).tolist(), "save_interval": save_interval,
               "hremd_temperature": 290.0},
              open(tmp_path / "chacra_run.json", "w"))
    cdir = tmp_path / "contact_output/run_1/contacts"
    cdir.mkdir(parents=True)
    for r in range(K):
        frames = np.arange(0, N_FR, 2)
        _write_long_parquet(cdir / f"cont_replica_{r}.parquet", frames,
                            [(RESIDUES[r], RESIDUES[r + 1])] * len(frames))
    # saved rows: cycles 8,10,...  -> rows 1,3,5,...
    rows = np.arange(1, n_cycles, 2)[:N_FR]
    return tmp_path, r2s[rows], lam


class TestReplicaStore:
    def test_frames_assigned_to_states_by_exchange_record(self, project):
        root, r2s_saved, lam = project
        store = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False)
        assert store.meta["contact_source"] == "replica"
        assert store.phi.shape == (K, N_FR, K)  # K contacts, one per replica
        np.testing.assert_allclose(store.lambdas, lam)
        for s in range(K):
            for j in range(N_FR):
                r = int(np.argmax(r2s_saved[j] == s))
                expected = np.zeros(K, dtype=np.uint8)
                if j % 2 == 0:
                    lo, hi = sorted((RESIDUES[r], RESIDUES[r + 1]))
                    expected[store.contacts.index(f"{lo}-{hi}")] = 1
                np.testing.assert_array_equal(store.phi[s, j], expected)

    def test_reduced_potentials_follow_replica(self, project):
        root, r2s_saved, lam = project
        store = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False)
        a, b, c, resid = store.energy_decomposition()
        assert resid < 1e-9
        np.testing.assert_allclose(a, 5.0, atol=1e-9)

    def test_roundtrip_and_freqs(self, project, tmp_path):
        root, _, _ = project
        store = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False)
        p = tmp_path / "occ.npz"
        store.save(p)
        back = OccupancyStore.load(p)
        # every contact is present on half of the frames at every state
        np.testing.assert_allclose(back.freqs().to_numpy().sum(axis=1), 0.5)
        # a modern store carries its temperature metadata and is not rescaled
        assert back.meta["u_temperature"] == 290.0
        assert back.meta["u_source_temperature"] == 290.0
        assert "legacy_rescaled" not in back.meta
        np.testing.assert_allclose(back.u, store.u)


class TestFemtoTemperatureCorrection:
    """femto scored u at 298.15 K (its default) in every run before 2026-11;
    the MD ran at temps[0].  Legacy data must be rescaled by 298.15 / temps[0]."""

    FACTOR = 298.15 / 290.0

    @staticmethod
    def _drop_temperature_from_config(root):
        cfg = json.load(open(root / "chacra_run.json"))
        cfg.pop("hremd_temperature", None)
        json.dump(cfg, open(root / "chacra_run.json", "w"))

    def test_legacy_run_without_record_is_rescaled(self, project):
        root, _, _ = project
        self._drop_temperature_from_config(root)
        store = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False)
        a, b, c, resid = store.energy_decomposition()
        assert resid < 1e-9
        np.testing.assert_allclose(a, 5.0 * self.FACTOR, atol=1e-9)
        assert store.meta["u_temperature"] == 290.0
        assert store.meta["u_source_temperature"] == pytest.approx(298.15)

    def test_run_meta_json_overrides_default(self, project):
        root, _, _ = project
        self._drop_temperature_from_config(root)
        json.dump({"hremd_temperature": 290.0},
                  open(root / "replica_trajectories/run_1/run_meta.json", "w"))
        store = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False)
        a, *_ = store.energy_decomposition()
        np.testing.assert_allclose(a, 5.0, atol=1e-9)
        assert store.meta["u_source_temperature"] == 290.0

    def test_explicit_source_temperature_wins(self, project):
        root, _, _ = project
        store = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False,
                                      u_source_temperature=580.0)
        a, *_ = store.energy_decomposition()
        np.testing.assert_allclose(a, 10.0, atol=1e-9)

    def test_legacy_npz_rescaled_on_load(self, project, tmp_path):
        root, _, _ = project
        store = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False)
        legacy = OccupancyStore(phi=store.phi, u=store.u, contacts=store.contacts,
                                lambdas=store.lambdas, temps=store.temps,
                                meta={"run": 1, "save_interval": 2})   # no u_temperature
        p = tmp_path / "legacy.npz"
        legacy.save(p)
        back = OccupancyStore.load(p, verbose=False)
        np.testing.assert_allclose(back.u, store.u * self.FACTOR)
        assert back.meta["legacy_rescaled"] is True
        assert back.meta["u_temperature"] == 290.0
        untouched = OccupancyStore.load(p, legacy_u_temperature=None)
        np.testing.assert_allclose(untouched.u, store.u)
        assert "u_temperature" not in untouched.meta
        # the file itself is not rewritten
        again = OccupancyStore.load(p, legacy_u_temperature=None)
        np.testing.assert_allclose(again.u, store.u)


class TestContactStride:
    def test_strided_contacts_divide_frame_indices(self, project):
        # the fixture's contact files hold even frames only; read them as the
        # output of `ultracontacts --stride 2` (absolute frame indices)
        root, r2s_saved, lam = project
        full = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False)
        strided = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False, contact_stride=2)
        assert strided.n_frames == N_FR // 2
        assert strided.meta["contact_stride"] == 2
        np.testing.assert_allclose(strided.u, full.u[:, ::2])
        # every strided frame carries its replica's contact: per-state sums are 1
        np.testing.assert_allclose(strided.freqs().to_numpy().sum(axis=1), 1.0)
        np.testing.assert_array_equal(strided.phi, full.phi[:, ::2])


class TestConvergenceFromStore:
    def test_dense_block_path_matches_frame_path(self, project, tmp_path):
        from chacra.convergence import (
            _block_sums_from_stores,
            _freqs_from_blocks,
            _load_all_states_from_stores,
            _build_freq_matrix,
            bootstrap_loadings_dense,
            split_half_rmsip_dense,
        )

        root, _, _ = project
        store = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False)
        cdir = tmp_path / "contact_output" / "run_1"   # the fixture's own run dir
        cdir.mkdir(parents=True, exist_ok=True)
        p = cdir / "occupancy.npz"
        store.save(p)
        sums, sizes, contacts = _block_sums_from_stores([p], K, block=1)
        assert contacts == store.contacts
        freqs = _freqs_from_blocks(sums, sizes)
        np.testing.assert_allclose(freqs, store.freqs().to_numpy())
        # block=1 split-half equals the per-frame split
        states = _load_all_states_from_stores([p], K)
        halves = [sc.frames[: len(sc.frames) // 2] for sc in states]
        ref = _build_freq_matrix(states, halves).reindex(columns=contacts, fill_value=0.0).to_numpy()
        wa = [np.r_[np.ones(len(n) // 2), np.zeros(len(n) - len(n) // 2)] for n in sizes]
        np.testing.assert_allclose(_freqs_from_blocks(sums, sizes, wa), ref)
        val = split_half_rmsip_dense([p], K, k=2, block=1)
        assert 0.0 <= val <= 1.0
        boot = bootstrap_loadings_dense([p], K, k=2, n_top=2, n_bootstrap=5, block=2)
        assert set(boot.index) == set(contacts)
        assert ((boot >= 0) & (boot <= 1)).all().all()

    def test_state_contacts_from_store(self, project, tmp_path):
        from chacra.convergence import _load_all_states_from_stores

        root, r2s_saved, _ = project
        store = build_occupancy_store(root, run=1, n_frames=N_FR, verbose=False)
        p = tmp_path / "occupancy.npz"
        store.save(p)
        states = _load_all_states_from_stores([p], K)
        assert [s.state_idx for s in states] == list(range(K))
        for s in states:
            np.testing.assert_array_equal(s.frames, np.arange(N_FR))
            total = sum(len(v) for v in s.pair_frames.values())
            assert total == N_FR // 2  # one contact on every even frame
            for arr in s.pair_frames.values():
                assert np.all(np.diff(arr) > 0)  # sorted, as the RMSIP code requires


class TestStripAndDemux:
    @pytest.fixture
    def tiny_system(self, tmp_path):
        """A 6-atom 'protein' + 4-atom 'solvent' universe with K replica DCDs."""
        import MDAnalysis as mda

        n_prot, n_sol = 6, 4
        n_atoms = n_prot + n_sol
        u = mda.Universe.empty(n_atoms, n_residues=n_prot + 1, atom_resindex=[*range(n_prot)] + [n_prot] * n_sol,
                               trajectory=True)
        u.add_TopologyAttr("name", ["CA"] * n_prot + ["OW"] * n_sol)
        u.add_TopologyAttr("resname", ["ALA"] * n_prot + ["SOL"])
        u.add_TopologyAttr("resid", list(range(1, n_prot + 2)))
        u.add_TopologyAttr("type", ["C"] * n_prot + ["O"] * n_sol)
        u.atoms.positions = np.random.default_rng(0).normal(size=(n_atoms, 3)) * 5 + 20
        u.dimensions = [50, 50, 50, 90, 90, 90]
        pdb = tmp_path / "full.pdb"
        u.atoms.write(str(pdb))
        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()
        coords = {}
        for r in range(K):
            frames = []
            with mda.Writer(str(traj_dir / f"r{r}.dcd"), n_atoms=n_atoms) as w:
                for j in range(N_FR):
                    # encode (replica, frame) in the first atom's x coordinate
                    pos = u.atoms.positions.copy()
                    pos[0, 0] = 100 * r + j
                    u.atoms.positions = pos
                    w.write(u.atoms)
                    frames.append(pos[:n_prot].copy())
            coords[r] = np.array(frames)
        return pdb, traj_dir, coords, n_prot

    def test_strip_then_demux_matches_exchange_record(self, tiny_system, tmp_path):
        import MDAnalysis as mda
        from chacra.trajectories.process_hremd import (
            strip_replica_trajectories, demux_state_trajectories,
        )

        pdb, traj_dir, coords, n_prot = tiny_system
        protein_dir = tmp_path / "protein"
        xtcs = strip_replica_trajectories(pdb, traj_dir, protein_dir, K,
                                          selection="resname ALA", n_jobs=1)
        prot_pdb = tmp_path / "protein.pdb"
        mda.Universe(str(pdb)).select_atoms("resname ALA").write(str(prot_pdb))
        for r, x in enumerate(xtcs):
            assert len(mda.Universe(str(prot_pdb), x).trajectory) == N_FR

        rng = np.random.default_rng(1)
        r2s = np.array([rng.permutation(K) for _ in range(N_FR)])
        out = tmp_path / "states"
        demux_state_trajectories(prot_pdb, xtcs, r2s, out, ref=None, n_jobs=1)
        for s in range(K):
            us = mda.Universe(str(prot_pdb), str(out / f"state_{s}.xtc"))
            assert len(us.trajectory) == N_FR
            for j, ts in enumerate(us.trajectory):
                r = int(np.argmax(r2s[j] == s))
                # alignment is a rigid transform; pairwise distances are invariant
                d_out = np.linalg.norm(ts.positions[0] - ts.positions[1])
                d_ref = np.linalg.norm(coords[r][j][0] - coords[r][j][1])
                assert d_out == pytest.approx(d_ref, abs=1e-2)

    def test_verify_and_delete_full_dcds(self, tiny_system, tmp_path):
        import json

        import MDAnalysis as mda
        from chacra.trajectories.process_hremd import (
            strip_replica_trajectories, verify_and_delete_full_dcds,
        )

        pdb, traj_dir, coords, n_prot = tiny_system
        protein_dir = tmp_path / "protein"
        xtcs = strip_replica_trajectories(pdb, traj_dir, protein_dir, K,
                                          selection="resname ALA", n_jobs=1)
        prot_pdb = tmp_path / "protein.pdb"
        mda.Universe(str(pdb)).select_atoms("resname ALA").write(str(prot_pdb))
        # one saved frame per cycle, N_FR cycles from cycle 0
        rng = np.random.default_rng(3)
        r2s = np.array([rng.permutation(K) for _ in range(N_FR)])
        u = np.zeros((N_FR, K, K))
        samples = tmp_path / "samples.arrow"
        _write_samples_arrow(samples, 0, N_FR, r2s, u)

        # wrong save_interval -> expected frames differ -> nothing deleted
        rep = verify_and_delete_full_dcds(pdb, traj_dir, prot_pdb, xtcs, samples, save_interval=2)
        assert all(e["status"] == "kept" for e in rep["replicas"].values())
        assert all((traj_dir / f"r{r}.dcd").exists() for r in range(K))
        assert not (traj_dir / "FULL_DCD_DELETED.json").exists()

        # dry run reports but keeps
        rep = verify_and_delete_full_dcds(pdb, traj_dir, prot_pdb, xtcs, samples, save_interval=1,
                                          dry_run=True)
        assert all(e["status"] == "would_delete" for e in rep["replicas"].values())
        assert all((traj_dir / f"r{r}.dcd").exists() for r in range(K))

        # matching counts -> deleted, with a record
        rep = verify_and_delete_full_dcds(pdb, traj_dir, prot_pdb, xtcs, samples, save_interval=1)
        assert all(e["status"] == "deleted" for e in rep["replicas"].values())
        assert not any((traj_dir / f"r{r}.dcd").exists() for r in range(K))
        assert rep["bytes_freed"] > 0
        rec = json.load(open(traj_dir / "FULL_DCD_DELETED.json"))
        assert rec["expected_frames"] == N_FR and len(rec["replicas"]) == K
        # the protein XTCs are untouched
        for x in xtcs:
            assert len(mda.Universe(str(prot_pdb), x).trajectory) == N_FR
