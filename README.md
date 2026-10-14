![chacra_logo](https://github.com/Dan-Burns/ChACRA/assets/58605062/a030ffbb-0a97-4b33-a968-fab2ec7dbee9)

# ChACRA

## **Ch**emically **A**ccurate **C**ontact **R**esponse **A**nalysis

Created by Dan Burns
https://github.com/Dan-Burns/ChACRA

Tools for identifying energy-sensitive interactions in proteins using contact data from Hamiltonian replica exchange molecular dynamics (HREMD). The energy-sensitive interaction modes (chacras) are the principal components of a protein's contact frequencies across temperature. Chacras reveal functionally critical residue interactions and allosteric communication when distinct structural regions share the same mode.

With ChACRA you can run the full pipeline — simulation, contact calculation, and analysis — with a single command.

---

## Installation

### You need
- An NVIDIA GPU driver (`nvidia-smi` works on the GPU machines)
- OpenMPI (`mpicc` on PATH)
- conda (e.g. Miniforge), mamba, or micromamba

### Install

Run these from a terminal on a machine with internet access.

**On an HPC cluster (login node):**
```bash
module load openmpi miniforge      # use your site's module names
export CONDA_OVERRIDE_CUDA=12.4    # see note 1
git clone https://github.com/Dan-Burns/ChACRA.git
cd ChACRA
./install.sh
```

**On your own GPU machine or a cloud GPU instance:**
```bash
sudo apt install libopenmpi-dev openmpi-bin   # if mpicc is missing
git clone https://github.com/Dan-Burns/ChACRA.git
cd ChACRA
./install.sh
```

**Then, every time you use ChACRA:**
```bash
module load openmpi miniforge      # HPC only, and always before activating
conda activate chacra-env
```

At the end, the installer prints the environment path. Use it in your batch script, e.g. `export PATH="<env path>/bin:$PATH"`.

### Notes
1. **`CONDA_OVERRIDE_CUDA`** is only needed where there is no GPU (e.g. a login node). Set it to the "CUDA Version" shown at the top right of `nvidia-smi` on a compute node (`srun --gpus=1 nvidia-smi`).
2. **The installer uses `conda`** if it's available. To use another tool, run e.g. `CONDA_CMD=micromamba ./install.sh`, then activate with `micromamba activate chacra-env`.
3. **Starting over:** `./install.sh --reinstall` deletes and rebuilds the environment.
4. **Changing MPI modules** later requires a reinstall (`./install.sh --reinstall`), because mpi4py is built against the MPI that was loaded during install.
5. **Don't load a CUDA module.** The environment brings its own CUDA libraries.
6. On a login node, the installer's last step (OpenMM CUDA test) prints a warning. That's expected.

---

## Quick Start

```bash
mkdir ~/chacra_example && cd ~/chacra_example

# Set up project directory with example structure (1tnf.pdb)
chacra project --example

# Solvate and create OpenMM system (--fix auto-protonates with pdbfixer)
chacra make-simulation -s structures/1tnf.pdb --fix --name 1tnf_example

# Run HREMD (4 GPUs, 20 replicas, 1000 exchange cycles)
chacra run-hremd \
    --system_file system/1tnf_example_system.xml \
    --structure_file structures/1tnf_example_minimized.pdb \
    --n_cycles 1000 \
    -j 4 \
    -n 20
```

`chacra run-hremd` automatically calls `chacra process-output` after simulation. Processing strips femto's full-system replica trajectories to protein-only XTCs, runs the contact calculation once per replica (GPU-accelerated via `ultracontacts` when available; all interaction types by default), assigns every frame to its thermodynamic state with the exchange record, and produces the ChACRA analysis. State trajectories are not needed for the analysis; pass `--write_state_trajectories` to `chacra process-output` if you want them for visualisation.

```bash
# Response functions R(1), R(2) at any lambda (default: every sampled state and
# lambda = 1, the physical Hamiltonian), pooled over ALL replicas with MBAR
chacra response --run 1

# Predicted response of every contact to perturbing each residue
chacra perturb-scan --run 1
```

### Restarts

Re-run the same `chacra run-hremd` command to continue. A new `run_N/` directory is created for each run and results accumulate across runs. If a run crashes mid-simulation, simply re-run — femto will automatically resume from the last checkpoint.

---

## Output

Results are organized by run:

| Directory | Contents |
|---|---|
| `replica_trajectories/run_N/protein/` | Protein-only replica XTCs (stripped from femto's DCDs) |
| `contact_output/run_N/contacts/` | Per-replica per-frame contacts (`cont_replica_*.parquet`) |
| `contact_output/run_N/occupancy.npz` | Per-state per-frame occupancy + reduced potentials (the single source for freqs, response functions, convergence) |
| `contact_output/run_N/freqs/` | Per-state contact frequencies |
| `analysis_output/run_N/` | ChACRA plots, `.pml` visualization, `top_chacra_contacts.csv`, `response_mbar_*`, `perturb_*` |
| `state_trajectories/run_N/` | Per-state XTCs (only with `--write_state_trajectories`) |
| `analysis_output/latest/` | Symlink to the most recent run |

The `total_contacts.parquet` in each run's analysis reflects accumulated data across all runs. The `.pml` and `.csv` files reflect the combined analysis.

If `chacra process-output` fails partway through, rerun it — it skips completed stages automatically.

---

## Visualization

![chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/00a98056-bd79-4a3f-95ec-656688838301)

*Projections of contact frequency principal components (chacras). The red mode captures decreasing contact probability with temperature (melting). The blue mode captures contacts that strengthen with temperature — often revealing functionally critical interactions.*

Load your PDB and the `.pml` file into PyMOL to visualize the most sensitive contacts colored by their response pattern:

![IGPS_chacras](https://github.com/Dan-Burns/ChACRA/assets/58605062/a8eb2448-26e5-48e6-a421-6b4cc798ac33)

*IGPS chacras: the fifth chacra (orange) captures allosterically coupled sites; the second chacra (blue) captures interactions critical for activity.*

---

## CLI Reference

All commands are accessed via `chacra <command>`. Run `chacra <command> --help` for full options.

| Command | Description |
|---|---|
| `chacra run-hremd` | Run HREMD simulation + post-processing |
| `chacra process-output` | Process HREMD output (strip → contacts per replica → occupancy store → analysis) |
| `chacra response` | R(1), R(2) contact response functions, MBAR-pooled over the ladder (`--estimator single` for the one-replica cumulant) |
| `chacra perturb-scan` | Predicted contact response to per-residue perturbations (contact-basis susceptibility, or OpenMM per-residue energy generator with `--generator energy`) |
| `chacra make-simulation` | Solvate structure and create OpenMM system |
| `chacra project` | Set up project directory |
| `chacra windowed-freqs` | Compute contact frequencies for frame subsets (convergence analysis) |
| `chacra check-convergence` | Run convergence diagnostics on contact data |
| `chacra benchmark-hremd` | Benchmark HREMD throughput and exchange statistics |

---

## Response functions and perturbation scans

ChACRA's chacras are principal components of the contact frequencies across the temperature ladder. The same data also give the *response functions* directly: femto evaluates every sampled configuration under every state's Hamiltonian, so MBAR can reweight all replicas to any scaling `lambda = beta_m/beta_0` and

- `R(1)_k = d<phi_k>/dlambda = -Cov(phi_k, dU/dlambda)`
- `R(2)_k = d2<phi_k>/dlambda2` (third-cumulant term plus the `d2U/dlambda2` correction)

`chacra response` writes `response_mbar_{R1,R2,phi}.parquet` (rows = lambda) and a headline CSV at `--report_lambda` (default 1.0, the physical Hamiltonian) with block-bootstrap errors. Validate against the local finite-difference slope of the ladder (`cos_R1_ladder_slope` in the summary), not against PC1: PC1 is the ladder-averaged slope, and the local response rotates along the ladder.

`chacra perturb-scan` uses the MBAR-weighted contact covariance `chi_kl = Cov(phi_k, phi_l)` to predict how every contact responds to perturbing each residue's contacts (`R = -chi M`, `M` the residue incidence matrix) and ranks residues by their projected effect on a target set (default: the top |PC1| contacts), excluding each residue's own contacts. `--generator energy` instead scales each residue's non-bonded parameters in OpenMM (`G_i = dU/ds_i`, one energy evaluation per residue per frame, run on a GPU with `--frame_stride`).

Install `pip install pymbar` (optional) for a faster MBAR solve; a built-in self-consistent solver is used otherwise.

## Notes

- **Continuation runs**: femto saves frames when the absolute cycle is a multiple of `save_interval`; ChACRA accounts for the offset when a continuation run starts at a non-multiple cycle. Keep `save_interval` fixed across runs where possible.
- **Contact types**: `ultracontacts` alone defaults to polar contacts only; ChACRA passes `--itypes all` (configurable with `--contact_itypes` / `contact_itypes` in `chacra_run.json`) so the observable matches the getcontacts-based reference datasets.
- **Replica count**: 20–40 replicas are typical for systems with 50k–300k particles, targeting ~15–25% exchange rates. This requires trial and error.
- **GPU oversubscription**: Use `--oversubscribe 2` to run multiple replicas per GPU simultaneously via NVIDIA MPS.
- **Ligands**: Systems with ligands may require adding custom force names to femto's `_SUPPORTED_FORCES` list in `femto/md/rest.py`. HREMD scaling is limited to the protein.
- **NaN errors**: Usually caused by inadequately minimized starting coordinates.
- **CUDA PTX errors**: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` means conda's CUDA toolkit is newer than your driver supports. The install script handles this automatically.

---

## Citations

Please cite the following if you use ChACRA:

1. Burns, D., Singh, A., Venditti, V. & Potoyan, D. A. Temperature-sensitive contacts in disordered loops tune enzyme I activity. *Proc. Natl. Acad. Sci. U. S. A.* **119**, e2210537119 (2022)

2. Burns, D., Venditti, V. & Potoyan, D. A. Temperature sensitive contact modes allosterically gate TRPV3. *PLoS Comput. Biol.* **19**, e1011545 (2023)

3. Burns, D., Venditti, V. & Potoyan, D. A. Illuminating protein allostery by chemically accurate contact response analysis (ChACRA). *J. Chem. Theory Comput.* (2024)

4. A. Singh,D. Burns,S.L. Sedinkin,S. Das,D.A. Potoyan, & V. Venditti, Integrating NMR and contact-response analysis reveals the allosteric network driving domain closure in Enzyme I, *Proc. Natl. Acad. Sci. U.S.A. **123** e2612191123 (2026)
