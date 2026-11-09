#!/bin/bash
# ChACRA end-to-end GPU smoke test on the bundled 1tnf example (about 5 minutes on one GPU).
#
# Steps, each reported as "[exit N] <command>"; the job exits 0 only if all pass:
#   1 chacra project --example; make-simulation (hydrogen mass 3 amu, 4 fs recorded in chacra_run.json)
#   2 benchmark-hremd, 4 replicas as 2 ranks under CUDA MPS (falls back to 1 rank per GPU)
#   3 run-hremd run_1: 40 cycles, --delete_full_trajectories; checks run_meta.json, the 290 K
#     reduced potentials, the protein-only XTC frame count and the occupancy store's u_temperature
#   4 run-hremd again: resumes into run_2 with no full DCDs present
#   5 chacra response, single-state and MBAR estimators, pooled over --runs all
#   6 sweep-benchmark --mps-only
#   7 check-convergence --bootstrap (dense occupancy-store path)
#
# Site settings: edit the block below or export the variables before sbatch, e.g.
#   CHACRA_MODULES="cuda/12.8.1 openmpi/5.0.8" CHACRA_CONDA_SH=~/miniconda3/etc/profile.d/conda.sh \
#   sbatch --account=... --partition=... tools/smoke_test_gpu.sbatch
# --ntasks must equal GPUs x --mps-replicas (2 here): inside Slurm, ChACRA launches ranks with srun.
#SBATCH --job-name=chacra-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=chacra-smoke-%j.out
set -uo pipefail
CHACRA_MODULES=${CHACRA_MODULES:-}                 # space-separated modules to load (CUDA, MPI)
CHACRA_CONDA_SH=${CHACRA_CONDA_SH:-}               # conda.sh to source; empty if chacra is already on PATH
CHACRA_ENV=${CHACRA_ENV:-chacra-env}               # conda env with chacra, femto, ultracontacts
WORK=${CHACRA_SMOKE_DIR:-${TMPDIR:-/tmp}/chacra-smoke-${SLURM_JOB_ID:-$$}}

echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
if [ -n "$CHACRA_MODULES" ]; then module purge; for m in $CHACRA_MODULES; do module load "$m"; done; fi
if [ -n "$CHACRA_CONDA_SH" ]; then source "$CHACRA_CONDA_SH"; conda activate "$CHACRA_ENV"; fi
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
# one CUDA MPS daemon per job, the convention chacra.mpi.configure_mps_env uses; quit at the end
export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/chacra-mps-$(id -u)-${SLURM_JOB_ID:-$$}"
echo "--- environment"; nvidia-smi -L; which nvidia-cuda-mps-control || echo "nvidia-cuda-mps-control: NOT on PATH (MPS steps fall back)"
command -v srun >/dev/null && { srun --mpi=list 2>&1 | tr '\n' ' '; echo; }
python -c 'import chacra, femto, os; print("chacra from", os.path.dirname(chacra.__file__)); print("femto", femto.__file__)' || exit 1

STEP() { echo; echo "=================== $(date +%H:%M:%S) $*"; }
rc_all=0
run() { "$@"; rc=$?; echo "[exit $rc] $*"; (( rc != 0 )) && rc_all=1; return $rc; }

STEP "1 project + system"
rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK" || exit 1
run chacra project --example
run chacra make-simulation -s structures/1tnf_truncated.pdb --fix --name 1tnf_example
python - <<'PYEOF'
from chacra.simulation import hydrogen_mass_from_xml, barostat_temperature_from_xml
print("H mass in XML:", hydrogen_mass_from_xml("system/1tnf_example_system.xml"),
      "| barostat T:", barostat_temperature_from_xml("system/1tnf_example_system.xml"))
PYEOF

STEP "2 benchmark-hremd (4 replicas, 1 GPU, MPS x2; falls back to x1)"
if run chacra benchmark-hremd -p system/1tnf_example_system.xml -s structures/1tnf_example_minimized.pdb -n 4 -j 1 -r 2 --n_cycles 20 --steps_per_cycle 200; then
  MPS=2
else
  echo "MPS x2 benchmark failed; retrying with -r 1"; MPS=1; rc_all=0
  run chacra benchmark-hremd -p system/1tnf_example_system.xml -s structures/1tnf_example_minimized.pdb -n 4 -j 1 -r 1 --n_cycles 20 --steps_per_cycle 200
fi
echo "MPS replicas per GPU used from here on: $MPS"

STEP "3 run-hremd run_1 (40 cycles, save 5, delete full DCDs)"
run chacra run-hremd -p system/1tnf_example_system.xml -s structures/1tnf_example_minimized.pdb -n 4 -j 1 -r "$MPS" \
    --n_cycles 40 --steps_per_cycle 200 --save_interval 5 --checkpoint_interval 20 --min_temp 290 --max_temp 320 \
    --delete_full_trajectories
echo "--- run_1"; ls replica_trajectories/run_1/trajectories 2>/dev/null
cat replica_trajectories/run_1/run_meta.json 2>/dev/null
grep -h "reduced potentials and swap" analysis_output/run_1/hremd_stdout.log | head -1
python - <<'PYEOF'
import glob, json, MDAnalysis as mda
pdb = glob.glob("structures/*_protein.pdb")[0]
xtcs = sorted(glob.glob("replica_trajectories/run_1/protein/*.xtc"))
print("protein xtcs:", len(xtcs), "frames:", [len(mda.Universe(pdb, x).trajectory) for x in xtcs])
from chacra.reweight import OccupancyStore
st = OccupancyStore.load("contact_output/run_1/occupancy.npz")
print("store:", st.n_states, "states x", st.n_frames, "frames x", st.n_contacts, "contacts; meta:", json.dumps(st.meta))
assert st.meta.get("u_temperature") == 290.0, st.meta
assert not glob.glob("replica_trajectories/run_1/trajectories/*.dcd"), "full DCDs still present"
PYEOF
[ $? -eq 0 ] || rc_all=1

STEP "4 run-hremd run_2 (resume, 20 more cycles, no DCDs present)"
run chacra run-hremd --n_cycles 20
ls replica_trajectories/; cat replica_trajectories/run_2/run_meta.json 2>/dev/null

STEP "5 response: single and mbar, pooled runs"
run chacra response --estimator single --n_boot 5 --boot_block 3 --compare_mbar --runs all
run chacra response --estimator mbar --n_boot 5 --boot_block 3 --compare_single --runs all
ls analysis_output/run_2/ | grep response

STEP "6 sweep-benchmark (MPS throughput only)"
run chacra sweep-benchmark -p system/1tnf_example_system.xml -s structures/1tnf_example_minimized.pdb --mps-only --mps-range 1,2 --mps-steps 2000

STEP "7 check-convergence --bootstrap (dense path)"
run chacra check-convergence --run 2 --bootstrap --n_bootstrap 20 2>&1 | tail -15

echo quit | timeout 10 nvidia-cuda-mps-control 2>/dev/null || true
echo; echo "=================== $(date +%H:%M:%S) DONE overall rc=$rc_all  (work dir: $WORK, $(du -sh "$WORK" | cut -f1))"
exit $rc_all
