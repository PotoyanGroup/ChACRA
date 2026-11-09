# -*- coding: utf-8 -*-
"""
MPI launcher utilities for ChACRA.

Provides a single ``build_mpi_command()`` helper that ``run_hremd``,
``benchmark_hremd`` and ``sweep_benchmark`` use to construct the correct
``mpirun`` / ``srun`` invocation.  Handles:

* Slurm allocations: ``srun`` with the best available ``--mpi=`` plugin, ranks
  spread evenly over every allocated node, and ``--export=ALL`` so the ranks
  inherit the launcher's environment even where the site sets
  ``SLURM_EXPORT_ENV=NONE``
* auto-detection of the MPI launcher (``mpirun``, ``mpiexec``, ``srun``)
* OpenMPI's ``--oversubscribe`` flag (only added when OpenMPI is detected)
* user-supplied ``--mpi-command`` strings (properly shell-split) as a site
  override

``chacra_command()`` is how a launched rank invokes the ChACRA CLI: through the
launcher's own interpreter by absolute path, so it does not depend on PATH.

The CUDA MPS helpers (``configure_mps_env``, ``ensure_mps``, ``stop_mps``,
``mps_daemon_running``) manage one control daemon per Slurm job.  ``run-hremd``
starts CUDA MPS on the node it runs on.  For multi-node runs, start MPS on every
node in your batch script before ``chacra run-hremd``.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import time


def _is_openmpi(launcher: str) -> bool:
    """Return True if *launcher* appears to be an OpenMPI binary."""
    try:
        result = subprocess.run(
            [launcher, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout + result.stderr
        return "open mpi" in output.lower() or "openmpi" in output.lower()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def find_mpi_launcher() -> str | None:
    """Return the first usable MPI launcher on PATH.

    Checks for ``mpirun``, ``mpiexec``, and ``srun`` in that order.
    Returns the absolute path (via ``shutil.which``) or *None*.
    """
    for name in ("mpirun", "mpiexec", "srun"):
        path = shutil.which(name)
        if path is not None:
            return path
    return None


def _srun_mpi_plugin(srun: str) -> str | None:
    """Return the best ``srun --mpi=`` plugin this site supports (pmix > pmi2)."""
    try:
        result = subprocess.run(
            [srun, "--mpi=list"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    plugins = set(re.findall(r"\b(pmix(?:_v\d+)?|pmi2)\b", result.stdout + result.stderr))
    pmix_versions = sorted(p for p in plugins if p.startswith("pmix_v"))
    if "pmix" in plugins:
        return "pmix"
    if pmix_versions:
        return pmix_versions[-1]
    return "pmi2" if "pmi2" in plugins else None


def build_mpi_command(
    n_ranks: int,
    *,
    mpi_command: str | None = None,
) -> list[str]:
    """Build the MPI launcher prefix (everything before the program args).

    Inside a Slurm allocation ``srun`` is used with the best MPI plugin the
    site supports, and ranks are spread evenly over all allocated nodes.
    Elsewhere ``mpirun`` / ``mpiexec`` is used.

    Parameters
    ----------
    n_ranks : int
        Total number of MPI ranks to launch (across all nodes).
    mpi_command : str or None
        User-supplied MPI launcher string that overrides auto-detection, e.g.
        ``"srun --mpi=pmix_v4"`` or ``"mpirun"``.

    Returns
    -------
    list[str]
        The MPI command tokens, e.g.
        ``["srun", "--mpi=pmix", "-n", "8", "--ntasks-per-node=4", "--export=ALL"]``
        or ``["mpirun", "-np", "8", "--oversubscribe"]`` (OpenMPI).

    Raises
    ------
    RuntimeError
        If no MPI launcher can be found.
    """
    n_nodes = os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES")
    n_nodes = int(n_nodes) if n_nodes else None

    if mpi_command is not None:
        parts = shlex.split(mpi_command)
    elif n_nodes is not None and shutil.which("srun") is not None:
        srun = shutil.which("srun")
        plugin = _srun_mpi_plugin(srun)
        parts = [srun] + ([f"--mpi={plugin}"] if plugin else [])
    else:
        launcher = find_mpi_launcher()
        if launcher is None:
            raise RuntimeError(
                "Cannot find mpirun, mpiexec, or srun on PATH.\n"
                "Ensure your MPI module is loaded (e.g. 'module load openmpi')\n"
                "or specify --mpi-command explicitly."
            )
        parts = [launcher]

    launcher = parts[0]
    ranks_per_node = -(-n_ranks // n_nodes) if n_nodes else None

    if os.path.basename(launcher) == "srun":
        cmd = [*parts, "-n", str(n_ranks)]
        if ranks_per_node and not any(p.startswith("--ntasks-per-node") for p in parts):
            cmd.append(f"--ntasks-per-node={ranks_per_node}")
        # srun's default is --export=ALL, but a site can set SLURM_EXPORT_ENV=NONE
        # (ISU Nova does); the ranks then start with no PATH, HOME or conda prefix
        # and the worker cannot even be found.  Saying ALL explicitly wins over the
        # variable and changes nothing elsewhere.
        if not any(p == "--export" or p.startswith("--export=") for p in parts):
            cmd.append("--export=ALL")
    else:
        cmd = [*parts, "-np", str(n_ranks)]
        # OpenMPI requires --oversubscribe when n_ranks > n_physical_cores
        # (common with CUDA MPS).  Other MPI implementations don't have it.
        if _is_openmpi(launcher):
            cmd.append("--oversubscribe")
            # by default OpenMPI fills the first node's slots before moving on
            if ranks_per_node and "--map-by" not in parts:
                cmd += ["--map-by", f"ppr:{ranks_per_node}:node"]

    return cmd


def chacra_command() -> list[str]:
    """How a launched rank invokes the ``chacra`` CLI.

    ``[sys.executable, "-m", "chacra.cli"]``: the interpreter running the launcher,
    by absolute path.  A bare ``chacra`` relies on the rank's PATH, which an
    ``srun`` task does not have on ``SLURM_EXPORT_ENV=NONE`` sites, and could
    resolve to a different installation than the launcher's.
    """
    return [sys.executable, "-m", "chacra.cli"]


# --------------------------------------------------------------------------- #
# CUDA MPS                                                                     #
# --------------------------------------------------------------------------- #

def mps_pipe_directory() -> str:
    """The pipe directory the CUDA MPS daemon and its clients agree on."""
    return os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")


def configure_mps_env() -> None:
    """Give CUDA MPS a per-job pipe directory and a writable log directory.

    The default pipe directory (``/tmp/nvidia-mps``) is shared by every process
    of the user on the node, so two Slurm jobs of one user on one GPU node would
    attach to each other's daemon and one job's ``quit`` would kill the other's
    server.  Inside a Slurm job the pipe directory becomes
    ``$TMPDIR/chacra-mps-<uid>-<jobid>`` (the convention of
    ``multi_node_job.sbatch``) unless ``CUDA_MPS_PIPE_DIRECTORY`` is already set.

    ``nvidia-cuda-mps-control`` logs to ``/var/log/nvidia-mps`` by default, which
    non-root users can't write to (harmless, but it prints a warning and loses
    the daemon logs); the log directory goes under the pipe directory instead.

    Call this before starting MPS and before copying ``os.environ`` for the
    ranks: daemon and clients must see the same variables.
    """
    tmp = os.environ.get("TMPDIR", "/tmp")
    job = os.environ.get("SLURM_JOB_ID")
    if job and "CUDA_MPS_PIPE_DIRECTORY" not in os.environ:
        pipe_dir = os.path.join(tmp, f"chacra-mps-{os.getuid()}-{job}")
        os.makedirs(pipe_dir, exist_ok=True)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir

    if "CUDA_MPS_LOG_DIRECTORY" in os.environ:
        return
    default = "/var/log/nvidia-mps"
    if os.path.isdir(default) and os.access(default, os.W_OK):
        return
    if "CUDA_MPS_PIPE_DIRECTORY" in os.environ:
        log_dir = os.path.join(os.environ["CUDA_MPS_PIPE_DIRECTORY"], "logs")
    else:
        user = os.environ.get("USER") or str(os.getuid())
        log_dir = os.path.join(tmp, f"nvidia-mps-log-{user}")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir
    os.environ["CUDA_MPS_LOG_DIR"] = log_dir


def mps_daemon_running() -> bool:
    """True if a CUDA MPS control daemon serves this process's pipe directory.

    femto's ``is_mps_running`` looks for ``nvidia-cuda-mps-server``, which only
    exists once the first CUDA client has connected; right after the control
    daemon starts it answers False, and a second ``nvidia-cuda-mps-control -d``
    then fails with "An instance of this daemon is already running".  This
    checks for the control daemon itself, and that it has populated the pipe
    directory this process would use.
    """
    pipe_dir = mps_pipe_directory()
    try:
        if not os.listdir(pipe_dir):
            return False
    except OSError:
        return False
    try:
        result = subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-f", "nvidia-cuda-mps-control"],
            capture_output=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return result.returncode == 0


def ensure_mps() -> bool:
    """Start the CUDA MPS control daemon for this job unless one already serves it.

    Returns True if this call started the daemon; the caller then owns it and
    stops it with :func:`stop_mps` when its ranks are done.  Returns False if a
    daemon was already running for this pipe directory.

    Raises
    ------
    RuntimeError
        If ``nvidia-cuda-mps-control`` is not on PATH or fails to start.
    """
    configure_mps_env()
    if mps_daemon_running():
        return False
    if not shutil.which("nvidia-cuda-mps-control"):
        raise RuntimeError(
            "nvidia-cuda-mps-control not found on PATH.  Load the CUDA module, or run "
            "with --mps-replicas 1."
        )
    os.makedirs(mps_pipe_directory(), exist_ok=True)
    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True, timeout=30)
    except subprocess.CalledProcessError as e:
        if mps_daemon_running():
            return False
        raise RuntimeError(
            f"'nvidia-cuda-mps-control -d' exited with status {e.returncode} and no "
            f"daemon serves {mps_pipe_directory()}.  Is another user's MPS daemon "
            "holding the GPU?  Run with --mps-replicas 1 to go without MPS."
        ) from e
    # Let the daemon come up before ranks attach.
    time.sleep(2)
    return True


def stop_mps(timeout: float = 10.0) -> None:
    """Ask the CUDA MPS control daemon of this job's pipe directory to quit."""
    if not shutil.which("nvidia-cuda-mps-control"):
        return
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input="quit\n",
            text=True,
            check=False,
            timeout=timeout,
            capture_output=True,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def validate_mpi_install() -> dict:
    """Validate that mpi4py can load and report which MPI library it uses.

    Returns a dict with keys:
        ok (bool): True if mpi4py imported and MPI_Init succeeded
        mpi_library (str): path to the linked libmpi.so (if detectable)
        mpi_version (str): MPI implementation version string
        error (str): error message if ok is False
    """
    result = {"ok": False, "mpi_library": "", "mpi_version": "", "error": ""}
    try:
        from mpi4py import MPI
        result["mpi_version"] = MPI.Get_library_version().strip()
        result["ok"] = True
    except ImportError as e:
        result["error"] = f"mpi4py import failed: {e}"
    except Exception as e:
        result["error"] = f"MPI init failed: {e}"

    # Try to find the linked libmpi
    try:
        import ctypes.util
        lib = ctypes.util.find_library("mpi")
        if lib:
            result["mpi_library"] = lib
    except Exception:
        pass

    return result
