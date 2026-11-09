"""
The MPI launcher helpers.

Ranks must inherit the launcher's environment (srun gets ``--export=ALL``: sites
that set ``SLURM_EXPORT_ENV=NONE`` otherwise start them with no PATH), must run
through the launcher's own interpreter rather than a bare ``chacra`` on PATH,
and CUDA MPS must get one daemon per Slurm job.
"""

from __future__ import annotations

import os
import subprocess
import sys

from chacra import mpi


def test_srun_command_exports_environment(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setattr(mpi.shutil, "which",
                        lambda name: "/usr/bin/srun" if name == "srun" else None)
    monkeypatch.setattr(mpi, "_srun_mpi_plugin", lambda srun: "pmix")

    cmd = mpi.build_mpi_command(4)

    assert cmd[:2] == ["/usr/bin/srun", "--mpi=pmix"]
    assert cmd[cmd.index("-n") + 1] == "4"
    assert "--ntasks-per-node=4" in cmd
    assert "--export=ALL" in cmd


def test_srun_ranks_spread_over_nodes(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setattr(mpi.shutil, "which",
                        lambda name: "/usr/bin/srun" if name == "srun" else None)
    monkeypatch.setattr(mpi, "_srun_mpi_plugin", lambda srun: None)

    cmd = mpi.build_mpi_command(8)

    assert cmd[0] == "/usr/bin/srun" and "--mpi=" not in " ".join(cmd)
    assert "--ntasks-per-node=4" in cmd and "--export=ALL" in cmd


def test_user_export_setting_is_kept(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")

    cmd = mpi.build_mpi_command(2, mpi_command="srun --mpi=pmix_v4 --export=ALL,FOO=1")

    assert [p for p in cmd if p.startswith("--export")] == ["--export=ALL,FOO=1"]
    assert cmd[cmd.index("-n") + 1] == "2"


def test_mpirun_command_has_no_export(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_NUM_NODES", raising=False)
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.setattr(mpi, "find_mpi_launcher", lambda: "/opt/bin/mpirun")
    monkeypatch.setattr(mpi, "_is_openmpi", lambda launcher: True)

    assert mpi.build_mpi_command(3) == ["/opt/bin/mpirun", "-np", "3", "--oversubscribe"]


def test_chacra_command_is_path_independent():
    cmd = mpi.chacra_command()

    assert cmd[0] == sys.executable and os.path.isabs(cmd[0])
    assert cmd[1:] == ["-m", "chacra.cli"]

    # The ranks of an srun task may have no PATH at all; the CLI must still start.
    env = {k: v for k, v in os.environ.items() if k != "PATH"}
    env["PATH"] = ""
    out = subprocess.run(cmd + ["--help"], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "run-hremd" in out.stdout


def _var_log_writable():
    return os.path.isdir("/var/log/nvidia-mps") and os.access("/var/log/nvidia-mps", os.W_OK)


def test_configure_mps_env_gives_each_job_its_own_pipe_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    for var in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "CUDA_MPS_LOG_DIR"):
        monkeypatch.delenv(var, raising=False)

    mpi.configure_mps_env()

    pipe = os.environ["CUDA_MPS_PIPE_DIRECTORY"]
    assert pipe == str(tmp_path / f"chacra-mps-{os.getuid()}-4242")
    assert os.path.isdir(pipe)
    assert mpi.mps_pipe_directory() == pipe
    if not _var_log_writable():
        assert os.environ["CUDA_MPS_LOG_DIRECTORY"] == os.path.join(pipe, "logs")
    # An empty pipe directory means no daemon serves this job.
    assert mpi.mps_daemon_running() is False


def test_configure_mps_env_outside_slurm_keeps_default_pipe(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    for var in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "CUDA_MPS_LOG_DIR"):
        monkeypatch.delenv(var, raising=False)

    mpi.configure_mps_env()

    assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
    assert mpi.mps_pipe_directory() == "/tmp/nvidia-mps"


def test_configure_mps_env_respects_user_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "mine"))
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", str(tmp_path / "logs"))

    mpi.configure_mps_env()

    assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "mine")
    assert os.environ["CUDA_MPS_LOG_DIRECTORY"] == str(tmp_path / "logs")


def test_ensure_mps_reuses_a_running_daemon(monkeypatch):
    monkeypatch.setattr(mpi, "configure_mps_env", lambda: None)
    monkeypatch.setattr(mpi, "mps_daemon_running", lambda: True)
    calls = []
    monkeypatch.setattr(mpi.subprocess, "run", lambda *a, **k: calls.append(a))

    assert mpi.ensure_mps() is False
    assert calls == []


def test_ensure_mps_without_control_binary(monkeypatch):
    import pytest

    monkeypatch.setattr(mpi, "configure_mps_env", lambda: None)
    monkeypatch.setattr(mpi, "mps_daemon_running", lambda: False)
    monkeypatch.setattr(mpi.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="nvidia-cuda-mps-control not found"):
        mpi.ensure_mps()
