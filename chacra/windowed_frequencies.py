"""
Windowed contact frequency computation via get-contact-frequencies.

For each state, uses Polars to lazily slice the per-frame atomistic TSV
(or Parquet) to a frame window, writes a temp file, calls
``get-contact-frequencies`` on it, and deletes the temp.  This is fast
because Polars streams the filter and get-contact-frequencies is already
optimized for frequency calculation.

Supports multi-run layouts via glob patterns in ``file_pattern``.

Usage::

    chacra windowed-freqs \\
      --contacts_dir contacts/ \\
      --output_dir convergence_freqs/ \\
      --file_pattern "rep_{state}_contacts.tsv" \\
      --n_states 24 \\
      --percentiles 50 60 70 80 90 100 \\
      --n_jobs 4
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_fmt(path: str) -> str:
    return "parquet" if Path(path).suffix == ".parquet" else "tsv"


def _max_frame_tsv(path: str) -> int:
    """Read the last non-empty line of a getcontacts TSV to get the max frame.

    getcontacts TSVs are frame-sorted so the last line has the highest frame.
    This reads only the tail of the file — effectively instant for any size.
    """
    with open(path, "rb") as f:
        # Seek to last 4KB; even the longest lines are much shorter
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf-8", errors="replace")

    for line in reversed(tail.strip().splitlines()):
        if line.startswith("#") or not line.strip():
            continue
        try:
            return int(line.split("\t")[0])
        except ValueError:
            continue
    return -1


def _max_frame_parquet(path: str) -> int:
    import polars as pl
    return pl.scan_parquet(path).select(pl.col("frame").max()).collect().item()


def _n_frames_tsv(path: str) -> int:
    """Max frame + 1 (frames are 0-indexed contiguous integers)."""
    return _max_frame_tsv(path) + 1


def _n_frames_parquet(path: str) -> int:
    import polars as pl
    return pl.scan_parquet(path).select(pl.col("frame").n_unique()).collect().item()


def _n_frames(path: str, fmt: str) -> int:
    return _n_frames_parquet(path) if fmt == "parquet" else _n_frames_tsv(path)


def _total_frames_for_state(files: list[tuple[str, str]]) -> int:
    """Sum frame counts across all run files for one state."""
    return sum(_n_frames(p, f) for p, f in files)


# ---------------------------------------------------------------------------
# File discovery — supports glob wildcards for multi-run layouts
# ---------------------------------------------------------------------------

def _discover_state_files(
    contacts_dir: str,
    file_pattern: str,
    n_states: int | None,
) -> list[tuple[int, list[tuple[str, str]]]]:
    """
    Returns list of (state_idx, [(path, fmt), ...]) sorted by state.
    Glob patterns in file_pattern (e.g. ``run_*/contacts/...``) are expanded.
    """
    base = Path(contacts_dir)
    is_glob = "*" in file_pattern or "?" in file_pattern

    def _files_for(i: int) -> list[tuple[str, str]]:
        pat = file_pattern.format(state=i)
        if is_glob:
            return [(str(p), _detect_fmt(str(p)))
                    for p in sorted(base.glob(pat)) if p.exists()]
        p = base / pat
        if p.exists():
            return [(str(p), _detect_fmt(str(p)))]
        alt = p.with_suffix(".parquet" if p.suffix != ".parquet" else ".tsv")
        return [(str(alt), _detect_fmt(str(alt)))] if alt.exists() else []

    results = []
    if n_states is not None:
        for i in range(n_states):
            fs = _files_for(i)
            if fs:
                results.append((i, fs))
    else:
        i = 0
        while True:
            fs = _files_for(i)
            if not fs:
                break
            results.append((i, fs))
            i += 1
    return results


# ---------------------------------------------------------------------------
# Core: slice + get-contact-frequencies
# ---------------------------------------------------------------------------

def _slice_and_freq(
    state_idx: int,
    files: list[tuple[str, str]],
    end_frame: int,
    output_dir: str,
) -> str:
    """
    For one state: copy per-frame contact lines where frame <= end_frame
    to a temp file, run get-contact-frequencies on it, return path of the
    output freq file.

    For TSV: raw line-by-line copy (preserves getcontacts format exactly).
    Since files are frame-sorted, we stop reading as soon as frame > end_frame.
    Multi-run: files are read in order with cumulative frame offsets.
    """
    out_path = os.path.join(output_dir, f"freqs_state_{state_idx}.tsv")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(output_dir) / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_file = tmp_dir / f"_tmp_state_{state_idx}.tsv"

    offset = 0
    with open(tmp_file, "w") as out:
        for path, fmt in files:
            if fmt == "parquet":
                # Parquet path: use Polars to filter and write as TSV
                import polars as pl
                lf = (
                    pl.scan_parquet(path)
                    .with_columns((pl.col("frame") + offset).alias("frame"))
                    .filter(pl.col("frame") <= end_frame)
                )
                df = lf.collect(streaming=True)
                for row in df.iter_rows():
                    parts = [str(x) for x in row if x is not None]
                    out.write("\t".join(parts) + "\n")
                offset += _n_frames(path, fmt)
            else:
                # TSV path: copy lines verbatim (preserves getcontacts format)
                done = False
                with open(path) as fh:
                    for line in fh:
                        if not line.strip() or line[0] == "#":
                            out.write(line)
                            continue
                        try:
                            local_frame = int(line.split("\t", 1)[0])
                        except ValueError:
                            out.write(line)
                            continue
                        global_frame = local_frame + offset
                        if global_frame > end_frame:
                            done = True
                            break
                        if offset > 0:
                            # Rewrite the frame number for multi-run offset
                            out.write(str(global_frame) + "\t" + line.split("\t", 1)[1])
                        else:
                            out.write(line)
                offset += _n_frames(path, fmt)
                if done:
                    break

    # Call get-contact-frequencies
    try:
        result = subprocess.run(
            ["get-contact-frequencies",
             "--input_files", str(tmp_file),
             "--output_file", out_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  [ERR] state {state_idx}: {e.stderr.strip()}")
        raise
    finally:
        tmp_file.unlink(missing_ok=True)

    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_windowed_frequencies(
    contacts_dir: str,
    output_dir: str,
    end_frame: int,
    file_pattern: str = "cont_state_{state}.tsv",
    n_states: int | None = None,
    n_jobs: int = 1,
) -> list[str]:
    """
    Compute contact frequencies from frames [0, end_frame] for each state.

    Parameters
    ----------
    contacts_dir : str
        Base directory for contact file discovery.
    output_dir : str
        Directory to write frequency TSV files into.
    end_frame : int
        Inclusive end frame for the window (global index for multi-run).
    file_pattern : str
        Filename with ``{state}`` placeholder;  ``*`` for multi-run glob.
    n_states : int or None
        Auto-discovered if None.
    n_jobs : int
        Number of states to process in parallel.

    Returns
    -------
    list[str]
        Paths to written frequency files, one per state.
    """
    state_files = _discover_state_files(contacts_dir, file_pattern, n_states)
    if not state_files:
        raise FileNotFoundError(
            f"No contact files found under '{contacts_dir}' "
            f"matching '{file_pattern}'"
        )

    paths = Parallel(n_jobs=n_jobs)(
        delayed(_slice_and_freq)(idx, files, end_frame, output_dir)
        for idx, files in state_files
    )

    return list(paths)


def percentile_windowed_frequencies(
    contacts_dir: str,
    base_output_dir: str,
    percentiles: list[float] | None = None,
    file_pattern: str = "cont_state_{state}.tsv",
    n_states: int | None = None,
    reference_state: int = 0,
    n_jobs: int = 1,
) -> dict[float, list[str]]:
    """
    Compute contact frequencies at multiple trajectory percentiles.

    Slices each state's file(s) to frames [0, pct% of total], calls
    ``get-contact-frequencies``, and writes the result.  Processing of
    states is parallelized with ``n_jobs``.

    Parameters
    ----------
    contacts_dir : str
        Base directory for contact file discovery.
    base_output_dir : str
        Root output directory; ``{pct}_percent/`` subdirs are created.
    percentiles : list of float or None
        Defaults to ``[50, 60, 70, 80, 90, 100]``.
    file_pattern : str
        Filename with ``{state}`` placeholder; ``*`` for multi-run.
    n_states : int or None
        Auto-discovered if None.
    reference_state : int
        State used to determine total frame count (default 0).
    n_jobs : int
        Number of states to process in parallel per percentile.

    Returns
    -------
    dict[float, list[str]]
        Percentile → list of written frequency file paths.
    """
    if percentiles is None:
        percentiles = [50, 60, 70, 80, 90, 100]

    state_files = _discover_state_files(contacts_dir, file_pattern, n_states)
    if not state_files:
        raise FileNotFoundError(
            f"No contact files found under '{contacts_dir}' matching '{file_pattern}'"
        )

    ref_files = next(
        (files for idx, files in state_files if idx == reference_state),
        state_files[0][1],
    )
    print(f"Scanning state {reference_state} for total frame count...", flush=True)
    total = _total_frames_for_state(ref_files)
    max_frame = total - 1
    print(f"  {total} total frames (0–{max_frame})\n")

    results: dict[float, list[str]] = {}
    for pct in sorted(percentiles):
        ef = int(round(max_frame * pct / 100.0))
        out_dir = os.path.join(base_output_dir, f"{int(pct)}_percent")
        print(f"[{pct}%] frames 0–{ef} → {out_dir}/", flush=True)
        paths = compute_windowed_frequencies(
            contacts_dir=contacts_dir,
            output_dir=out_dir,
            end_frame=ef,
            file_pattern=file_pattern,
            n_states=len(state_files),
            n_jobs=n_jobs,
        )
        results[pct] = paths

    # Clean up temp dirs
    for pct in percentiles:
        tmp = Path(base_output_dir) / f"{int(pct)}_percent" / "_tmp"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    print("\nDone.")
    for pct, paths in sorted(results.items()):
        out_dir = os.path.join(base_output_dir, f"{int(pct)}_percent")
        print(f"  {int(pct)}%: {len(paths)} states → {out_dir}/")

    return results
