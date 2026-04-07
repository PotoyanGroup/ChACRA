#!/usr/bin/env python3
"""Convert a conda-lock v1 unified YAML file to a conda @EXPLICIT spec file.

The resulting .txt can be installed directly by conda/mamba/micromamba:
    micromamba create -n ENV --file explicit-cuda12.txt

No conda-lock tool required on the target machine.

Usage:
    python tools/lock_to_explicit.py conda-lock.cuda12.yml   # → explicit-cuda12.txt
    python tools/lock_to_explicit.py conda-lock.cuda13.yml   # → explicit-cuda13.txt
"""
import sys
import yaml
from pathlib import Path


def convert(lock_path: str) -> str:
    """Read a conda-lock YAML and return an @EXPLICIT spec string."""
    with open(lock_path) as f:
        lock = yaml.safe_load(f)

    lines = ["@EXPLICIT"]
    for pkg in lock.get("package", []):
        # Skip pip-managed packages (shouldn't exist, but be safe)
        if pkg.get("manager") != "conda":
            continue
        url = pkg["url"]
        md5 = pkg.get("hash", {}).get("md5", "")
        if md5:
            lines.append(f"{url}#{md5}")
        else:
            lines.append(url)

    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <conda-lock.cudaXX.yml> [output.txt]")
        sys.exit(1)

    lock_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        # conda-lock.cuda12.yml → explicit-cuda12.txt
        stem = Path(lock_path).stem  # conda-lock.cuda12
        tag = stem.replace("conda-lock.", "")  # cuda12
        out_path = f"explicit-{tag}.txt"

    spec = convert(lock_path)
    with open(out_path, "w") as f:
        f.write(spec)

    n_pkgs = spec.count("\n") - 1  # minus the @EXPLICIT header
    print(f"Wrote {n_pkgs} packages → {out_path}")


if __name__ == "__main__":
    main()
