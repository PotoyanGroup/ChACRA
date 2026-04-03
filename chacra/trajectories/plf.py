"""
ProLIF-based contact fingerprinting — DEPRECATED.

This module previously used the `prolif` package to compute protein-ligand
interaction fingerprints.  ProLIF has been removed from ChACRA's dependencies
and this functionality is no longer maintained.

If you need protein-ligand interaction fingerprints, use ProLIF directly:
    pip install prolif
    import prolif as plf
"""

# ruff: noqa


def _not_implemented(*args, **kwargs):
    raise NotImplementedError(
        "ProLIF-based contact analysis has been deprecated and removed from ChACRA. "
        "Install prolif independently if needed: pip install prolif"
    )


get_prolif_contacts = _not_implemented
get_prolif_freqs = _not_implemented
