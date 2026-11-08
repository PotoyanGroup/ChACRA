"""
The femto HREMD configuration ChACRA hands to ``femto.md.hremd.run_hremd``.

femto's ``HREMD.temperature`` sets beta for every stored reduced potential and
for the swap acceptance test.  ChACRA runs before 2026-11 never set it, so
femto's default (298.15 K) was used while the thermostat ran at ``min_temp``.
These tests pin the fix and document the trap.
"""

from __future__ import annotations

import pytest

from chacra.simulation import (
    FEMTO_DEFAULT_TEMPERATURE_K,
    barostat_temperature_from_xml,
)

femto_config = pytest.importorskip("femto.md.config")
openmm_unit = pytest.importorskip("openmm.unit")


def test_femto_default_temperature_is_the_documented_trap():
    cfg = femto_config.HREMD()
    assert cfg.temperature.value_in_unit(openmm_unit.kelvin) == pytest.approx(298.15)
    assert FEMTO_DEFAULT_TEMPERATURE_K == pytest.approx(298.15)


def test_build_hremd_config_sets_temperature():
    from chacra.simulation import build_hremd_config

    cfg = build_hremd_config(
        n_warmup_steps=0, n_steps_per_cycle=1000, n_cycles=10,
        trajectory_interval=5, checkpoint_interval=5, temperature_K=290.0,
        verbose=False,
    )
    assert cfg.temperature.value_in_unit(openmm_unit.kelvin) == pytest.approx(290.0)
    assert cfg.n_steps_per_cycle == 1000
    assert cfg.n_cycles == 10
    assert cfg.trajectory_interval == 5
    assert cfg.checkpoint_interval == 5


@pytest.mark.parametrize("bad", [None, 0.0, -5.0])
def test_build_hremd_config_rejects_missing_temperature(bad):
    from chacra.simulation import build_hremd_config

    with pytest.raises(ValueError):
        build_hremd_config(
            n_warmup_steps=0, n_steps_per_cycle=1, n_cycles=1,
            trajectory_interval=None, checkpoint_interval=None, temperature_K=bad,
            verbose=False,
        )


def test_run_hremd_kwargs_match_installed_signature():
    import inspect

    import femto.md.hremd

    from chacra.simulation import femto_run_hremd_kwargs

    kw = femto_run_hremd_kwargs(save_final_coords=False)
    has = "save_final_coords" in inspect.signature(femto.md.hremd.run_hremd).parameters
    assert kw == ({"save_final_coords": False} if has else {})


def test_barostat_temperature_from_xml(tmp_path):
    xml = tmp_path / "system.xml"
    xml.write_text(
        '<System><Forces>'
        '<Force cutoff="1" name="NonbondedForce" type="NonbondedForce" version="4"/>'
        '<Force defaultPressure="1" defaultTemperature="290" forceGroup="0" '
        'frequency="25" name="MonteCarloBarostat" randomSeed="0" '
        'type="MonteCarloBarostat" version="3"/>'
        '</Forces></System>'
    )
    assert barostat_temperature_from_xml(str(xml)) == pytest.approx(290.0)
    xml.write_text('<System><Forces><Force type="NonbondedForce"/></Forces></System>')
    assert barostat_temperature_from_xml(str(xml)) is None


def test_hydrogen_mass_from_xml(tmp_path):
    from chacra.simulation import hydrogen_mass_from_xml

    xml = tmp_path / "system.xml"
    particles = ['<Particle mass="1.007947"/>'] * 20 + ['<Particle mass="2"/>'] * 5 + \
                ['<Particle mass="11.018"/>'] * 5 + ['<Particle mass="15.99943"/>'] * 10
    xml.write_text("<System><Particles>" + "".join(particles) + "</Particles></System>")
    assert hydrogen_mass_from_xml(str(xml)) == pytest.approx(2.0)
    xml.write_text("<System><Particles>" + "".join(['<Particle mass="1.007947"/>'] * 20 +
                   ['<Particle mass="12.01"/>'] * 5) + "</Particles></System>")
    assert hydrogen_mass_from_xml(str(xml)) is None
