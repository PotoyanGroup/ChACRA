"""``make-simulation`` defaults: the force-field list must carry a water model."""

from chacra.scripts.make_simulation import DEFAULT_FORCEFIELDS, has_water_model


def test_default_forcefields_include_a_water_model():
    ffs = DEFAULT_FORCEFIELDS.split(",")
    assert "amber14-all.xml" in ffs
    assert has_water_model(ffs)


def test_has_water_model_detects_the_common_files():
    assert has_water_model(["amber14-all.xml", "amber14/tip3pfb.xml"])
    assert has_water_model(["amber14-all.xml", "amber14/tip3p.xml"])
    assert has_water_model(["charmm36.xml", "charmm36/water.xml"])
    assert has_water_model(["amber14-all.xml", "amber14/spce.xml"])
    assert not has_water_model(["amber14-all.xml"])
    assert not has_water_model(["amber99sb.xml", "my_ligand.xml"])
