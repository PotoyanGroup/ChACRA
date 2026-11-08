# -*- coding: utf-8 -*-
"""
Simulation setup utilities for ChACRA.

Provides helpers for preparing OpenMM systems and running energy minimization.
All OpenMM imports are deferred to each method so that users who only need
the analysis side of ChACRA are not required to have OpenMM installed.
"""

import os


def fix_pdb(input_pdb, output_pdb, pH=7.0, keep_water=False, replace_nonstandard_resis=True):
    """
    PDBFixer convenience function.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_pdb : str
        Path to write the fixed PDB file.
    pH : float
        pH used to determine protonation states when adding missing hydrogens.
    keep_water : bool
        If True, retain crystallographic water molecules.
    replace_nonstandard_resis : bool
        If True, replace nonstandard residues with their standard equivalents.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    # https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html
    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    if replace_nonstandard_resis:
        fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keep_water)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb, 'w'))


def top_pos_from_sim(simulation):
    """Return (topology, positions) from a running OpenMM Simulation."""
    state = simulation.context.getState(getPositions=True,
                                        enforcePeriodicBox=True)
    return simulation.topology, state.getPositions()


class OMMSetup:
    """
    Helper class to compose an OpenMM simulation object step by step.

    Parameters
    ----------
    structures : list of str
        Paths to prepared PDB files for each component of the system.
        Example: ['lysozyme.pdb']
    nonbonded_cutoff : float
        Non-bonded cutoff distance in nanometers.
    forcefields : list of str
        Force field XML files to load.
    temperature : float
        Simulation temperature in Kelvin.
    pressure : float
        Simulation pressure in bar.
    box_shape : str
        Solvent box shape passed to Modeller.addSolvent (e.g. 'dodecahedron').
    padding : float
        Padding around the solute in nanometers.
    name : str
        Base name used for output files.
    Hmass : float
        Hydrogen mass in atomic mass units (>2 enables longer timesteps via HMR).
    timestep : int
        Integration timestep in femtoseconds.
    forcefield : str
        Forcefield(s) (comma separated) to use for the simulation.
    """

    def __init__(self,
                 structures,
                 nonbonded_cutoff=1,
                 forcefields=['amber14-all.xml', 'amber14/tip3pfb.xml'],
                 temperature=310.0,
                 pressure=1,
                 box_shape='dodecahedron',
                 padding=1.0,
                 name='system',
                 Hmass=2.0,
                 timestep=2,
                 ):
        from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
        from openmm.unit import nanometer, bar, atomic_mass_unit

        self.structures = structures
        self.nonbonded_cutoff = nonbonded_cutoff * nanometer
        self.integrator_type = LangevinMiddleIntegrator
        self.forcefields = forcefields
        self.temperature = temperature
        self.pressure = pressure * bar
        self.box_shape = box_shape
        self.padding = padding * nanometer
        self.name = name
        self.Hmass = Hmass * atomic_mass_unit
        self.timestep = timestep

    def model(self):
        """Load structures, build the Modeller topology, and add solvent."""
        from openmm.app import PDBFile, Modeller, ForceField
        from openmm.unit import molar

        pdb_file = self.structures[0]
        pdb = PDBFile(pdb_file)
        modeller = Modeller(pdb.topology, pdb.positions)
        if len(self.structures) > 1:
            for structure in self.structures[1:]:
                pdb = PDBFile(structure)
                modeller.add(pdb.topology, pdb.positions)
        self.modeller = modeller
        self.forcefield = ForceField(*self.forcefields)
        self.modeller.addSolvent(self.forcefield, padding=self.padding,
                                 ionicStrength=0.1 * molar, model='tip3p',
                                 boxShape=self.box_shape)

    def make_system(self):
        """Create the OpenMM System object with PME, HBond constraints, and a barostat."""
        from openmm.app import PME, HBonds
        from openmm import MonteCarloBarostat

        system = self.forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=self.nonbonded_cutoff,
            constraints=HBonds,
            hydrogenMass=self.Hmass,
        )
        system.addForce(MonteCarloBarostat(self.pressure, self.temperature))
        self.system = system

    def make_simulation(self):
        """Build the Simulation, set positions, and run energy minimization."""
        from openmm.app import Simulation
        from openmm.unit import picosecond, femtoseconds

        integrator = self.integrator_type(
            self.temperature,
            1 / picosecond,
            self.timestep * femtoseconds,
        )
        simulation = Simulation(self.modeller.topology, self.system, integrator)
        simulation.context.setPositions(self.modeller.positions)
        simulation.minimizeEnergy()
        self.simulation = simulation

    def save(self, output):
        """
        Write the OpenMM system XML and minimized PDB to *output* directory.

        Parameters
        ----------
        output : str
            Destination directory.  Created if it does not exist.
        """
        from openmm import XmlSerializer
        from openmm.app import PDBFile

        os.makedirs(output, exist_ok=True)
        for directory in ['system', 'structures']:
            os.makedirs(f'{output}/{directory}', exist_ok=True)

        topology, positions = top_pos_from_sim(self.simulation)
        with open(f'{output}/system/{self.name}_system.xml', 'w') as outfile:
            outfile.write(XmlSerializer.serialize(self.system))
        with open(f'{output}/structures/{self.name}_minimized.pdb', 'w') as f:
            PDBFile.writeFile(topology, positions, f)

def build_hremd_base_state(
    system_file: str,
    structure_file: str,
    lambda_selection: str,
    temperature: float,
    timestep: float,
):
    """
    Builds the base simulation state and REST-scaled system for HREMD.

    This function reads a system XML and structure PDB, applies femto REST2
    scaling to the specified lambda selection, and generates a thermalized base
    state (positions, velocities, forces). 
    
    It abstracts away the boilerplate required to initialize a `femto.md.hremd`
    compatible system.

    Parameters
    ----------
    system_file : str
        Path to the OpenMM system XML file.
    structure_file : str
        Path to the structure PDB file.
    lambda_selection : str
        MDAnalysis selection string for the atoms to undergo REST2 scaling.
    temperature : float
        The base temperature (K) to thermalize the state to.
    timestep : float
        The integration timestep in femtoseconds.

    Returns
    -------
    tuple
        A tuple containing:
        - system (openmm.System): The REST2-scaled OpenMM system.
        - structure (mdtop.Topology): The system topology.
        - base_state (openmm.State): The fully initialized and thermalized 
          base OpenMM state.
    """
    import femto.md.config
    import femto.md.rest
    import MDAnalysis as mda
    import mdtop
    from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    with open(system_file) as f:
        system = XmlSerializer.deserialize(f.read())

    u = mda.Universe(structure_file)
    solute_idxs = set(u.select_atoms(lambda_selection).atoms.ix)

    rest_config = femto.md.config.REST(scale_torsions=True, scale_nonbonded=True)
    femto.md.rest.apply_rest(system, solute_idxs, rest_config)

    pdb = PDBFile(structure_file)
    structure = mdtop.Topology.from_file(structure_file)

    integrator = LangevinMiddleIntegrator(
        temperature, 1 / unit.picosecond, timestep * unit.femtosecond
    )
    integrator.setRandomNumberSeed(12345)
    
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temperature, 12345)

    base_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getForces=True,
        getEnergy=True,
        enforcePeriodicBox=True,
    )
    return system, structure, base_state


def _run_sim_worker(builder_func, steps, kwargs):
    """Worker process that builds and runs the simulation."""
    import openmm.unit
    
    # Build the simulation fresh inside this process
    simulation = builder_func(**kwargs)
    
    # Run the simulation
    simulation.step(steps)
    
    # Return the final state potential energy
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system)


def run_concurrently_with_mps(builder_func, configs, steps=1000):
    """
    Run multiple OpenMM simulations concurrently on a single GPU using CUDA MPS.
    
    This function delegates the OpenMM initialization to child processes to bypass
    OpenMM's inability to pickle and share Simulation objects across processes. 
    It automatically configures the MPS environment for the workers.
    
    Parameters
    ----------
    builder_func : callable
        A function that takes **kwargs and returns an openmm.app.Simulation object.
        This function will be executed in the child processes.
    configs : list of dict
        A list of kwargs dictionaries to pass to builder_func for each replica.
    steps : int
        Number of steps to run for each simulation.
        
    Returns
    -------
    list of float
        The final potential energy of each simulation.

    Example
    -------
    >>> def my_sim_builder(temperature):
    ...     # This runs inside the child process!
    ...     from chacra.simulation import build_hremd_base_state
    ...     from openmm.app import Simulation
    ...     from openmm import LangevinMiddleIntegrator, unit
    ...     
    ...     system, structure, base_state = build_hremd_base_state(
    ...         system_file="system.xml", structure_file="structure.pdb", 
    ...         lambda_selection="protein", temperature=290.0, timestep=2.0
    ...     )
    ...     
    ...     integrator = LangevinMiddleIntegrator(
    ...         temperature * unit.kelvin, 1/unit.picosecond, 2.0*unit.femtosecond
    ...     )
    ...     sim = Simulation(structure.topology, system, integrator)
    ...     sim.context.setState(base_state)
    ...     return sim
    ...
    >>> configs = [
    ...     {"temperature": 290.0},
    ...     {"temperature": 310.0},
    ... ]
    >>> energies = run_concurrently_with_mps(my_sim_builder, configs, steps=5000)
    """
    import multiprocessing as mp
    import os
    import time
    
    num_replicas = len(configs)
    
    # 1. Configure the environment for MPS
    # Tell MPS to evenly divide the GPU threads among the replicas
    thread_pct = max(1, 200 // num_replicas)
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(thread_pct)
    
    print(f"Starting {num_replicas} concurrent simulations (MPS thread %: {thread_pct})")
    
    # 2. Use multiprocessing "spawn" to ensure completely clean CUDA contexts
    ctx = mp.get_context("spawn")
    
    start_time = time.time()
    
    # 3. Launch the workers
    with ctx.Pool(processes=num_replicas) as pool:
        # We pass the builder_func, the number of steps, and the specific config dict
        results = pool.starmap(
            _run_sim_worker, 
            [(builder_func, steps, cfg) for cfg in configs]
        )
        
    wall_time = time.time() - start_time
    print(f"All simulations finished in {wall_time:.2f} seconds.")
    
    return results


# --------------------------------------------------------------------------- #
# femto HREMD configuration                                                    #
# --------------------------------------------------------------------------- #

FEMTO_DEFAULT_TEMPERATURE_K = 298.15
"""femto's own default for ``HREMD.temperature`` (K).

ChACRA runs before 2026-11 never set the field, so femto scored every stored
reduced potential, and the swap acceptance test, with beta at this temperature
while the thermostat ran at ``min_temp`` (290 K by default).  Analysis code
rescales such legacy energies by ``298.15 / min_temp``; see
:func:`chacra.reweight.OccupancyStore.load`.
"""


def build_hremd_config(
    n_warmup_steps: int,
    n_steps_per_cycle: int,
    n_cycles: int,
    trajectory_interval: int | None,
    checkpoint_interval: int | None,
    temperature_K: float,
    verbose: bool = True,
):
    """
    Return a ``femto.md.config.HREMD`` whose ``temperature`` is set explicitly.

    femto uses ``temperature`` for beta in every stored reduced potential and in
    the replica-exchange acceptance test.  Its default (298.15 K) silently
    disagrees with the Langevin thermostat whenever ``min_temp`` is anything
    else, so always pass the base (coldest, unscaled) REST2 temperature here.
    """
    import femto.md.config
    import openmm.unit

    if temperature_K is None or not float(temperature_K) > 0:
        raise ValueError(f"temperature_K must be a positive number, got {temperature_K!r}")
    config = femto.md.config.HREMD(
        temperature=float(temperature_K) * openmm.unit.kelvin,
        n_warmup_steps=n_warmup_steps,
        n_steps_per_cycle=n_steps_per_cycle,
        n_cycles=n_cycles,
        trajectory_interval=trajectory_interval,
        checkpoint_interval=checkpoint_interval,
    )
    if verbose:
        print(f"HREMD reduced potentials and swap test evaluated at {float(temperature_K):.2f} K")
    return config


def femto_run_hremd_kwargs(save_final_coords: bool = False) -> dict:
    """
    Keyword arguments for ``femto.md.hremd.run_hremd`` that only newer femto
    forks accept.

    ``save_final_coords=False`` (Dan-Burns/femto from 457574f on) skips the final
    all-rank coordinate gather.  Older installs raise ``TypeError`` on the
    keyword, so it is passed only when the signature has it.
    """
    import inspect

    import femto.md.hremd

    params = inspect.signature(femto.md.hremd.run_hremd).parameters
    return {"save_final_coords": save_final_coords} if "save_final_coords" in params else {}


def barostat_temperature_from_xml(system_file: str) -> float | None:
    """Default temperature (K) of the ``MonteCarloBarostat`` in an OpenMM system XML, or None."""
    import re

    with open(system_file) as fh:
        xml = fh.read()
    for m in re.finditer(r"<Force\b[^>]*>", xml):
        tag = m.group(0)
        if 'type="MonteCarloBarostat"' in tag:
            t = re.search(r'defaultTemperature="([0-9.eE+-]+)"', tag)
            if t:
                return float(t.group(1))
    return None


def hydrogen_mass_from_xml(system_file: str) -> float | None:
    """
    Repartitioned hydrogen mass (amu) of an OpenMM system XML, or None when the
    system has no hydrogen mass repartitioning (HMR).

    OpenMM's ``hydrogenMass`` moves mass onto every hydrogen bound to a heavy
    atom except water, so an HMR system has a population of particles at the
    chosen mass (1.5-4.5 amu) beside water's 1.008 amu.  The most common mass
    in that window is returned.
    """
    import re
    from collections import Counter

    counts: Counter = Counter()
    with open(system_file) as fh:
        for line in fh:
            if "<Particle" not in line:
                continue
            for m in re.finditer(r'<Particle mass="([0-9.eE+-]+)"', line):
                mass = float(m.group(1))
                if 1.2 <= mass <= 4.5:
                    counts[round(mass, 3)] += 1
    if not counts:
        return None
    return float(counts.most_common(1)[0][0])
