import re
import json
import os
import warnings

import pandas as pd
import psutil

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)  # pdbfixer


def make_contact_frequency_dictionary(freq_files: list) -> pd.DataFrame:
    """
    Deprecated in favor of make_contact_dataframe().
    go through a list of frequency files and record all of the frequencies for
    each replica.

    freq_files : list
        List of paths to each contact frequency file, presorted.

    Returns : Dict
        The Dictionary of contact keys and frequency lists.
    """
    contact_dictionary = {}

    regex = r"\w:\w+:\d+\s+\w:\w+:\d+"
    # go through each of the contact files and fill in the lists
    for i, file in enumerate(freq_files):
        with open(file, "r") as freqs:
            for line in freqs.readlines():
                if re.search(regex, line):
                    line = line.strip()
                    first, second, num_str = line.split()
                    label = first + "-" + second

                    if label not in contact_dictionary.keys():
                        contact_dictionary[label] = [0 for n in range(i)]
                        contact_dictionary[label].append(float(num_str))
                    else:
                        contact_dictionary[label].append(float(num_str))

        # Extend all the lists before opening the next freq_file
        for key in contact_dictionary.keys():
            if i > 0 and len(contact_dictionary[key]) != i + 1:
                length = len(contact_dictionary[key])
                extend = (i + 1) - length
                contact_dictionary[key].extend([0 for n in range(extend)])

    return contact_dictionary


def sort_dictionary_values(dictionary:dict) -> dict:
    return dict(sorted(dictionary.items(), key=lambda item: -item[1]))


def parse_id(contact:str) -> dict:
    """
    take the contact name (column id) and return a dictionary of
    the residue A descriptors and residue B descriptors
    """
    chaina, resna, resida, chainb, resnb, residb = re.split(":|-", contact)
    ### for combined contact data, the prepended name needs to be removed from
    ### chain a
    ##### This might break something if multiple contacts
    #### are going into the keys of another dictionary because
    #### duplicate names will be overwritten.
    ## shouldn't be a problem for averaging functions because combined data
    ## will be produced from pre-averaged data
    ## to_heatmap() will not give correct results as is - need to prepare
    ## the data with original names for that....

    if "_" in chaina:
        chaina = chaina.split("_")[1]

    return {
        "chaina": chaina,
        "resna": resna,
        "resida": resida,
        "chainb": chainb,
        "resnb": resnb,
        "residb": residb,
    }


def split_id(contact:str) -> dict:
    """
    take the contact name and split it into its two residue parts
    returns a dictionary where 'resa' will contain 'CH:RES:NUM'
    """
    resa, resb = re.split("-", contact)
    return {"resa": resa, "resb": resb}


def multi_intersection(lists:list[list], cutoff:float|int|None=None, 
                       verbose:bool=False)-> list:
    """
    Return the intersection of the values in lists.
    Parameters
    ----------
    lists : list of lists
        The lists of values to identify the shared elements from.

    cutoff : float or int
        If not None, return the intersection of a subset of lists that meet the criteria.
        float < 1 will only include lists that have a length of cutoff percent of
        the longest list.
        int > 1 will only include lists that are longer than cutoff.

    verbose : bool
        If True, print the number of lists that were provided as input and the number
        of lists that were used in constructing the intersection.

    Returns
    -------
    list
    intersection of values in lists.

    """

    initial = len(lists)
    if cutoff is not None and cutoff < 1:
        longest_len = max([len(data) for data in lists])
        lists = [data for data in lists if len(data) > longest_len * cutoff]
    elif cutoff is not None and cutoff > 1:
        lists = [data for data in lists if len(data) > cutoff]

    final = len(lists)
    set1 = set(lists[0])
    setlist = [set(data) for data in lists[1:]]
    if verbose == True:
        print(f"n lists initial: {initial} \nn lists final: {final}")
    return sorted(list(set1.intersection(*setlist)))


def sort_nested_dict(d: dict) -> dict:
    """
    Sort the split sum dictionary. This is expecting the keys of the nested dictionary to be
    in the form of "A:ALA:5".
    """
    sorted_dict = {}
    for outer_key, nested_dict in d.items():
        sorted_keys = sorted(
            nested_dict.keys(),
            key=lambda x: (x.split(":")[0], int(x.split(":")[-1])),
        )
        sorted_nested_dict = {key: nested_dict[key] for key in sorted_keys}
        sorted_dict[outer_key] = sorted_nested_dict
    return sorted_dict


def get_resources():
    resources = {
        "num_cores": psutil.cpu_count(logical=False),  # physical cores
        "num_threads": psutil.cpu_count(logical=True),  # includes hyperthreads
        "total_ram_gb": psutil.virtual_memory().total / 1e9,
        "available_ram_gb": psutil.virtual_memory().available / 1e9,
        "available_ram_mb": psutil.virtual_memory().available / 1e6,
    }
    return resources


##################### Simulation Prep ######################
def fix_pdb(input_pdb, output_pdb, pH=7.0, keep_water=False, replace_nonstandard_resis=True):
    '''
    PDBFixer convenience function 
    
    '''
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
    state = simulation.context.getState(getPositions=True,
                                        enforcePeriodicBox=True)
    return simulation.topology, state.getPositions()

class OMMSetup:
    '''
    Class to piece together an openmm simulation object
    '''

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
        from openmm.app import PDBFile, Modeller, ForceField, Simulation, PME, HBonds
        from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, XmlSerializer
        from openmm.unit import nanometer, bar, picosecond, femtoseconds, molar, atomic_mass_unit

        self.structures = structures
        self.nonbonded_cutoff = nonbonded_cutoff*nanometer
        self.integrator_type = LangevinMiddleIntegrator
        self.forcefields = forcefields
        self.temperature = temperature
        self.pressure = pressure*bar
        self.box_shape = box_shape
        self.padding = padding*nanometer
        self.name = name
        self.Hmass = Hmass*atomic_mass_unit
        self.timestep = timestep
        
    '''
    structures : dict
        dict of keys of user supplied names and values of paths to prepared PDB 
        files for each component of the system.
        Example
        -------
        structures = ['lysozyme':'./253L.pdb']

    '''
    def model(self):
        # modeler components
        pdb_file = self.structures[0]
        pdb = PDBFile(pdb_file)
        modeller = Modeller(pdb.topology, pdb.positions)
        if len(self.structures) > 1:
            for structure in self.structures[1:]:
                pdb_file = structure
                pdb = PDBFile(pdb_file)
                modeller.add(pdb.topology, pdb.positions)
        self.modeller = modeller
        self.forcefield = ForceField(*self.forcefields)
        self.modeller.addSolvent(self.forcefield, padding=self.padding,
                            ionicStrength=0.1*molar, model='tip3p',
                            boxShape=self.box_shape)
    
    
    def make_system(self):
        # create system object
        system = self.forcefield.createSystem(self.modeller.topology, 
                                         nonbondedMethod=PME,
                                         nonbondedCutoff=self.nonbonded_cutoff, 
                                         constraints=HBonds,
                                         hydrogenMass=self.Hmass)
        # Add pressure control
        system.addForce(MonteCarloBarostat(self.pressure, self.temperature))
        self.system=system


    def make_simulation(self):
        integrator = self.integrator_type(self.temperature, 
                                          1/picosecond, 
                                          self.timestep*femtoseconds)
        simulation = Simulation(self.modeller.topology, self.system, integrator)
        simulation.context.setPositions(self.modeller.positions)
        simulation.minimizeEnergy()
        self.simulation = simulation


    def save(self, output,):
        '''
        save gromacs files and openmm system files

        output : str
            Path to output. Directory will be created if none exists.
        '''
        # create folders within the output directory 
        os.makedirs(f'{output}',exist_ok=True)
        directories = ['system', 'structures']
        for directory in directories:
            os.makedirs(f'{output}/{directory}', exist_ok=True)

        # save the system and minimized structure
        topology, positions = top_pos_from_sim(self.simulation)
        with open(f'{output}/system/{self.name}_system.xml', 'w') as outfile:
            outfile.write(XmlSerializer.serialize(self.system))
        #os.chmod(file, stat.S_IREAD) #set to read only to prevent deletion
        with open(f'{output}/structures/{self.name}_minimized.pdb', 'w') as f:
            PDBFile.writeFile(topology, positions, f)

class RunConfig:
    """
    Configuration manager for ChACRA HREMD runs.

    Reads and writes ``chacra_run.json`` in the project root directory.
    On the first run this file is created with all resolved parameters
    (including the full temperature list).  On subsequent runs it is loaded
    automatically so the user does not need to re-supply every CLI flag.
    CLI arguments always take precedence over values stored in the JSON.

    Parameters
    ----------
    config_file : str or None
        Path to an existing ``chacra_run.json``.  If *None* an empty config
        is created using built-in defaults.

    Attributes
    ----------
    defaults : dict
        Hard-coded default values for every parameter.
    config : dict
        The resolved configuration.  Starts from ``defaults`` and is
        overridden by any values loaded from *config_file*.
    """

    #: Default path written/read in the project root directory.
    DEFAULT_PATH: str = "chacra_run.json"

    defaults: dict = {
        "n_jobs": None,
        "n_cycles": 1000,
        "n_systems": None,
        "min_temp": 290.0,
        "max_temp": 450.0,
        "temps": None,          # populated after first run
        "structure_file": None,
        "system_file": None,
        "steps_per_cycle": 1000,
        "save_interval": 10,
        "checkpoint_interval": 500,
        "warmup_steps": 0,
        "lambda_selection": "protein",
        "output_selection": "protein",
        "timestep": 2,
        "current_run": 0,
    }

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file
        # Start from defaults, then overlay file contents
        self.config: dict = dict(self.defaults)

        if config_file is not None:
            if os.path.exists(config_file):
                self._load(config_file)
            else:
                print(f"[RunConfig] Config file not found: {config_file}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, path: str | None = None) -> None:
        """
        Serialise ``self.config`` to JSON.

        Parameters
        ----------
        path : str or None
            Destination path.  Defaults to :attr:`DEFAULT_PATH`
            (``chacra_run.json`` in the current working directory).
        """
        dest = path or self.DEFAULT_PATH
        with open(dest, "w") as fh:
            json.dump(self.config, fh, indent=2)
        print(f"[RunConfig] Config written to {dest}")

    def update(self, **kwargs) -> None:
        """
        Update individual config keys, ignoring keys whose value is *None*
        (so CLI arg defaults don't silently erase stored values).

        Parameters
        ----------
        **kwargs
            Key-value pairs to merge into ``self.config``.
        """
        for key, value in kwargs.items():
            if value is not None:
                self.config[key] = value

    def apply_to_namespace(self, namespace) -> None:
        """
        Back-fill an ``argparse.Namespace`` with config values for any
        attribute that is currently *None*.  This lets CLI-supplied arguments
        always win while filling gaps from the JSON.

        Parameters
        ----------
        namespace : argparse.Namespace
        """
        for key, value in self.config.items():
            if getattr(namespace, key, None) is None and value is not None:
                setattr(namespace, key, value)

    def get(self, key: str, default=None):
        """Return a config value, falling back to *default*."""
        return self.config.get(key, default)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: str) -> None:
        """Load JSON from *path* and overlay onto ``self.config``."""
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            self.config.update(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[RunConfig] Could not read config file '{path}': {exc}")
