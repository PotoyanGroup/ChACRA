from chacra.simulation import OMMSetup, fix_pdb
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="""
        Create an OpenMM simulation system xml file. 
        Provide a PDB file with the protein to simulate. If you provide a .cif
         and there are missing residues, they can be naively fixed with 
        pdbfixer.
        """
    )
    parser.add_argument(
        "-s", "--structure", type=str, required=True, 
        help="The structure to simulate."
    )
    parser.add_argument(
        "-f", "--fix", required=False, default="False",
        action="store_true",
        help="""
        Allow pdbfixer to fix the structure. Protonation state will be 
        set to pH 7.0 and non-standard residues will be replaced. Check the 
        output PDB file to ensure the structure is as expected.
        """
    )
    parser.add_argument(
        "-o", "--output", type=str, required=False,  
        help="""
        The directory to save the simulation system xml file and pdb 
        file. 'system' and 'structure' directories will be created here.
        """,
        default="./"
    )
    parser.add_argument(
        "-t", "--temperature", type=float, required=False,
        help="""
        The temperature that the replica exchange solvent will be run at
        in Kelvin.
        """,
        default=290.0
    )
    parser.add_argument(
        "-n", "--name", type=str, required=False,
        help="""
        Provide a name for the simulation and system to prepend to the output."
        """,
        default="chacra_simulation"
    )
    parser.add_argument(
        "-p", "--pressure", type=float, required=False,
        help="""
        Pressure to simulate the system at in bar."
        """,
        default=1
    )

    parser.add_argument(
        "-d", "--timestep", type=int, required=False,
        help="""
        Timestep in femtoseconds for the HREMD (recorded in chacra_run.json and
        used by run-hremd unless overridden).  4 fs needs hydrogen mass
        repartitioning (--hmass >= 3); use 2 fs otherwise.
        """,
        default=4
    )

    parser.add_argument(
        "-m", "--hmass", type=float, required=False,
        help="""
        Hydrogen mass in amu (hydrogen mass repartitioning; water is left
        alone).  3 amu supports a 4 fs timestep with HBonds constraints; 1.008
        means no repartitioning and a 2 fs timestep.
        """,
        default=3.0
    )

    parser.add_argument(
        "--forcefield", type=str, required=False,
        help="""
        The forcefield to use for the simulation.
        """,
        default="amber14-all.xml"
    )

    args = parser.parse_args()
    pressure = float(args.pressure)
    temperature = float(args.temperature)
    if args.timestep > 2 and args.hmass < 3.0:
        parser.error(
            f"--timestep {args.timestep} fs needs hydrogen mass repartitioning "
            f"(--hmass >= 3); got --hmass {args.hmass}.  Use --timestep 2 or --hmass 3."
        )

    if args.fix:
        print("Running pdbfixer.")
        os.makedirs(f"{args.output}/structures", exist_ok=True)
        fix_pdb(args.structure, f"{args.output}/structures/{args.name}_fixed.pdb")
        structure = f"{args.output}/structures/{args.name}_fixed.pdb"
    else:
        structure = args.structure   

    setup = OMMSetup(
        structures=[structure],
        temperature=temperature,
        name=args.name,
        pressure=pressure,
        Hmass=args.hmass,
        timestep=args.timestep,
        forcefields=[f for f in args.forcefield.split(",")]
    )
    setup.model()
    print(f"Adding solvent.")
    setup.make_system()
    print("Minimizing the system.")
    setup.make_simulation()
    setup.save(args.output)

    # Record how the system was built so run-hremd can pick a consistent timestep.
    from chacra.run_config import RunConfig

    cfg_path = os.path.join(args.output, "chacra_run.json")
    cfg = RunConfig(cfg_path if os.path.exists(cfg_path) else None)
    cfg.update(
        system_file=os.path.join("system", f"{args.name}_system.xml"),
        structure_file=os.path.join("structures", f"{args.name}_minimized.pdb"),
        min_temp=temperature,
        hmass=float(args.hmass),
        timestep=int(args.timestep),
    )
    cfg.write(cfg_path)
    print(f"Recorded system_file, structure_file, min_temp, hmass={args.hmass}, "
          f"timestep={args.timestep} in {cfg_path}")

if __name__ == "__main__":
    main()
