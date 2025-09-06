
"""
This script can do several routine calculations for liquid systems
February 2025

September 2025: I don't use it anymore; not sure it's of much use
except you have a very unstable MLIP
"""
#!/usr/bin/env python
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, force_temperature
from ase import units, Atoms
from ase.io import read, Trajectory, write
from ase.optimize import FIRE
from sys import exit, argv

#custom script in $HOME/Active_Learning/
from md_quality_assurance import check_system_integrity
#from ~/water_tools/
from analyze_system import deuterate, TIMESTEP_FACTOR, MOLAR_MASS, KG_TO_G, A3_TO_M3

#check if in work or work_broken env
try:
    from mace.calculators import mace_mp
except ImportError:
    from nequip.ase import NequIPCalculator


#define constants
FMAX = 0.2 #eV/A
STEPS = 1500
TIMESTEP = 1.0 #fs
TOTAL_TIME = 500 #ps
PS_TO_FS = 1E3
NVT_STEPS = TOTAL_TIME * PS_TO_FS // TIMESTEP
TEMPERATURE = 300 #kelvin
FRICTION = 0.002
LOGGIN_INTERVAL = 10
CHECK_FREQUENCY = 10 #ps


def pre_opt(
        atoms: Atoms, trajectory_file: str = None, logfile: str = None,
        fmax: float = FMAX, steps: int = STEPS
        ) -> Atoms:
    """
    Optimization before MD
    """
    atoms.calc = calc
    opt = FIRE(atoms, trajectory = trajectory_file, logfile = logfile)
    opt.run(fmax = fmax, steps = steps)

    return atoms



def water_nvt(
        atoms: Atoms, logfile: str, trajectory_name: str, calculator,
        friction: float = FRICTION, interval: int = LOGGIN_INTERVAL,
        timestep: float = TIMESTEP, temperature: float = TEMPERATURE,
        nvt_steps: int = NVT_STEPS, deuteration: bool = True, quality_checks: bool = False,
        check_frequency: float = CHECK_FREQUENCY * PS_TO_FS
        ):
    """
    Run an NVT using Langevin dynamics
    May check the MD every now and then to see if something has gone wrong
    """
    if deuteration:
        atoms = deuterate(atoms, TIMESTEP_FACTOR) ##increase the mass of hydrogen so we can use a larger timestep

    try:
        atoms.calc = calculator
    except Exception as e:
        print("Could not attach calculator: {e}")

    MaxwellBoltzmannDistribution(atoms, temperature_K = temperature)
    Stationary(atoms)
    traj = Trajectory(trajectory_name, "w", atoms = atoms)
    
    dyn = Langevin(
            atoms, timestep = timestep * units.fs, temperature_K = temperature,
            fixcm = True, friction = friction, logfile = logfile
            )
    dyn.attach(traj.write, interval = interval)

    steps_run = 0
    if quality_checks and check_frequency:
        steps_before_checking = check_frequency // timestep

        while steps_run < nvt_steps:
            dyn.run(steps_before_checking)
            steps_run += steps_before_checking

            #Quit the MD immediately if the MLIP shows instability
            if not check_system_integrity(atoms):
                print("NVT has been prematurely ended")
                exit(1)
            else:
                print("MD is probably alright")

    else:
        dyn.run(nvt_steps)

    traj.close()





