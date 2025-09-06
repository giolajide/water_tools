"""
Creates bulk water systems of specified Nwater molecules and density
"""
#!/usr/bin/env python
from ase.io import read, write
from sys import argv, exit
import os, subprocess
from os import makedirs, system
from ase import Atoms
import numpy as np
from ase.units import _Nav
from ase.build import molecule
from typing import List, Tuple, Union
from argparse import ArgumentParser

#from analyze_system import MOLAR_MASS, KG_TO_G, A3_TO_M3

MOLAR_MASS = np.sum(molecule("H2O").get_masses()) #18.01528 g/gmol
KG_TO_G = 1E3
A3_TO_M3 = 1e-30

#define constants and conversion factors
SEED = 123
TOLERANCE = 2.5 #Angstrom
DEFAULT_WATERS = 128

parser = ArgumentParser(
        description = "This script creates water cells of different densities, all having a specified number of molecules"
        )

parser.add_argument(
        "--densities", "-d", type = float, nargs = "+", required = True,
        help = "List of densities in kg/m3. Format: a b c d e. REQUIRED"
        )

parser.add_argument(
        "--tolerance", "-t", type = float, default = TOLERANCE,
        help = f"Min. spacing btwn any atom of (H2O)i and any atom of (H2O)j. Default = {TOLERANCE} A"
        )

parser.add_argument(
        "--waters", "-w", type = int, default = None,
        help = f"Number of waters in your unit cell. Default = {DEFAULT_WATERS} water molecules")

args = parser.parse_args()

WATERS = args.waters if args.waters else DEFAULT_WATERS

if not args.densities:
    print(f"Run as {argv[0].split('/')[-1]}  -d densities (Required): List[float] (kg/m3)  -t tolerance (Optional; default = {TOLERANCE} A): float -w number_of_waters (Optional; default = {DEFAULT_WATERS}): int.")
    exit(1)

write("h2o.xyz", molecule("H2O"))

requested_densities = args.densities #kg/m3
tolerance = args.tolerance
print(f"Note!!! Will run packmol with a tolerance of {tolerance} A")


def main():
    lengths = [do_packmol(density) for density in requested_densities]

    print("Requested density\tActual density (kg/m3)")
    for index, length in enumerate(lengths): #kg/m3
        crosscheck_density(requested_densities[index], length)

#    system("rm -f h2o.xyz") #deleting the file might cause a race condition if you run multiple in parallel



def calculate_cube_dimensions(particles: int, density: float) -> float:
    """
    Calculate the dimensions of the cube
    Given the number of molecules, and the required density
    """
    mass = MOLAR_MASS * (particles / _Nav) / KG_TO_G #kg
    volume = mass / density #m3
    volume /= A3_TO_M3 #A3
    length = np.power(volume, 1 / 3) #A

    return length


def do_packmol(density: float) -> Union[None, float]:
    """
    Create packmol input file and run it, given required density
    """
    lengths = calculate_cube_dimensions(WATERS, density) #A
    clearance = 2 #to allow for space between periodic images

    """Write packmol input file"""
    input_file = f"#System: Bulk H2O\n"
    input_file += f"#Density (kg/m3): {density:.3f}\n"
    input_file += f"tolerance {tolerance}\nseed {SEED}\nfiletype xyz\n"
    input_file += f"output {WATERS}_{density}.xyz\n\nstructure h2o.xyz\n"
    input_file += f"\tnumber {WATERS}\n\tinside box 0. 0. 0. {lengths - clearance} {lengths - clearance} {lengths - clearance}\n"
    input_file += "end structure\n\n"

    with open(f"{WATERS}_{density}.inp", "w") as file:
        file.write(input_file) 
#    print("Will now run packmol")
    """Run packmol"""
    command = f"packmol < {WATERS}_{density}.inp"

    try:
        with open(os.devnull, "w") as devnull, \
             open(f"{WATERS}_{density}.out", "w") as out_file, \
             open(f"{WATERS}_{density}.err", "w") as err_file:
            subprocess.run(command, shell = True, stdout = out_file, stderr = err_file)
#        print("Packmol run succeeded!")

    except subprocess.CalledProcessError:
        print(f"Packmol failed to execute correctly.")
        return None


        return None

    return lengths


def crosscheck_density(density: float, length: float, particles: int = WATERS):
    """
    Set cell, also check actual density
    """
    print(WATERS)
#    print("Will now set cell, and calculate density")
    atoms = read(f"{WATERS}_{density}.xyz")
#    length += 0.07
    atoms.set_cell([length, length, length], scale_atoms = False)
    atoms.set_pbc([True, True, True])
    atoms.center()
    print("Cell and PBC set")
    write(f"{WATERS}_{density}.xyz", atoms)
    mass = MOLAR_MASS * (particles / _Nav) / KG_TO_G #kg
    volume = atoms.get_volume() * A3_TO_M3 #A3

    print(f"{density:.3f}\t{mass / volume:.3f}") #kg/m3


if __name__ == "__main__":
    main()

