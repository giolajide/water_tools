import numpy as np
from ase.io import read
from sys import exit
from os.path import splitext, basename
from os import system, makedirs, chdir, getcwd
from argparse import ArgumentParser


###TODO: script is still a bit disfunctional

template = """! TRAVIS input file
! Created with TRAVIS version compiled at Apr 18 2025 16:22:24
! Source code version: Jul 29 2022
! Input file written at Fri May  2 20:16:59 2025.
! Use the advanced mode until the analysis selection menu (y/n)? [no] 
y
! Use these values (y) or enter different values (n)? [yes] 

! Update cell geometry in every time step (i.e., NPT ensemble) (y) or use fixed cell (n)? [yes] 

! Should the periodic box be multiplied (y/n)? [no] 

! Execute molecule recognition for which time step (-1 = disable)? [0] 

! Use spatial domain decomposition for faster mol. rec. (disable if there are problems) (y/n)? [yes] 

! Use refined topological atom ordering (can take very long for large molecules) (y/n)? [yes] 

! Show bond matrices only for first representant of each molecule type (y/n)? [yes] 

! Show only bonds in the bond matrices (y/n)? [yes] 

! Create images of the structural formulas (y/n)? [no] 

! Accept these molecules (y) or change something (n)? [yes] 

! Change the atom ordering in some molecule (y/n)? [no] 

! Define additional virtual atoms (y/n)? [no] 

! Do you want to define pseudo-molecules (y/n)? [no] 

! Which functions to compute (comma separated)?
rdf
! Use the advanced mode for the main part (y/n)? [no] 
y
! Please enter the atom to put into the box center (e.g. C3): [center of mass]

! Perform this observation intramolecular (within the reference molecule) (0) or intermolecular (1)? [1] 
inter_or_intra
! Observe only certain molecules of H2O / H2O (y/n)? [no] 

! Decompose this observation into contributions from different elements (y/n)? [no] 

! Take reference atom(s) from RM H2O (0) or from OM H2O (1)? [0] 
reference_atom
! Take observed atom(s) from RM H2O (0) or from OM H2O (1)? [1] 
observed_atom
! Which atom(s) to take from RM H2O (e.g. "C1,C3-5,H", "*"=all)? [#2]
first_atom
! Which atom(s) to take from OM H2O (e.g. "C1,C3-5,H", "*"=all)? [#2]
second_atom
! Add another set of atoms to this (!) RDF (y/n)? [no] 

! Use values (0), 1st time derivative (1), or 2nd time derivative (2) of the values? [0] 

! Enter the minimal radius of this RDF (in pm): [0] 

! Enter the maximal radius of this RDF (in pm): [700.0] 
maximal_radius
! Switch on the long-range mode by Kuehne and Roehrig (y/n)? [yes] 

! Enter the resolution (bin count) for this RDF: [240] 

! Please enter histogram resolution (0=no histogram): [0] 

! Draw a line in the agr file at g(r) = 1 (y/n)? [no] 

! Correct radial distribution for this RDF (y/n)? [yes] 

! Compute occurrence in nm^(-3) (y) or rel. to uniform density (n)? [no] 

! Save temporal development for this observation (y/n)? [no] 

! Create a temporal difference plot for this observation (y/n)? [no] 

! Create a fraction temporal development plot (y/n)? 

! Add a condition to this observation (y/n)? [no] 

! Add another observation (y/n)? [no] 

! Perform a multi-interval analysis (y/n)? [no] 

! In which trajectory frame to start processing the trajectory? [1] 
start_index
! How many trajectory frames to read (from this position on)? [all] 
how_many_images
! Use every n-th read trajectory frame for the analysis: [1] 
"""

if __name__ == "__main__":

    parser = ArgumentParser(description = "This script runs TRAVIS for water RDFs")
    parser.add_argument("--file", "-f", type = str, default = "input.txt", help = "Name for TRAVIS input file")
    parser.add_argument("--input", "-i", type = str, required = True, help = "PDB file's full path. Required")
    parser.add_argument("--type", "-t", type = str, default = "Inter", choices = ["Inter", "Intra"],
            help = "Inter- or Intra- molecular RDF")
    parser.add_argument("--ref", "-rf", type = str, default = "O1", choices = ["H1", "H2", "O1"],
            help = "Reference atom")
    parser.add_argument("--second", "-s", type = str, default = "O1", choices = ["H1", "H2", "O1"],
            help = "Second atom")
    parser.add_argument("--rmax", "-r", type = float, default = None,
            help = """Max distance (in pm) for computing the RDF.
            Default will be gotten from cell vectors of the PDB file's first image""")
    parser.add_argument("--start_index", "-si", type = int, required = True,
            help = "In which trajectory frame to start processing the trajectory?")
    parser.add_argument("--len_images", "-l", type = int, default = None,
            help = """How many trajectory frames to read (from this start_index on)?
            Default = None (for all frames)""")

    args = parser.parse_args()

    inter_or_intra = 0 if args.type == "Intra" else 1
    rmax = args.rmax if args.rmax else np.min(read(args.input, 0).cell.lengths()) * 100 / 2
    print(f"Rmax:\t{rmax:.3f} pm")
    start_index = args.start_index
    how_many_images = args.len_images if args.len_images else ""
    observed_atom = "1" if inter_or_intra == 1 else "0"

    input_string = template.replace("inter_or_intra", str(inter_or_intra)).\
            replace("reference_atom", "0").replace("observed_atom", observed_atom).\
            replace("first_atom", args.ref).replace("second_atom", args.second).\
            replace("maximal_radius", str(rmax)).\
            replace("start_index", str(start_index)).\
            replace("how_many_images", str(how_many_images))

    pdb_file = args.input #basename(args.input)
    directory = splitext(args.file)[0]
    makedirs(directory, exist_ok = True)

    with open(f"{directory}/{args.file}", "w") as input_file:
        input_file.write(input_string)
    print(f"Written input file {directory}/{args.file}")
    
    chdir(directory)
    print(f"Entered {getcwd}")
    system(f"travis -p {pdb_file} -i {args.file}")
    print("Finished running TRAVIS")
    chdir("../")

