"""
This script will give us values of geometric descriptors for our system
"""
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry.analysis import Analysis
from sys import exit, argv, stdout
from argparse import ArgumentParser
from typing import List, Union, Tuple
from ase import Atoms
from numba import prange, jit, njit
from os.path import basename, splitext
from ascii_colors import ASCIIColors
#from utilities.print_bonds import (setup_neighborlist, setup_analyzer, get_bond_lengths)
from utilities.check_sanity import check_water, StructureError

BINS = 50


def get_water_distances(
        first_element: str, second_element: str, analyzer: Analysis,
        ) -> np.ndarray:
    """
    Gets distances from one atom type to another atom type
    
    Example Usage:
    from plot_OH_and_HOH import get_water_distances

    analyzer = Analysis(atoms)
    H_O_distances = get_water_distances(atoms, 'H', 'O', analyzer)
    """
    return np.array(analyzer.get_values(
        analyzer.get_bonds(first_element, second_element, unique = True)
        )[0])


def get_water_angles(
        first_element: str, reference_element: str, second_element: str, analyzer: Analysis
        ) -> np.ndarray:
    """
    Gets angles centered at 'reference_element'
    
    Example Usage:
    from plot_OH_and_HOH import get_water_angles

    analyzer = Analysis(atoms)
    H_O_H_angles = get_water_angles(atoms, 'H', 'O', 'H', analyzer)
    ↑↑The angle is centred at O
    """
    return np.array(analyzer.get_values(
        analyzer.get_angles(first_element, reference_element, second_element)
        )[0])


def prepare_special_function(atoms: Atoms):
    """
    Function to prepare inputs for the calculate_special_distances()
    function so that it is compilable with Numba
    """
    distance_matrix = atoms.get_all_distances(mic=True)
    O_mask = [index for index, i in enumerate(atoms) if i.symbol == "O"]
    H_mask = [index for index, i in enumerate(atoms) if i.symbol == "H"]
    oxygens, hydrogens = atoms[O_mask], atoms[H_mask]
    oxygens_distance_mat = oxygens.get_all_distances(mic=True)
    hydrogens_distance_mat = hydrogens.get_all_distances(mic=True)

    return distance_matrix, oxygens_distance_mat, hydrogens_distance_mat,\
            np.array(O_mask).astype(np.int64), np.array(H_mask).astype(np.int64)


##also calculate O-O and H-H and O-H (H-bond) distances
@jit(parallel = True, nopython = True)
def calculate_special_distances(
        distance_matrix: np.ndarray,
        oxygens_distance_mat: np.ndarray,
        hydrogens_distance_mat: np.ndarray,
        O_mask: np.ndarray,
        H_mask: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    calculate O-nearest_O, H-nearest_H, and O-H (H-bond) distances
    """
    intramolecular_OH = 1.15 #Ang

    len_o, len_h = oxygens_distance_mat.shape[0], hydrogens_distance_mat.shape[0]
    O_nearestO, H_nearestH, O_H_Hbond = np.ones(len_o), np.ones(len_h), np.ones(len_o)

    ###get O-nearestO and H-nearestH
    for i in prange(len_o):
        distances = np.sort(oxygens_distance_mat[i])
        O_nearestO[i] = np.min(distances[1:]) #exclude the first element, which will be 0. (i.e. self-distance)
    for i in prange(len_h):
        distances = np.sort(hydrogens_distance_mat[i])
        H_nearestH[i] = np.min(distances[1:]) #exclude the first element

    ## get the O-H (Hbond)
    for o_index in prange(O_mask.size):        # parallel over oxygens
        o_global = O_mask[o_index]
        distances = distance_matrix[o_global]

        min_val = np.inf                     # running minimum instead of a Python list
        for h_i in range(H_mask.size):
            h_global = H_mask[h_i]
            d = distances[h_global]
            if d > intramolecular_OH and d < min_val:
                min_val = d

        O_H_Hbond[o_index] = min_val if np.isfinite(min_val) else np.nan

    return O_nearestO, H_nearestH, O_H_Hbond


def main(trajectory: List[Atoms]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns all H-O distances and H-O-H angles present in your trajectory

    Also returns O-nearestO, H-nearestH, and O-H (Hbond)

    First checks to be sure that there are twice as many H and there are O
    If the above check fails, we can be sure we do not have water but something else!
    """
    if not check_water(trajectory):
        raise StructureError("This can't be water! Fuck off!!")

    analyzer_list = [Analysis(atoms, skin = 0.12, bothways = True, self_interaction = False) for atoms in trajectory]

    H_O_distances = np.concatenate([get_water_distances("O", "H", analyzer_list[index]) for index, _ in enumerate(trajectory)])
    H_O_H_angles = np.concatenate([get_water_angles("H", "O", "H", analyzer_list[index]) for index, _ in enumerate(trajectory)])

    print("Done with H-O and H-O-H")
    print("Now starting O_nearestO, H_nearestH, and H-O (Hbond)")
    O_nearestO, H_nearestH, O_H_Hbond = [], [], []

    for atoms in trajectory:
        intermediate_results = prepare_special_function(atoms)
        o_nearesto, h_nearesth, o_h_hbond = calculate_special_distances(*intermediate_results)
        O_nearestO.extend(list(o_nearesto))
        H_nearestH.extend(list(h_nearesth))
        O_H_Hbond.extend(list(o_h_hbond))

    return H_O_distances, H_O_H_angles, np.array(O_nearestO), np.array(H_nearestH), np.array(O_H_Hbond)


if __name__ == "__main__":
    parser = ArgumentParser(description = "Displays O-H lengths and H-O-H angles for your trajectory")
    parser.add_argument("--trajectory", "-t", type = str, help = "Trajectory file. REQUIRED")
    parser.add_argument("--bins", "-b", type = int, default = BINS,
            help = f"Number of bins for histogram. Optional. Default: {BINS}")
    parser.add_argument("--interval", "-i", type = int, default = 10,
            help = f"Read frames every x interval. Optional. Default: 10")
    parser.add_argument("--show", action = "store_true", help = "Display plots. Optional.")

    args = parser.parse_args()
    if not args.trajectory:
        print(f"Run as {basename(argv[0])} -t <traj_file> (Required) -b <n_bins> (Optional)")
        exit(1)

    trajectory = read(args.trajectory, f"::{args.interval}")
    bins = int(args.bins)
    dpi = 100
    alpha = 0.8
    H_O_distances, H_O_H_angles, O_nearestO, H_nearestH, O_H_Hbond = main(trajectory)

    ASCIIColors.print(
        "\n\n\n    STATISTICS BLOCK",
        color=ASCIIColors.color_yellow,
        style=ASCIIColors.style_underline,
        background=ASCIIColors.color_black,
        end="\n", flush=True, file=stdout
    )
    print(f"""H-O distances:\n \tMax:\t{np.max(H_O_distances):.3f}
        Min:\t{np.min(H_O_distances):.3f}\n Mean:\t{np.mean(H_O_distances):.3f}\n \tStdev:\t{np.std(H_O_distances):.3f}\n\n""")
    print(f"""Nearest H-H distances:\n \tMax:\t{np.max(H_nearestH):.3f}
        Min:\t{np.min(H_nearestH):.3f}\n Mean:\t{np.mean(H_nearestH):.3f}\n \tStdev:\t{np.std(H_nearestH):.3f}\n\n""")
    print(f"""Nearest O-O distances:\n \tMax:\t{np.max(O_nearestO):.3f}
        Min:\t{np.min(O_nearestO):.3f}\n Mean:\t{np.mean(O_nearestO):.3f}\n \tStdev:\t{np.std(O_nearestO):.3f}\n\n""")
    print(f"""O-H (H-bond) distances:\n \tMax:\t{np.max(O_H_Hbond):.3f}
        Min:\t{np.min(O_H_Hbond):.3f}\n Mean:\t{np.mean(O_H_Hbond):.3f}\n \tStdev:\t{np.std(O_H_Hbond):.3f}\n\n""")

    print(f"""H-O-H angles:\n \tMax:\t{np.max(H_O_H_angles):.1f}
        Min:\t{np.min(H_O_H_angles):.1f}\n Mean:\t{np.mean(H_O_H_angles):.1f}\n \tStdev:\t{np.std(H_O_H_angles):.1f}""")



    plt.hist(H_O_distances, bins = bins, density = True, alpha = alpha, color = "blue")
    plt.xlabel("H-O Bond Distances (Å)")
    plt.ylabel("Frequency")
    plt.title("H-O Intramolecular Bond Length Distribution")
    plt.grid(True)

    output_filename = splitext(basename(args.trajectory))[0] + "_HO_lengths.png"
    plt.savefig(output_filename, dpi = dpi)
    print(f"H-O plot saved in {output_filename}")
    if args.show:
        plt.show()


    plt.close()

    plt.hist(H_O_H_angles, bins = bins, density = True, alpha = alpha, color = "red")
    plt.xlabel("H-O-H Bond Angles (°)")
    plt.ylabel("Frequency")
    plt.title("H-O-H Intramolecular Bond Angles Distribution")
    plt.grid(True)

    output_filename = splitext(basename(args.trajectory))[0] + "_HOH_angles.png"
    plt.savefig(output_filename, dpi = dpi)
    print(f"H-O-H plot saved in {output_filename}")
    if args.show:
        plt.show()


    plt.close()

    plt.hist(O_H_Hbond, bins = bins, density = True, alpha = alpha, color = "red")
    plt.xlabel("H-O distances (A)")
    plt.ylabel("Frequency")
    plt.title("H-O INTERmolecular Bond Length Distribution")
    plt.grid(True)

    output_filename = splitext(basename(args.trajectory))[0] + "_HO_Hbond_lengths.png"
    plt.savefig(output_filename, dpi = dpi)
    print(f"H-O (Hbond) plot saved in {output_filename}")
    if args.show:
        plt.show()


    plt.close()

    plt.hist(H_nearestH, bins = bins, density = True, alpha = alpha, color = "purple")
    plt.xlabel("Nearest H-H distances (A)")
    plt.ylabel("Frequency")
    plt.title("Nearest H-H Distances Distribution")
    plt.grid(True)

    output_filename = splitext(basename(args.trajectory))[0] + "_H_H_distances.png"
    plt.savefig(output_filename, dpi = dpi)
    print(f"Nearest H-H distances plot saved in {output_filename}")
    if args.show:
        plt.show()


    plt.close()

    plt.hist(O_nearestO, bins = bins, density = True, alpha = alpha, color = "green")
    plt.xlabel("Nearest O-O distances (A)")
    plt.ylabel("Frequency")
    plt.title("Nearest O-O Distances Distribution")
    plt.grid(True)

    output_filename = splitext(basename(args.trajectory))[0] + "_O_O_distances.png"
    plt.savefig(output_filename, dpi = dpi)
    print(f"Nearest O-O distances plot saved in {output_filename}")
    if args.show:
        plt.show()


