
"""
This script will give us several important properties of a liquid-phase system
February 2025
"""
#!/usr/bin/env python
from tqdm import tqdm
from ase import units, Atoms
from ase.io import read, write
from ase.geometry.analysis import Analysis
from sys import exit, argv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
import warnings
from scipy.stats import gaussian_kde
from ase.build import molecule
from joblib import Parallel, delayed
from pymbar import timeseries #mamba activate md_statistics
                              #now I have it also installed in my work env
import statistics
from scipy import stats
import functools


N_WATERS = 128
RMAX = 13
NBINS = 100
ELEMENT = None
INTERVAL = 10
HMASS = Atoms("H").get_masses()[0] #1.008 g/mol
TIMESTEP_FACTOR = 2

MOLAR_MASS = np.sum(molecule("H2O").get_masses()) #18.01528 g/gmol
KG_TO_G = 1E3
A3_TO_M3 = 1e-30

AMU_TO_KG = 1.66053907E-27
AMU_A3_TO_KG_M3 = AMU_TO_KG / A3_TO_M3

NPROCS = 8 #-1 #for parallel processing
TIMEOUT = 5 * 60 #5 minutes

def deprecated(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"""{function.__name__} is not the best.
            Use TRAVIS instead""",
            DeprecationWarning,
            stacklevel = 2
        )
        return function(*args, **kwargs)
    return wrapper


def undeuterate(atoms: Atoms) -> Atoms:
    """
    Undeuterate the system
    """
    for atom in atoms:
        if atom.symbol == 'H':
            atom.mass = HMASS

    return atoms


def deuterate(atoms: Atoms, timestep_factor: float = TIMESTEP_FACTOR) -> Atoms:
    """
    Deuterate the system,
    to enable timestep_factor x default_timestep timestep
    """
    masses = atoms.get_masses()
    masses = [
            mass if atoms[index].symbol != "H" else
            (mass * timestep_factor) for (index, mass) in enumerate(masses)
            ]
    atoms.set_masses(masses)

    return atoms


def calc_density(atoms: Atoms, molar_mass: float = MOLAR_MASS) -> float:
    """
    Calculates density, in g/cm3
    """
    atoms = undeuterate(atoms)
    amu_A3 = np.sum(atoms.get_masses()) / atoms.get_volume() #density in amu/A^3
    kg_m3 = amu_A3 * AMU_A3_TO_KG_M3 #density in kg/m3

    return kg_m3 / 1000 #g/cm3


def equilibration(data: np.ndarray, nskip: int = None) -> Tuple[float, float, float]:
    """
    Check if system is equilibrated, based on 'data' (see below)
    which can be densities (for an NPT) or total energies (for an NVT)
    Requires:
        data (np.ndarray)       array of data
        nskip (int)             analyze data every nskip images
    Returns:
        t0:                     index at which simulation equilibrates
        g:                      correlation, i.e. data[::g] would yield highly uncorrelated samples
        Neff_max:               how many uncorrelated datapoints you have, i.e. len(data) / g
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if not nskip:
        nskip = 1
    [t0, g, Neff_max] = timeseries.detectEquilibration(data, nskip = nskip)
    warnings.warn(f"""System is NOT necessarily equlibrated!
    Check visually too""", category = UserWarning)

    return t0, g, Neff_max


def subsample_uncorrelated(data: np.ndarray, nskip: int = None) -> List:
    """
    Subsamples uncorrelated data
    which can be densities (for an NPT) or total energies (for an NVT)
    Requires:
        data (np.ndarray)           array of data
        nskip (int)                 analyze data every nskip images
    Returns:
        indices:                Uncorrelated image indices
    """
    [t0, g, Neff_max] = equilibration(data, nskip)
    indices = timeseries.subsampleCorrelatedData(data[t0:], g = g)
    
    return indices


#No longer needed
def calculate_cube_length(density: float,
        N: int = N_WATERS) -> float:
    """
    Calculate the cube side length in Å for a system of
    N water molecules, given the density in g/mL.

    Parameters:
        density (float): Density in g/mL.
    Returns:
        float: Cube side length in Å.
    """
    mass = N * (MOLAR_MASS / units._Nav)
    volume_ml = mass / density #ml (same as cm^3)
    volume_angstrom3 = (volume_ml / A3_TO_M3) / 1e6 #A^3
    L = volume_angstrom3 ** (1/3) #A

    return L


def track_density(
        trajectory: List[Atoms], interval: int = INTERVAL,
        show: bool = False, splits: int = None, processors: int = NPROCS,
        timeout: Union[float, int] = TIMEOUT, ylim: Tuple = (0.9, 1.1)
        ) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Tracks the evolution of the density of the system
    Example usage:

    from analyze_system import track_density, subsample_uncorrelated
    from ase.io import read, write

    interval = 20
    atoms = read('traj.traj', f"::{interval}")
    densities,mean,med,skew,kurt = track_density(atoms,interval = interval, show = True, processors = 7)

    print(f"Mean: {mean}\nMedian: {med}") #\nSkew: {skew}\nKurtosis: {kurt}")
    """

    interval = max(interval, 1) #ensure interval > 0

    with Parallel(n_jobs = processors, timeout = timeout) as parallel:
        densities = np.array(parallel(delayed(calc_density)(atoms)
            for (index, atoms) in enumerate(tqdm(trajectory))
            if index % interval == 0))

    plt.close("all")
    ##plot raw densities in order
    plt.plot(range(len(densities)), densities)
    plt.xlabel("Image index")
    plt.ylabel("Density (g/cm3)")
    plt.title("Change of density over the course of the trajectory")
    plt.tight_layout()
    plt.savefig("densities.png")

    if show:
        plt.show()

    plt.close("all")

    ##plot KDE of densities
    kde = gaussian_kde(densities)
    x_vals = np.linspace(densities.min(), densities.max(), 100)
    y_vals = kde(x_vals)

    plt.plot(x_vals, y_vals)
    plt.xlabel("Density (g/cm3)")
    plt.ylabel("P(r)")
    plt.title("KDE of densities")
    plt.tight_layout()
    plt.savefig("kde_densities.png")

    if show:
        plt.show()

    plt.close("all")

    if splits:
        splits = min(len(densities), splits)
        ##plot a histogram of densities
        plt.hist(densities, bins = splits)
        plt.title("Histogram of densities")
        plt.savefig("hist_densities.png")
        if show:
            plt.show()

        plt.close()
        ##plot binned densities in order
        binned_densities = np.array_split(densities, splits)
        binned_means, binned_stdevs = zip(*[(np.mean(i),np.std(i)) for i in binned_densities])
        plt.scatter(range(len(binned_densities)), binned_means, label = "Mean density")
        plt.plot(range(len(binned_densities)), np.array(binned_means) + np.array(binned_stdevs), label = "+ (stdev)")
        plt.plot(range(len(binned_densities)), np.array(binned_means) - np.array(binned_stdevs), label = "- (stdev)")
        plt.xlabel("Bin")
        plt.ylabel("Density (g/cm3)")
        plt.title(f"Densities averaged over {splits} splits")
        plt.tight_layout()
        plt.legend()
        plt.ylim(ylim) 
        plt.savefig("binned_densities.png")
        if show:
            plt.show()
        plt.close("all")

    skew, kurtosis = stats.skew(densities), stats.kurtosis(densities)
    mean, median = np.mean(densities), np.median(densities)
    warnings.warn("""skewness > 0 means your data is skewed towards larger values.
    Should (normally) be -2 < s < 2""", category = UserWarning)
    warnings.warn("""kurtosis tells how 'gaussian' the data distribution is.
    < 0 means it's rather flat
    Should (normally) be -2 < s < 2""", category = UserWarning)

    return densities, mean, median, skew, kurtosis

@deprecated
def averaged_rdf(
        trajectory: List[Atoms], interval: int = INTERVAL, r_max: float = RMAX,
        n_bins: int = NBINS, show: bool = False, element: str = ELEMENT,
        processors: int = NPROCS, timeout: Union[float, int] = TIMEOUT, name: str = "RDF"
        ) -> np.ndarray:
    """
    Generates a mean RDF
    """

    interval = max(interval, 1) #ensure interval > 0
    
    with Parallel(n_jobs = processors, timeout = timeout) as parallel:
        rdfs = np.array(parallel(delayed(calculate_rdfs)(atoms, atoms,
            return_single_value = True, element = element,
            n_bins = n_bins, r_max = r_max) for (index, atoms)
            in enumerate(tqdm(trajectory)) if index % interval == 0))

    rdfs = np.mean(rdfs, axis = 0)
    distances = np.linspace(0, r_max / 2, n_bins)

    plt.close("all")
    plt.plot(distances, np.transpose(rdfs))
    plt.xlabel("Distance (A)")
    plt.ylabel("RDF")
    plt.title("RDFs from different frames of the trajectory")
    plt.savefig(f"{name}.png")

    if show:
        plt.show()

    plt.close()

    return rdfs


def wrights_factor(rdf_truth: np.ndarray, rdf_simulated: np.ndarray) -> float:
    """
    Calculates Wright's factor
    The higher the better the agreement
    """
    wf = 100 * np.sum(rdf_truth ** 2) / np.sum((rdf_simulated - rdf_truth) ** 2)
    return wf


def rdf_metrics(rdf_truth: np.ndarray, rdf_simulated: np.ndarray):
    """
    Calculates several RDF metrics:
        RMSE
        Cosine similarity
        Earth Mover Distance
        KL divergence
    """
    rmse = np.sqrt(np.mean((rdf_truth - rdf_simulated) ** 2))
    cos_similarity = np.dot(rdf_truth, rdf_simulated) / (np.linalg.norm(rdf_truth) * np.linalg.norm(rdf_simulated))
    pearson_corr_coeff = np.corrcoef(rdf_truth, rdf_simulated)[0, 1]

    ##for kl divergence
    epsilon = 1e-10
    rdf_truths = np.clip(rdf_truth, epsilon, None)
    rdf_simulateds = np.clip(rdf_simulated, epsilon, None)
    rdf_truths /= np.sum(rdf_truths)
    rdf_simulateds /= np.sum(rdf_simulateds)
    kl = np.sum(rdf_truths * np.log(rdf_truths / rdf_simulateds))

    ##for earth mover's distance
    # Normalize both to sum to 1 (turn into probability distributions)
    rdf_truth = rdf_truth / np.sum(rdf_truth)
    rdf_simulated = rdf_simulated / np.sum(rdf_simulated)

    # Compute cumulative distributions
    cdf_ref = np.cumsum(rdf_truth)
    cdf_sim = np.cumsum(rdf_simulated)

    # Compute L1 distance between the CDFs
    emd = np.sum(np.abs(cdf_ref - cdf_sim)) / len(rdf_truth)  # Normalize by number of bins
    print("RMSE,COsine-similarity,corr coeff, KL divdergence, earth movers distance")
    return rmse, cos_similarity, pearson_corr_coeff, kl, emd


def calculate_rdfs(
        ground_truth: Atoms, simulated: Atoms, r_max: float = RMAX,
        n_bins: int = NBINS, element: str = ELEMENT,
        show: bool = False, return_single_value: bool = False, name: str = "rdf"
        ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Plots RDFs of DFT vs of MLIP.
    You can plot for only some element (using the 'element' argument)
    Of course, if you want only one rdf, then just have ground_truth = simulated
    """
    permissible_rmax = min(
            (min(ground_truth.cell.cellpar()[:3]) / 2), (min(simulated.cell.cellpar()[:3]) / 2)
            ) #just in case the cell is not cubic

    if r_max > permissible_rmax:
        r_max = np.floor(permissible_rmax)
        warnings.warn(
                f"r_max changed to {r_max} A due to cell size being too small in at least one direction",
                category = UserWarning
                )

    geometry_analyzer_ground_truth = Analysis(ground_truth)
    geometry_analyzer_simulated = Analysis(simulated)

    rdf_values_ground_truth = np.array(
            geometry_analyzer_ground_truth.get_rdf(rmax = r_max,
                nbins = n_bins, elements = element)[0]
            )
    rdf_values_simulated = np.array(
            geometry_analyzer_simulated.get_rdf(rmax = r_max,
                nbins = n_bins, elements = element)[0]
            )

    if return_single_value:
        return rdf_values_ground_truth

    distances = np.linspace(0, r_max / 2, n_bins)

    plt.close("all")
    plt.figure()
    plt.plot(distances, rdf_values_ground_truth, label = "Ground Truth")
    plt.plot(distances, rdf_values_simulated, label = "Simulated", linestyle = "--")
    plt.xlabel("Distance (A)")
    plt.ylabel("RDF g(r)")
    plt.title("RDF comparison")
    plt.legend()
    plt.savefig(f"{name}.png")

    if show:
        plt.show()

    plt.close()

    return (rdf_values_ground_truth, rdf_values_simulated)


