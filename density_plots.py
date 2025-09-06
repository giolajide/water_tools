#!/usr/bin/env python
from analyze_system import track_density, subsample_uncorrelated
from sys import argv, exit
from ase.io import read, write, iread


if len(argv) != 7:
    print("Run as {argv[0]} traj interval splits output action processes")
    print("action should be 'write' if you want to write the equilibrated traj")
    exit()

interval = int(argv[2])
output = argv[4]
action = argv[5]
if action == "write":
    atoms = read(argv[1], f"::{interval}")
else:
    atoms = iread(argv[1], f"::{interval}")

densities,mean,med,skew,kurt = track_density(
        atoms,
        interval = interval,
        show = True,
        splits = int(argv[3]),
        processors = int(argv[6])
        )

print(f"Mean: {mean}\nMedian: {med}") #\nSkew: {skew}\nKurtosis: {kurt}")

if action == "write":
    indices = subsample_uncorrelated(densities)
    write(f"{output}/equilibrated_npt.traj", [atoms[i] for i in indices])


