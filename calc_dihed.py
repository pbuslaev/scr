import MDAnalysis as mda
import numpy as np
import math
import os, sys

top="/Users/pbuslaev/work/sim/2000_fo_aa/fo_oriented.pdb"
traj="/Users/pbuslaev/work/sim/2000_fo_aa/fo_oriented.pdb"

mol = mda.Universe(top,traj)
sel1=mol.select_atoms("resid 100 and segid A and name N")
sel2=mol.select_atoms("resid 100 and segid A and name CA")
sel3=mol.select_atoms("resid 100 and segid A and name C")
sel4=mol.select_atoms("resid 101 and segid A and name N")
r1=sel1.positions
r2=sel2.positions
r3=sel3.positions
r4=sel4.positions
print(mda.lib.distances.calc_dihedrals(r1,r2,r3,r4))