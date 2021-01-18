#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np

import yaml
import json
import matplotlib.pyplot as plt
import urllib.request
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import calc_dihedrals

from OrderParameter import *

from IPython import get_ipython

import copy

class DihedralFromAtoms(AnalysisBase):
        """Calculate dihedral angles for specified atomgroups.
        Dihedral angles will be calculated for each atomgroup that is given for
        each step in the trajectory. Each :class:`~MDAnalysis.core.groups.AtomGroup`
        must contain 4 atoms.
        Note
        ----
        This class takes a list as an input and is most useful for a large
        selection of atomgroups. If there is only one atomgroup of interest, then
        it must be given as a list of one atomgroup.
        """

        def __init__(self, atomgroups, orders, **kwargs):
                """Parameters
                ----------
                atomgroups : list
                        a list of atomgroups for which the dihedral angles are calculated
                Raises
                ------
                ValueError
                        If any atomgroups do not contain 4 atoms
                """
                super(DihedralFromAtoms, self).__init__(
                        atomgroups[0].universe.trajectory, **kwargs)
                self.atomgroups = atomgroups

                if any([len(ag) != 4 for ag in atomgroups]):
                        raise ValueError("All AtomGroups must contain 4 atoms")

                if len(atomgroups) != len(orders):
                    raise ValueError("Order data should be provided for every atom group")

                self.ag1 = mda.AtomGroup([atomgroups[i][orders[i][0]] for i in range(len(atomgroups))])
                self.ag2 = mda.AtomGroup([atomgroups[i][orders[i][1]] for i in range(len(atomgroups))])
                self.ag3 = mda.AtomGroup([atomgroups[i][orders[i][2]] for i in range(len(atomgroups))])
                self.ag4 = mda.AtomGroup([atomgroups[i][orders[i][3]] for i in range(len(atomgroups))])


        def _prepare(self):
                self.angles = []

        def _single_frame(self):
                angle = calc_dihedrals(self.ag1.positions, self.ag2.positions,
                                                             self.ag3.positions, self.ag4.positions,
                                                             box=self.ag1.dimensions)
                self.angles.append(angle)

        def _conclude(self):
                self.angles = np.rad2deg(np.array(self.angles))

# Download link
def download_link(doi, file):
    if "zenodo" in doi.lower():
        zenodo_entry_number = doi.split(".")[2]
        return 'https://zenodo.org/record/' + zenodo_entry_number + '/files/' + file
    else:
        print ("DOI provided: {0}".format(doi))
        print ("Repository not validated. Please upload the data for example to zenodo.org")
        return ""
    
# read mapping file
def read_mapping_file(mapping_file, atom1):
    with open(mapping_file, 'rt') as mapping_file:
            for line in mapping_file:
                if atom1 in line:
                    m_atom1 = line.split()[1]
    return m_atom1

def read_mapping_filePAIR(mapping_file, atom1, atom2, molname):
    with open(mapping_file, 'rt') as mapping_file:
            print(mapping_file)
            for line in mapping_file:
                if atom1 in line:
                    m_atom1 = line.split()[1]
                    try:
                        res = line.split()[2]
                    except:
                        res = molname
#                    print(m_atom1)
                if atom2 in line: 
                    m_atom2 = line.split()[1]
#                    print(m_atom2)
    return m_atom1, m_atom2, res

def make_positive_angles(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = x[i] + 360
        else:
            x[i] = x[i]
    return x


#Calculate the angle between PN vector and z-axis for each lipid residues


#  LIPIDS AND ATOMS IN MAPPING FILE CONVENTION GIVEN HERE
#
#lipids = {'POPS','POPE','POPG','POPC'}
lipids = {'POPG'}
atom1_nm = 'M_G3C4_M'
atom2_nm = 'M_G3C5_M'
atom3_nm = 'M_G3C5O1_M'
atom4_nm = 'M_G3C6_M'
#atom2 = 'M_G3N6_M'

colors = {'POPC' :'black','POPS':'red','POPE':'blue','POPG':'green'}

h = []

found = 0
for subdir, dirs, files in os.walk(r'../Data/Simulations/'):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith("README.yaml"):
            READMEfilepath = subdir + '/README.yaml'
            print(READMEfilepath)
            with open(READMEfilepath) as yaml_file:
                readme = yaml.load(yaml_file, Loader=yaml.FullLoader)
                for molname in lipids:
                    doi = readme.get('DOI')
                    trj = readme.get('TRJ')
                    tpr = readme.get('TPR')
                    trj_name = subdir + '/' + readme.get('TRJ')[0][0]
                    tpr_name = subdir + '/' + readme.get('TPR')[0][0]
                    gro_name = subdir + '/conf.gro'
                    trj_url = download_link(doi, trj[0][0])
                    tpr_url = download_link(doi, tpr[0][0])
                    EQtime=float(readme.get('TIMELEFTOUT'))*1000

                    #HGorientationFOLDERS = subdir.replace("Simulations","HGorientation")    
                    #outfilename = str(HGorientationFOLDERS) + '/' + molname + 'PNvectorDIST.dat' 

                    #print(molname, readme['NPOPC'][0], outfilename)                    
                    if int(readme['N' + molname][0]) > 0:
                                            
                        print('Analyzing '+molname+' in '+filepath)
                        
                        #Download tpr and xtc files to same directory where dictionary and data are located
                        if (not os.path.isfile(tpr_name)):
                            response = urllib.request.urlretrieve(tpr_url, tpr_name)
                        
                        if (not os.path.isfile(trj_name)):
                            response = urllib.request.urlretrieve(trj_url, trj_name)
                        
                        #fig= plt.figure(figsize=(12,9))
                        if (not os.path.isfile(gro_name)):
                            os.system('echo System | gmx trjconv -f {} -s {}  -dump 0 -o {}'.format(trj_name,tpr_name,gro_name))
                        
                        xtcwhole=subdir + '/whole.xtc'
                        if (not os.path.isfile(xtcwhole)):
                            os.system('echo System | gmx trjconv -f {} -s {} -o {} -pbc mol -b {}'.format(trj_name,tpr_name,xtcwhole,EQtime))
                        
                        try:
                            traj = mda.Universe(tpr_name,xtcwhole)
                        except FileNotFoundError or OSError:
                            continue

                        # with mda.Writer(subdir+"test.xtc", traj.select_atoms("all").n_atoms) as W:
                        #     for ts in traj.trajectory:
                        #         pos = traj.select_atoms("all").positions
                        #         traj.select_atoms("all").positions = np.array([pos[:,0],pos[:,1],-pos[:,2]]).T
                        #         W.write(traj)
                        #     #print(pos[:,2],traj.select_atoms("all").positions[:,2])
                    
                        mapping_file = './mapping_files/'+readme['MAPPING_DICT'][molname] # readme.get('MAPPING')[0][0]

                        try:
                            atom1 = read_mapping_file(mapping_file, atom1_nm)
                            atom2 = read_mapping_file(mapping_file, atom2_nm)
                            atom3 = read_mapping_file(mapping_file, atom3_nm)
                            atom4 = read_mapping_file(mapping_file, atom4_nm)
                            print(atom1,atom2,atom3,atom4)
                        except:
                            print("Some atom not found in the mapping file.")
                            continue

                        ags = []
                        orders = []
                        for residue in traj.select_atoms("name {} {} {} {} and resname POG".format(atom1,atom2,atom3,atom4)).residues:
                            print(residue.resid)
                            #print(residue.atoms.names)
                            atoms=traj.select_atoms("name {} {} {} {} and resid {}".format(atom1,atom2,atom3,atom4,str(residue.resid)))
                            if len(atoms) == 4:
                                print(residue.resname)
                                ags.append([])
                                ags[-1] = copy.deepcopy(atoms)
                                orders.append([
                                np.where(atoms.names==atom1)[0][0],
                                np.where(atoms.names==atom2)[0][0],
                                np.where(atoms.names==atom3)[0][0],
                                np.where(atoms.names==atom4)[0][0]])
                        #print(len(ags))
                        R = DihedralFromAtoms(ags,orders).run()
                        dihRESULT = R.angles.T
                        print(dihRESULT[0])
                        dihRESULT = [make_positive_angles(x) for x in dihRESULT ]
                        print(dihRESULT[0])
                        distSUM = np.zeros(360)
                        for i in dihRESULT:
                            distSUM += np.histogram(i,np.arange(0,361,1),density=True)[0]
                            print(distSUM)
                        #dihRESULT = [make_positive_angles(x) for x in dihRESULT ]
                        #dist = [ 0 for i in range(len(dihRESULT))]
                        #distSUM = [ 0 for i in range(360)]
                        #for i in range(len(dihRESULT)):
                        #   dist[i] =  plt.hist(dihRESULT[i], range(360),density=True);
                        #   distSUM = np.add(distSUM,dist[i][0])
                                            
                                    
                        distSUM = [x / len(dihRESULT) for x in distSUM]
                        xaxis = (np.histogram(i,np.arange(0,361,1),density=True)[1][1:]+
                            np.histogram(i,np.arange(0,361,1),density=True)[1][:-1])/2.
                        #xaxis = [ 0 for i in range(len(dist[0][1])-1)]
                        #for i in range(len(dist[0][1])-1):
                        #   xaxis[i]=(dist[0][1][i])
                                        
                        #plt.plot(xaxis,distSUM,color=colors[molname], label = readme.get('SYSTEM'))[0] 
                        #plt.legend()
                        #plt.xlabel("Angle (°)")
                        #plt.show()
                                
                        dihedralFOLDERS = subdir.replace("Simulations","dihedral")
                        os.system('mkdir -p {}'.format(dihedralFOLDERS))
                        os.system('cp {} {}'.format(READMEfilepath,dihedralFOLDERS))
                        outfile=open(str(dihedralFOLDERS) + '/' + 'isomer_' + molname + "_" + atom1_nm + "_" + atom2_nm + "_" + atom3_nm + "_" + atom4_nm +'.dat','w')
                        for i in range(len(xaxis)):
                            outfile.write(str(xaxis[i]) + " " + str(distSUM[i])+'\n')
                        outfile.close()

                        # Now you have trajectory in xtcwhole, tpr file in tpr_name, gro file in gro_name,
                        # two atom in atoms (you need more for isomer calculation so this has to be updated).
                        # Now you should be able to call your isomer checking code here within the loop that goes through the trajectories in the databank.
                        # Below is the example call for PN-vector calculation inside comments.
                        #
                        #
                        # EXAMPLE:
                        #try:
                        #    anglesMeanError = read_trj_PN_angles(atoms[2], [atoms[0],atoms[1]] , tpr_name, xtcwhole, gro_name)
                        #except OSError:
                        #    print("Could not calculate angles for " + molname + " from files " + tpr_name + " and " + trj_name)
                        #    continue
    
                        






