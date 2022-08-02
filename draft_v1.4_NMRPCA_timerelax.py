#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import urllib.request
import sys
import os
import subprocess
#sys.path.insert(1, '/home/sashq/.local/lib/python3.6/site-packages/')
#from databankLibrary import download_link

import MDAnalysis as mda
import MDAnalysis.transformations
from MDAnalysis.analysis import align
import numpy as np
import math
from scipy import signal
import gc

################################# AMBER ################################################
def findHead(ind, nahead, headlist):
    
    headcoord=np.empty((nahead, 3))
    headcoord[0:nahead, :]=headlist.positions[ind*nahead:(ind+1)*nahead]
    
    return headcoord
           
def findTails(ind, natail1, natail2, tail1list, tail2list):
    
    tail1coord=np.empty((natail1, 3))
    tail2coord=np.empty((natail2, 3))
    
    tail1coord[0:natail1, :]=tail1list.positions[ind*natail1:(ind+1)*natail1]
    tail2coord[0:natail2, :]=tail2list.positions[ind*natail2:(ind+1)*natail2] 
    
    return tail1coord, tail2coord

def mergeHeadTails(nahead, headlist, natail1, tail1list, natail2, tail2list, ind):    
    
    headc=findHead(ind, nahead, headlist)
    tail1c,tail2c=findTails(ind, natail1, natail2, tail1list, tail2list)
    
    mergecoord=np.concatenate((tail1c, headc, tail2c))
    
    return mergecoord

def framesLipCoord(nhead, nahead, headlist, natail1, tail1list, natail2, tail2list):
    
    frameLipCoord=np.empty((0,3))
    for j in range(nhead):
        frameLipCoord=np.concatenate((frameLipCoord, mergeHeadTails(nahead, headlist, natail1, tail1list, natail2, tail2list, j)))
    
    return frameLipCoord
###############################################################################################
    
def load_traj(traj_file, top_file, lipid_resname, stride, sf, ef):
    
    if not traj_file or not top_file or not lipid_resname:
        print("Trajectory and topology files have to be provided.\n\
              Name of residue of interest have to be indicated.")
        return 0

#    traj_file = PATH + 'RUNC.xtc'
    topol = mda.Universe(top_file, traj_file)
    
#    ag = topol.atoms
#    
#    # unwrapping atoms
#    transform = mda.transformations.unwrap(ag)
#    topol.trajectory.add_transformations(transform)
    
    #traj = topol.trajectory[sf:ef:stride]
    traj = topol.trajectory[:]
    
#    topol.select_atoms('name POPC').write('1.pdb', frames=traj[:2])
    
    return topol, traj

def concat(traj, topol, FF, lipid_resname):
    
    nframes = len(traj)

##################################### AMBER ################################################################################################  
    if FF == 'Lipid17' or FF == 'Lipid14' or FF == 'lipid17':
        
        if lipid_resname == 'POPC':
            head, tail1, tail2 = 'PC','PA','OL'
            
        if lipid_resname == 'POPG':
            head, tail1, tail2 = 'PGR','PA','OL'
            
        if lipid_resname == 'POPS':
            head, tail1, tail2 = 'PS','PA','OL'
            
        if lipid_resname == 'POPE':
            head, tail1, tail2 = 'PE','PA','OL'
            
        headlist = topol.select_atoms('not name H* and resname '+'{}'.format(head))
        nhead = headlist.n_residues
        nahead = headlist.n_atoms // nhead

        tail1list = topol.select_atoms('(not name H* and resname '+'{}'.format(tail1)+') and around 2.0 (resname '+'{}'.format(head)+' and not name H*)')
        selResTail1 = "("
        for i, j in enumerate(tail1list.resids[:-1]):
            selResTail1 = selResTail1+'resid '+'{}'.format(j)+' or '
        selResTail1 = selResTail1+'resid '+'{}'.format(tail1list.resids[-1])+') and not name H*'

        tail2list = topol.select_atoms('(not name H* and resname '+'{}'.format(tail2)+') and around 2.0 (resname '+'{}'.format(head)+' and not name H*)')
        selResTail2 = "("
        for i, j in enumerate(tail2list.resids[:-1]):
            selResTail2 = selResTail2+'resid '+'{}'.format(j)+' or '
        selResTail2 = selResTail2+'resid '+'{}'.format(tail2list.resids[-1])+') and not name H*'

        tail1list = topol.select_atoms(selResTail1) 
        ntail1 = tail1list.n_residues
        natail1 = tail1list.n_atoms // ntail1

        tail2list = topol.select_atoms(selResTail2) 
        ntail2 = tail2list.n_residues
        natail2 = tail2list.n_atoms // ntail2
   
        nalip = nahead+natail1+natail2

        headnames = list(headlist.atoms[:nahead].names)
        tail1names = list(tail1list.atoms[:natail1].names)
        tail2names = list(tail2list.atoms[:natail2].names)
        lipnames = tail1names+headnames+tail2names
        atom_name = lipnames
        
        nlip = nhead
        trajxyz = np.empty((nframes*nlip*nalip, 3))
        for i, ts in enumerate(traj):
            trajxyz[i*nalip*nlip:(i+1)*nalip*nlip, :] = framesLipCoord(nlip, nahead, headlist, natail1, tail1list, natail2, tail2list)
########################################################################################################################################    
    
    else:
        
        ailist = topol.select_atoms('not resname SOL and not (name H* or type HW or type OW or type W \
                                                          or type H or type Hs or type WT4 or type NaW \
                                                          or type KW or type CLW or type MgW) and resname %s' \
        % lipid_resname) #selection of heavy lipid atoms
    
        nlip   = ailist.n_residues # amount of lipids
#    na      = ailist.n_atoms # amount of lipid atoms
        nalip    = ailist.n_atoms // nlip # amount of atoms per lipid
        
        lipnames = list(ailist.atoms[:nalip].names)
        atom_name = lipnames
        
    # positions
        trajxyz = np.empty((nframes*nlip*nalip, 3))
        for i, ts in enumerate(traj):
            trajxyz[i*nalip*nlip:(i+1)*nalip*nlip, :] = ailist.positions
    
    trajxyz = trajxyz.reshape((nframes,nlip,nalip,3))
    trajxyz = np.swapaxes(trajxyz,0,1)
    trajxyz = trajxyz.reshape((nframes*nlip,nalip,3))
    # new Universe (trajn) based on heavy atom positions
    atom_resindex = np.repeat(0, nalip)
    # [] are neaded
    res_name = [lipid_resname]
    trajn = mda.Universe.empty(nalip, 1, atom_resindex=atom_resindex, trajectory=True)
    trajn.add_TopologyAttr('name', atom_name)
    trajn.add_TopologyAttr('resname', res_name)
    trajn.load_new(trajxyz)
    # average structure
    av = align.AverageStructure(trajn, ref_frame=0).run()
    avg_str = av.results.universe 
    align.AlignTraj(trajn, avg_str).run() #aligning of the new trajectory along the average structure
    # positions from the average structure
    avg_pos = avg_str.select_atoms('all').positions.astype(np.float64).\
    reshape(1, avg_str.select_atoms('all').n_atoms * 3)
    # positions from trajn   
    nframesn = len(trajn.trajectory)
    nalip = trajn.select_atoms('all').n_atoms
    trajxyz = np.empty((nframesn*nalip, 3))
    for i, ts in enumerate(trajn.trajectory):
        trajxyz[i*nalip:(i+1)*nalip, :] = trajn.select_atoms('all').positions
    
    return trajxyz, avg_pos, nlip, nframes

def PCA(trajxyz, avg_pos):
    
    nalip = len(avg_pos[0,:]) // 3
    nframes = len(trajxyz) // nalip  
    # centering of positions relative to the origin
    X = trajxyz.astype(np.float64).reshape(nframes, nalip * 3) - avg_pos
    # the sum of all coordinates (to calculate mean)
    X_1 = X.sum(axis=0)
    # production of X and X^T
    X_X = np.tensordot(X, X, axes=(0,0))
    # covariance matrix calculation
    cov_mat = (X_X - np.dot(X_1.reshape(len(X_1),1), \
		(X_1.reshape(len(X_1),1)).T) / nframes) / (nframes - 1)
    # eigenvalues and eigenvectors calculation
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    eig_vecs = np.flip(eig_vecs, axis=1).T
    
    return X, eig_vecs

def get_proj(cdata, eig_vecs, first_PC = 1):
    # projection on PC1
    proj = np.tensordot(cdata, eig_vecs[first_PC - 1:1], axes = (1,1)).T
    proj=np.concatenate(np.array(proj), axis=None)
    
    return proj

def estimated_autocorrelation(x, variance, mean):
	# x - data
	n = len(x)
	# Centering the data
	x = x-mean
	# Convolve the data
	r = signal.fftconvolve(x, x[::-1], mode="full")[-n:]
	# Weight the correlation to get the result in range -1 to 1
	result = r/(variance*(np.arange(n, 0, -1)))
    
	return result

# Interpolation
def get_nearest_value(iterable, value):
    
	for idx, x in enumerate(iterable):
		if x < value:
			break

	A = idx
	if idx == 0:
		for idx, x in enumerate(iterable):
			if x < iterable[A]:
				break
		B = idx
	else:
		B = idx - 1
	# print(A,B)
	return A, B

def calc(proj, nlip, timestep):
	# mean,variance calculation
    variance  = proj.var()
    mean = proj.mean()
	# correlation for the first lipid
    data_1 = proj[:len(proj) // nlip]
    R = estimated_autocorrelation(data_1, variance, mean)
	# correlation for other lipids
    for i in range(1, nlip):
	    data_1 = proj[len(proj) * i // nlip:len(proj) * (i + 1) // nlip]
	    R += estimated_autocorrelation(data_1, variance, mean)
	# averaged correlation calculation
    R /= nlip
	# corresponding times
    T = []
    for i in range(0, len(R)):
	    T.append(timestep * i)
        
    return T, R
    
def timerelax(time, aucor):
    
    points = []
    j = 1
    while j < len(aucor):
        points.append(j)
        j = int(1.5 * j) + 1
    # decay in e^1
    T_pic = np.array([time[i] for i in points if aucor[i] > 0.])
    R_pic = np.array([aucor[i] for i in points if aucor[i] > 0.])
    R_log = np.log(R_pic[:])
    T_log = np.log(T_pic[:])
	# data interpolation
    power = -1
    A, B = get_nearest_value(R_log, power)
    a = (T_log[B] - T_log[A]) / (R_log[B] - R_log[A])
    b = T_log[A] - a * R_log[A]
    t_relax1 = a * power + b
    T_relax1 = math.e ** t_relax1
    
    return T_relax1

def esttime(file_1, file_2, FF, trj_len, lipid_resname, stride = 1, sf = 1, ef = -1):

#    PATH = os.getcwd() + '/'
    # topology, trajectory loading
    topol, traj = load_traj(traj_file = file_1, top_file = file_2, \
                     lipid_resname = lipid_resname, stride = stride, sf = sf, ef = ef)
    # positions of heavy lipid atoms
    trajxyz, avg_pos, nlip, nframes = concat(traj=traj, topol=topol, FF=FF, lipid_resname=lipid_resname)
    # PCA
    X, eig_vecs = PCA(trajxyz=trajxyz, avg_pos=avg_pos)
    # projection on PC1
    proj = get_proj(cdata=X, eig_vecs=eig_vecs)
    #timestep
    ts=trj_len/(nframes)
    # autocorrelation
    T, R = calc(proj=proj, nlip=nlip, timestep=ts)
    # relaxation time at e^1 decay
    te1 = timerelax(time=T, aucor=R)
    
    return te1 * 49 # 49 from sample data average of tkss2/tac1

def download_link(doi, file):
    if "zenodo" in doi.lower():
        zenodo_entry_number = doi.split(".")[2]
        return 'https://zenodo.org/record/' + zenodo_entry_number + '/files/' + file
    else:
        print ("DOI provided: {0}".format(doi))
        print ("Repository not validated. Please upload the data for example to zenodo.org")
        return ""
    
def findlipids(readme):
    # must be a full list of lipids
    lipids_list = ['POPC', 'POPE', 'POPG', 'POPS', 'DMPC', 'DPPC', 'DPPE', 'DPPG', 'DEPC', 'DLPC', 'DLIPC', 'DOPC', 'DDOPC', 'DOPS', 'DSPC', 'DAPC', 'POPI', 'SAPI', 'SLPI', 'CER', 'DCHOL', 'DHMDMAB', 'DPPG', 'CHOL', 'DPP'] #add the rest
    lipids = []
    # add to the lipids those lipids that are indicated in .yaml 
    for lipid in lipids_list:
        try:
            if readme['COMPOSITION'][lipid] != 0:
                lipids.append(lipid)
        except KeyError:
            continue
        
    return lipids

i=0
listall=[]
for subdir, dirs, files in os.walk (r'../Data/Simulations/'):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith("README.yaml"):
            READMEfilepath = subdir + '/README.yaml'
#            print(READMEfilepath)
            with open(READMEfilepath) as yaml_file:
                readme = yaml.load(yaml_file, Loader=yaml.FullLoader)
                #for molname in lipids:
                doi = readme.get('DOI')
                trj = readme.get('TRJ')
                tpr = readme.get('TPR')
                FF = readme.get('FF')
#                Temp = readme.get('TEMPERATURE')
                
                trjLen = readme.get('TRJLENGTH')/1000 # ns
                trj_name = subdir + '/' + readme.get('TRJ')[0][0]
                tpr_name = subdir + '/' + readme.get('TPR')[0][0]
                trj_url = download_link(doi, trj[0][0])
                tpr_url = download_link(doi, tpr[0][0])
                
                gro = readme.get('GRO')
                if gro!=None:
                    gro_name = subdir + '/' + readme.get('GRO')[0][0]
                    gro_url = download_link(doi, gro[0][0])
                    if (not os.path.isfile(gro_name)):
                        response = urllib.request.urlretrieve(gro_url, gro_name)
                    
                if not os.path.isfile(tpr_name):
                    response = urllib.request.urlretrieve(tpr_url, tpr_name)
                        
                if not os.path.isfile(trj_name):
                    response = urllib.request.urlretrieve(trj_url, trj_name)
                    
                EQtime=float(readme.get('TIMELEFTOUT'))*1000

                if os.path.isfile(trj_name): 
                    i+=1
                    listall.append({})
                    summ=0
                    
                    subprocess.run('echo System | gmx trjconv -s ' + tpr_name + ' -f ' + trj_name + ' -pbc mol -o ../Data/Simulations/whole.xtc', shell=True)
#                    subprocess.run('rm ' + trj_name, shell=True)
#                    PATH = os.getcwd() + '/'
                    trj_name = '../Data/Simulations/whole.xtc'
                    
                    for lipid in findlipids(readme):
#                        summ=summ+sum(readme['COMPOSITION'][lipid]['COUNT'])
#                        listall[i-1]['totalLipidNumber'] = summ
#                        listall[i-1]['trjLen'] = trjLen
#                        listall[i-1]['other'] = [FF]
#                        listall[i-1]['T'] = Temp
                               
                        if lipid=='CHOL' or lipid=='DCHOL':
                            continue
                        else:
                            if lipid=='DDOPC':
                                lipid='DDPC'
                            if lipid=='DHMDMAB':
                                lipid='T7H'
                            if lipid=='CER':
                                lipid='CER160'
                            try:
                                if (readme.get('FF')=='Berger' or readme.get('FF')=='Berger and Modified H\xF6ltje model for cholesterol') and readme['COMPOSITION']['POPC']!=0:
                                    lipid='PLA'
                            except KeyError:
                                print('next')
                            
                            if gro!=None:
                                te2 = esttime(trj_name, gro_name, FF, trjLen, lipid_resname=lipid)
                            else:
                                te2 = esttime(trj_name, tpr_name, FF, trjLen, lipid_resname=lipid)
                                
                        listall[i-1]['%s' % lipid] = te2/trjLen
                    
#                    PATH = os.getcwd() + '/'
                    subprocess.run('rm ' + trj_name, shell=True)
                    
                    gc.collect()
