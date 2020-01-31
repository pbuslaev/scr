import MDAnalysis as mda
import numpy as np
import math
import os, sys

def load_traj(traj_file, top_file):
	mol = mda.Universe(top_file,traj_file)
	print(mol)
	return mol

def calc_dihed(r):
	return mda.lib.distances.calc_dihedrals(r[0],r[1],r[2],r[3])

def parseDihedFile(file,traj):
	nl = []
	with open(file) as f:
		for line in f:
			names=line.split()
			nl.append(names)
			# for each line from the input file select atoms and 
			# calculate dihed for the trajectory
		
	diheds = np.array([])
	i=0
	k=0

	hists = np.zeros((len(nl),360))

	for ts in traj.trajectory:
		if k==0:
			print(i)
			i+=1
			k+=1
		elif k<50:
			k+=1
		elif k==50:
			k=0
		
		namep = 0
		for names in nl:
		#atoms = traj.select_atoms("name " + line)
			a1 = traj.select_atoms("name "+names[0])
			a2 = traj.select_atoms("name "+names[1])
			a3 = traj.select_atoms("name "+names[2])
			a4 = traj.select_atoms("name "+names[3])
				
			p1 = a1.positions
			p2 = a2.positions
			p3 = a3.positions
			p4 = a4.positions
				
				#po = np.array(np.split(atoms.positions,atoms.positions.shape[0]//4))
				#po1=np.swapaxes(po,0,1)
			dihed = (180/math.pi) * np.concatenate(mda.lib.distances.calc_dihedrals(p1,p2,p3,p4),axis=None)
			
			hists[namep] += np.histogram(dihed,np.arange(-180,181,1))[0]
			namep += 1
			
	np.savetxt("diheds.txt",hists)

def parseDihedDynamicsFile(file,traj):
	nl = []
	ll = []
	with open(file) as f:
		for line in f:
			names=line.split()[0:4]
			limits=list(int(x) for x in line.split()[4:])
			nl.append(names)
			ll.append(limits)
			# for each line from the input file select atoms and 
			# calculate dihed for the trajectory
	print(nl,ll)

	k = 0
	ddihp = []
	for ts in traj.trajectory:
		print()
		if k<=1:
			k+=1
		else:
			return 0

		
		namep = 0
		
		for (names,limits) in zip(nl,ll):
		#atoms = traj.select_atoms("name " + line)
			a1 = traj.select_atoms("name "+names[0])
			a2 = traj.select_atoms("name "+names[1])
			a3 = traj.select_atoms("name "+names[2])
			a4 = traj.select_atoms("name "+names[3])
				
			p1 = a1.positions
			p2 = a2.positions
			p3 = a3.positions
			p4 = a4.positions
				
				#po = np.array(np.split(atoms.positions,atoms.positions.shape[0]//4))
				#po1=np.swapaxes(po,0,1)
			dihed = (180/math.pi) * np.concatenate(mda.lib.distances.calc_dihedrals(p1,p2,p3,p4),axis=None)
			ddih = np.digitize(dihed,limits)
			ddih[ddih==len(limits)] = 0
			if len(ddihp) == 0:
				ddihp = ddih
			else:
				print(list(zip(ddih,ddihp)))

	



class Option:
    def __init__(self,func=str,num=1,default=None,description=""):
        self.func        = func
        self.num         = num
        self.value       = default
        self.description = description
    def __nonzero__(self): 
        return self.value != None
    def __str__(self):
        return self.value and str(self.value) or ""
    def setvalue(self,v):
        if len(v) == 1:
            self.value = self.func(v[0])
        elif isinstance(v,str):
        	self.value = self.func(v)
        else:
        	self.value = [ self.func(i) for i in v ]

def main(args):

	options = [
	# options for concat feature
	"Input/output options",
	("-f", Option(str, 1, None, "Input trajectory file (.xtc, .trr, ...)")),
	("-t", Option(str, 1, None, "Input topology file (.pdb, .gro, ...)")),
	("-l", Option(str, 1, None, "Lipid type = resname")),
	("-di", Option(str, 1, None, "List of dihedrals")),
	("-dd", Option(str, 1, None, "List of dihedrals for dynamics analysis"))
	]

	# if the user asks for help: pcalipids.py concat -h
	if (len(args)>0 and (args[0] == '-h' or args[0] == '--help')) or len(args)==0:
		print("\n",__file__[__file__.rfind('/')+1:])
		for thing in options: # print all options for selected feature
			print(type(thing) != str and "%10s: %s"%(thing[0],thing[1].description) or thing)
		print()
		sys.exit()

	options = dict([i for i in options if not type(i) == str])

	print(args)
	while args:
		ar = args.pop(0) # choose argument
		if options[ar].num == -1:
			listOfInputs = ""
			while args:
				ar1 = args.pop(0)
				if ar1 in list(options.keys()):
					options[ar].setvalue(listOfInputs)
					args.insert(0,ar1)
					break
				else:
					listOfInputs += (ar1+" ")
			options[ar].setvalue(listOfInputs)
		else:
			options[ar].setvalue([args.pop(0) for i in range(options[ar].num)]) # set value

	if not options["-f"].value or not options["-t"].value or not options["-l"].value:
		print("Trajectory, structure, and the lipid resname have to be provided")
		return
	
	traj = load_traj(options["-f"].value,\
					 options["-t"].value)

	if options["-di"].value:
		parseDihedFile(options["-di"].value, traj)

	if options["-dd"].value:
		parseDihedDynamicsFile(options["-dd"].value, traj)

	
if __name__ == '__main__':
	args = sys.argv[1:]
	main(args)
