# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:08:42 2017

@author: nenian
"""

''' Import modules '''

import numpy as np
from pymatgen import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import minimize
import itertools, time
import argparse

def isanion(atom, anions):
    #print "in isanion fun... atom is {} and anions are {}".format(atom, anions)
    
    return atom in anions
    
def iscation(atom, cations):


    return atom in cations
    
''' GII functional computes the global instabilty index for each iteration. 
Args provided enable the code to call pymatgen or other modules that are needed
to assess the GII value. '''    
def gii_fun(x, *args):

    cations = args[0]
    anions = args[1]
    lattice = args[2]
    Species_list = args[3]
    num_atoms = args[4]
    cutoff_distance = args[5]
    BVpara = args[6]
    Formal_Valence = args[7]
    wycks = args[8]
    space_group = args[9]
    b0 = args[10]    
    center_atom_in_regular_poly = args[11]
    max_angle = args[12]
    nearest_neighbor_distance = args[13]
    struct = Structure(lattice, Species_list, x.reshape(int(num_atoms),3))
    pymat_neighbors = struct.get_all_neighbors(cutoff_distance, include_index=True)
    
    values_BV = []
    
    for atom_indx, neigh_data in enumerate(pymat_neighbors):
        bv = 0
        for pair_data in neigh_data:
            
            atom = struct.species[atom_indx].symbol 
            neighbor = struct.species[pair_data[2]].symbol
            if iscation(atom, cations) and isanion(neighbor, anions):
                bv += np.exp((BVpara[atom][neighbor]- pair_data[1])/b0)
            elif iscation(neighbor, cations) and isanion(atom,anions):
                
                bv += np.exp((BVpara[neighbor][atom]- pair_data[1])/b0)
        values_BV.append((Formal_Valence[struct.species[atom_indx].symbol] - bv)**2)

    GII_val = np.sqrt((sum(values_BV[:]))/struct.composition.num_atoms)
    #print "curent GII = {}".format(GII_val) 
    return GII_val


''' Here we compute the Bond angle variance (See VESTA manual page 74), 
i.e. the deviation of the internal angles of a regular polygon. Of course the 
ideal bond angle changes depending on the polygon. Below, the current setting 
is for an octahedral unit, thus bonds are either 180 or 90 degrees. 
Improvements are needed to account for other polygons'''
def angle_fun(x, *args):
    
    cations = args[0]
    anions = args[1]
    lattice = args[2]
    Species_list = args[3]
    num_atoms = args[4]
    cutoff_distance = args[5]
    BVpara = args[6]
    Formal_Valence = args[7]
    wycks = args[8]
    space_group = args[9]
    b0 = args[10]    
    center_atom_in_regular_poly = args[11]
    max_angle = args[12]  
    nearest_neighbor_distance = args[13]
    
    E_angle = 0

    pymat_structure = Structure(lattice, Species_list, x.reshape(int(num_atoms),3))

    for i, ion in enumerate(Species_list):
        
        if str(ion) == center_atom_in_regular_poly:

            NN = pymat_structure.get_neighbors(pymat_structure.sites[i], nearest_neighbor_distance, include_index=True)
            
            for x in itertools.combinations(NN,2):
                
                '''I compute the angles manually because I found inconsitancies
                in the pymatgen angle calculator. This may not be the case 
                in newer versions. Please test.'''
                ba = x[0][0].coords - pymat_structure.cart_coords[i]
                bc = x[1][0].coords - pymat_structure.cart_coords[i]
                
                cosine_angle = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                degr = np.rad2deg(angle)
                if np.isnan(angle) or 0 <= degr <= 20 or 150 <= degr <= 180:
                    #print "true"
                    pass
                else:
                    #print "angle = {}".format(np.rad2deg(angle))
                    diff = np.array([abs(np.pi - angle), abs(np.pi/2 - angle)])
                    
                    ''' Be mindful that this sum E_angle grows proportionally with 
                    the number of ions in the cell. Therefore, if the number of ions is large
                    E_angle may dominate the minimizer. Benchmarking and Scaling is needed... BY UNDERGRAD'''
                    E_angle += abs(np.deg2rad(max_angle)- diff.min())
                    
    #print "This is the angle sum {}".format(E_angle)

    return (E_angle)

'''If you don't want the space group to change during relaxation 
use the functional below.
Simply make the penalty for any space group change a large number. (STEP FUNCTION if you like) '''

def chksym(x,*args):
    cations = args[0]
    anions = args[1]
    lattice = args[2]
    Species_list = args[3]
    num_atoms = args[4]
    cutoff_distance = args[5]
    BVpara = args[6]
    Formal_Valence = args[7]
    wycks = args[8]
    space_group = args[9]
    b0 = args[10]    
    center_atom_in_regular_poly = args[11]
    max_angle = args[12]  
    nearest_neighbor_distance = args[13]    

    pymat_structure = Structure(lattice, Species_list, x.reshape(int(num_atoms),3))
    
    current_SG = SpacegroupAnalyzer(pymat_structure, symprec=0.01).get_space_group_number()
    #print current_SG
    if int(current_SG) is int(space_group):
        val = 0
    else:
        val = abs(float(current_SG) - float(space_group))/1
    #print val
    return val

''' In some cases anions can get too close to each other. anion_inter will
provide a penality that grows according to the logistic function (https://en.wikipedia.org/wiki/Logistic_function).
Again proper scaling and benchmarking are needed. '''
def anion_inter(x, *args):
    cations = args[0]
    anions = args[1]
    lattice = args[2]
    Species_list = args[3]
    num_atoms = args[4]
    cutoff_distance = args[5]
    BVpara = args[6]
    Formal_Valence = args[7]
    wycks = args[8]
    space_group = args[9]
    b0 = args[10]    
    center_atom_in_regular_poly = args[11]
    max_angle = args[12]  
    nearest_neighbor_distance = args[13]    
    Shannon_anion_anion = args[14]
    
    pymat_structure = Structure(lattice, Species_list, x.reshape(int(num_atoms),3))
        
    anions_indx = pymat_structure.indices_from_symbol(anions[0])
    logic = 0
    for anion in anions_indx:
        
        NN = pymat_structure.get_neighbors(pymat_structure.sites[anion], 4, include_index=True)
        #print len(NN)
        for neigh in NN:
            if str(neigh[0].specie.symbol) in anions:
                #print neigh[1]
                logic += 0.25/(1+np.exp(50*(neigh[1]-Shannon_anion_anion)))
    #print logic            
    return logic/10


def minime(x, *args):
    return gii_fun(x, *args) + angle_fun(x, *args) + chksym(x,*args) + anion_inter(x, *args)


''' Record starting time of code for timing/benchmarking'''
start_time = time.time()


""" Get input data first """ 
'''
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="file with calculation variables",
                    required=True, metavar="FILE")

parser.add_argument("-p", "--parent", help="parent file in vasp POSCAR format",
                    required=True, metavar="FILE")  
                    
args = parser.parse_args()

invars = Conf(args.input)
'''
# file path
fpath = "./BTO_test_str.vasp"

'''Read structure from file POSCAR or cif'''
try:    
    structure = Structure.from_str(open(fpath).read(), fmt="poscar")
except:
     structure = Structure.from_str(open(fpath).read(), fmt="cif")  
finally:
    "ERROR check file format: Please use POSCAR or cif"
    
''' Part of input: Define what the cations and anions are'''
cations = ['Ba', 'Ti']
anions = ['O']
''' temp variables that are needed in calculation of the angular contribution.
Improvements and benchmarking needed to limit/remove user overhead''' 
center_atom_in_regular_poly = 'Ti' 
''' Inequality constraint, allowed angle deviation from regular polygon angles in degree. 
If ion site symmetry does not have degrees of freedom this max_angle should be zero for best results.
This ad hoc parameter might be removed by scaling '''
max_angle = 6
nearest_neighbor_distance = 2.3 #Use this cutoff distance to compute nearest neighbors for center atom 

''' Inputs '''
cutoff_distance = 6 # Cut off distance to determine neighbors

''' Bond valence params to be read from input file. 
Or read default values from table for cation-anion pairs.
Currently, we use a dictonary set up.'''

BVpara = {'K':{'O':2.113, 'F':1.992}, 'Nb':{'O': 1.911, 'F':1.87}, 'La':{'O':2.148}, 'Mn':{'O':1.732},
'Na':{'O':1.803, 'F':1.677},'Pb':{'O':2.112},'Ba':{'O':2.285}, 'Ca':{'O':1.967},\
 'Ti':{'O':1.815}, 'Er':{'O':1.979}, 'Ni':{'O':1.75}}

b0 = 0.37 #Most common b0 value. Code should be formatted to allow user to choose which they want

''' Formal valence of each ion 
Should be added to input file '''

Formal_Valence = {'K':1, 'Na':1, 'Nb':5, 'Pb':2, 'Ba':2, 'Ca':2, 'Ti':4, 'O':2, 'F':1, 'La':3, 'Mn':3, 'Er':3, 'Ni':3}

''' Define args for minimizer. '''

num_atoms = structure.composition.num_atoms # number of atoms in cell
lattice = structure.lattice.matrix # lattice vectors
print(lattice)
# Check and save space group number
space_group = SpacegroupAnalyzer(structure, symprec=0.01).get_space_group_number()
# wyckoff list used to sets bounds for space group symmetries
wycks = SpacegroupAnalyzer(structure, symprec=0.01).get_symmetry_dataset()['wyckoffs'] 
Species_list = structure.species #list of ion names. 

'''sum of Shannon ionic radii to account for hard spheres distances of anions'''
Shannon_anion_anion = 2.7

myargs = (cations, anions, lattice, Species_list, num_atoms, cutoff_distance, \
BVpara, Formal_Valence, wycks, space_group, b0, center_atom_in_regular_poly, max_angle,\
 nearest_neighbor_distance, Shannon_anion_anion)

'''Define site symmetries for each wyckoff position in cell.
Currently requires user input, that can be obtained from SITESYM on the Bilbao Cyst. Server.
Possible Python wrapper for this may exist to automate process in Compuational Crystallography toolbox 
https://cctbx.github.io'''
site_operations = {'a':[[0, 0, 0],[0, 0, 0],[0, 0, 1]],
                   'b':[[0, 0, 0],[0, 0, 0],[0, 0, 1]],
                   'c':[[0, 0, 0],[0, 0, 0],[0, 0, 1]]}

'''maximum allowed factional displacement for each coordinate 
Default set to 0.15 frac. unit, can ask user to change as an input''' 
max_fractional_displacement = 0.15 

''' Bounds are used to enforce symmetry allowed displacements. 
For an ion with no degrees of freedom, the upper and lower bounds are set to 
the fractional coord in x, y or z. 
If displacement is allowed then the ion is moved between upper and lower bounds 
set by "max_fractional_displacement" '''
bnds = tuple()
for i in range(len(structure.frac_coords)):
    identity = np.array([1,1,1])
    sym_point = np.dot(identity, site_operations[wycks[i]])
    for j in range(len(sym_point)):
        if sym_point[j]==0:
            bnds += (structure.frac_coords[i][j],structure.frac_coords[i][j]), 
        else:          
            bnds += ((-1*max_fractional_displacement)+structure.frac_coords[i][j],\
            max_fractional_displacement+structure.frac_coords[i][j]),

'''Print space goup number. This should be in the output file'''
print("The space group of the structure is {}".format(space_group))

'''Flatten fractional coordinates of intial (input) structure to vector. 
Vectors are the standard format that the minimizer can understand. '''
x0 = structure.frac_coords.flatten()

'''Now the actual mimimizer is "Sequential Least SQuares Programming optimization" (SLSQP) 
algorithm. This method allows us to simply use constraints and bounds. 
I tried other methods including Limited-memory Broyden-Fletcher-Goldfarb-Shanno 
algorithm but they tend to be unstable. More testing may be needed if other algorithms are to be used'''
res = minimize(minime, x0,args=myargs, method='SLSQP', bounds = bnds, options={'disp':True})

'''Outputs: These should be written to file'''
#output ion coordinates after minimization must be reshaped
#relaxed_coordinates = res.x.reshape(num_atoms,3)
'''output relaxed structure as POSCAR type file'''
#out_structure = Structure(lattice, Species_list, relaxed_coordinates)
#w = Poscar(out_structure)
#w.write_file("out_ErNiO3_ions-l-b.vasp")

print(time.time() - start_time)
