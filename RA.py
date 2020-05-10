############################
#### Ring Analysis (RA) ####
############################

import math
import numpy as np
import pandas as pd
from rdkit import Chem
import py_rdl

################################
########## Fix Zero ############
################################

def fixzero(x):
    x_ = np.array([0.0]) if np.allclose(0,x, rtol=1e-06, atol=1e-08) else x 
    return x_ 

################################
#### Cremer Pople Parameter ####
################################
def Translate(coordinates):
    """
    Translate the ring coordinates to origin center
    
    Input: 

    coordinates: array

    Return:

    new_coordinates: array
    """
    new_coordinate = coordinates - coordinates.mean(axis=0)
    return new_coordinate

def GetMeanPlane(coordinates):
    """
    Compute the mean plane
    
    Input:
    
    coordinates: array

    Return:
    
    R1: array

    R2: array
    """
    N = coordinates.shape[0] # ring size
    R1 = np.dot(np.sin(2*np.pi*np.arange(0,N)/N),coordinates)
    R2 = np.dot(np.cos(2*np.pi*np.arange(0,N)/N),coordinates)
    return R1, R2

def GetNormal(coordinates):
    """
    Compute normal to mean plan

    Input:

    coordinates: array

    Output:

    unit_normal: array
    """
    R1,R2 = GetMeanPlane(coordinates)
    cross_product = np.cross(R1,R2)
    unit_normal = cross_product/np.linalg.norm(cross_product) 
    return unit_normal

def Displacement(coordinates):
    """
    Compute the displacement (z) 
    
    Input:
    
    coordinates: array

    Output:

    Z: array
    """
    n = GetNormal(coordinates)
    z = np.dot(coordinates, n)
    return z
   
def GetRingPuckerCoords(coordinates):
    """
    Compute Ring Pucker Parameters
    
    Input:

    coordinates: array
    
    Return:

    qs: puckering amplitude (q_i>=0 for all i)
    
    angle: angle defined in 0<= phi_i <= 2pi 

    """

    N = coordinates.shape[0]  # number of atoms in the ring
    z = Displacement(coordinates)
    if N>4 and N<=20: # In our analysis, we fit it to be smaller than 16.
        if (N%2 == 0): # N even
            m = range(2,int((N/2)))
            cos_component = [np.dot(z,np.cos(2*np.pi*k*np.arange(0,N)/N)) for k in m]
            sin_component = [np.dot(z,np.sin(2*np.pi*k*np.arange(0,N)/N)) for k in m]
            qcos = fixzero(np.sqrt(2/N)*np.array(cos_component))
            qsin = fixzero(-np.sqrt(2/N)*np.array(sin_component))
            q = np.sqrt(qsin**2 + qcos**2)
            amplitude = np.append(q, (1/np.sqrt(N))*np.dot(z,np.cos(np.arange(0,N)*np.pi)).sum()).tolist()
            angle = np.arctan2(qsin,qcos).tolist()
        else: # N odd
            m = range(2,int((N-1)/2)+1)
            cos_component = [np.dot(z,np.cos(2*np.pi*k*np.arange(0,N)/N)) for k in m]
            sin_component = [np.dot(z,np.sin(2*np.pi*k*np.arange(0,N)/N)) for k in m]
            qcos = fixzero(np.sqrt(2/N)*np.array(cos_component))
            qsin = fixzero(-np.sqrt(2/N)*np.array(sin_component))
            amplitude = np.sqrt(qsin**2 + qcos**2).tolist()
            angle = np.arctan2(qsin,qcos).tolist()
    else:
        print("Ring Size is too big or too small")
    return amplitude, angle

def GetTotalPuckeringAmplitude(amplitude):
    """
    Compute Total Puckering Amplitude

    Input:
    
    amplitude: array

    Output:

    Q: positive float
    """
    Q = np.sqrt(np.square(amplitude).sum())
    return Q

###############################
#### Substituents Analysis ####
###############################
def GetRingSubstituentPosition(mol, ring, ring_substituent):
    """
    mol: rdMol
    
    ring: list (ring index)
    
    ring_substitutent: tuples  (ring atom, substituent)
    
    Return:
    
    alpha: float range [0,pi]
    
    beta: float range [0,2*pi)
    """
    molconformer = mol.GetConformer()
    bondlength = rdMolTransforms.GetBondLength(molconformer, *ring_substituent)
    ring_coord = [list(molconformer.GetAtomPosition(node)) for node in ring]
    substituent_coord = [list(molconformer.GetAtomPosition(node)) for node in ring_substituent]
    ringcenter = np.mean(ring_coord, axis=0)
    ring_coord_ = np.array(ring_coord) - ringcenter
    substituent_coord_ = np.array(substituent_coord) - ringcenter
    S = np.diff(substituent_coord_,axis=0)
    s = S/np.linalg.norm(S)
    n = GetNormal(ring_coord_)
    alpha = np.asscalar(np.arccos(fixzero(np.dot(s,n))))
    R = np.array(substituent_coord)[0]
    U = R - np.dot(R,n)*n
    u = U/np.linalg.norm(U)
    v = np.cross(n,u)
    su = fixzero(np.dot(s,u))
    sv = fixzero(np.dot(s,v))
    beta = np.asscalar(np.arctan2(-sv,su))
    return alpha, beta

#################################
#### Ring Atom/Bond Property ####
#################################
def GetRingElement(mol, idxlist):
    """
    Get Element of particular atom

    Input:
    
    mol: rdMol

    idx: list (all int)

    Return:

    atomicnum: list of tuples  [(ring atom idx, atomic number)]
    """
    atoms = [(node, mol.GetAtomWithIdx(node).GetAtomicNum()) for node in idxlist]
    return atoms

def CompareElementSum(orderlist, ringsize, elements):
    """
    Select atom orders using element sum in a ring

    Input:

    orderlist: list of list

    ringsize: int

    elements: list of int (list of atomic numbers) 

    Return:

    pos_order: list of list
    """
    k = int(ringsize/2+1) if ringsize%2==0 else int((ringsize+1)/2)
    elementdict = dict(elements)
    orderframe = pd.DataFrame(orderlist)
    ele = []
    for i in orderlist:
        tmp = []
        for j in i:
            tmp.append(elementdict.get(j))
        ele.append(tmp)
    esum = pd.DataFrame(ele).cumsum(axis=1).iloc[:,:k]
    minimum = min(esum[k-1])
    seleorder = orderframe[esum[k-1]==minimum]
    seleesum =  esum[esum[k-1]==minimum]
    if len(seleorder)>1:
        for i in range(k-1):
            colmin = min(seleesum[i])
            neworder = seleorder[seleesum[i]==colmin]
            newesum = seleesum[seleesum[i]==colmin]
            seleorder = neworder
            seleesum = newesum
    pos_order = seleorder.values.tolist()
    return pos_order


def GetRingBonds(mol, rp):
    """
    Input:

    mol: rdmol

    rp: ringpath (list)

    Return:

    bonds: list
    """
    N = len(rp)
    bonds_ = [int(mol.GetBondBetweenAtoms(rp[i], rp[(i+1)%N]).GetBondType()) for i in range(N)]
    bonds = [1.5 if b==12 else b for b in bonds_]
    return bonds

def CompareBondSum(orderlist, ringsize, bonds):
    """
    Return:

    pos_order: list of lists [order1, order2, ..., order n]
    """
    k = int(ringsize/2+1) if ringsize%2==0 else int((ringsize+1)/2)
    bonddict = dict(bonds)
    fullb = []
    tmp = []
    for i in orderlist:
        tmp.append([tuple(sorted([i[x%ringsize],i[(x+1)%ringsize]])) for x in range(ringsize)])
    for i in tmp:
        fullb.append([bonddict.get(j) for j in i])
    fullbsum = pd.DataFrame(fullb).cumsum(axis=1).iloc[:,:k]
    fullorder = pd.DataFrame(orderlist)
    maximum = max(fullbsum[k-1])
    seleorder = fullorder[fullbsum[k-1]==maximum]
    selebsum = fullbsum[fullbsum[k-1]==maximum]
    if len(seleorder)>1:
        for i in range(1,k-1):
            colmax = max(selebsum[i])
            neworder = seleorder[selebsum[i]==colmax]
            newbsum = selebsum[selebsum[i]==colmax]
            seleorder = neworder
            selebsum = newbsum
    pos_order = seleorder.values.tolist()
    return pos_order
def GetExocyclicConnectivity(mol,idxlist):
    """
    Input: 

    mol: rdmol

    idxlist: list of int

    Return:

    connectivity: list of tuples  (ring atom idx, bond type that link to substitutent)
    """
    atomidx = []
    bonds = []
    for i in idxlist:
        connected_atoms = mol.GetAtomWithIdx(i).GetNeighbors()
        atoms = [atom.GetIdx() for atom in connected_atoms if atom.GetIdx() not in idxlist]
        for j in atoms:
            atomidx.append(i)
            bonds.append(float(mol.GetBondBetweenAtoms(i,j).GetBondType()))
    connectivity = list(zip(atomidx,bonds))
    return connectivity


def ComputeConnectivityScore(mol, idxlist):
    """

    Input:

    connectivity: list of tuples

    Return:

    cscore:  list of tuples (ring atom, connectivity score)

    cscore 4 levels:
    exocyclic double bond (2),
    two exocyclic non-hydrogen single bond (1.5),
    one exocycle non-hydrogen single bond (1)
    no excocycle non-hydrogen single bond (0)

    """
    connectivity = GetExocyclicConnectivity(mol, idxlist)
    cscore = []
    if len(connectivity)>0:
        dataframe = pd.DataFrame(connectivity, columns=["Idx","Bond"])
        grouped = dataframe.groupby("Idx")
        groupname = list(grouped.groups.keys())
        for k in groupname:
            arr = grouped.get_group(k)["Bond"]
            if max(arr.index)==2:
                cscore.append((k,2.0))
            elif max(arr.index)==1 and len(arr)==2:
                cscore.append((k,1.5))
            elif max(arr.index)==1 and len(arr)==1:
                cscore.append((k,1.0))
            else:
                break
    return cscore


def CompareConnectivity(orderlist, ringsize, cscore):
    """
    Return

    pos_order: list
    """
    k = int(ringsize/2+1) if ringsize%2==0 else int((ringsize+1)/2)
    csidx = [x[0] for x in cscore]
    orderframe = pd.DataFrame(orderlist)
    fullscore = []
    for i in orderlist:
        tmp = []
        for j in i:
            if j not in csidx:
                tmp.append(0.0)
            else:
                tmp.append(cscore[csidx.index(j)][1])
        fullscore.append(tmp)
    scoreframe = pd.DataFrame(fullscore).cumsum(axis=1).iloc[:,:k]
    maximum = max(scoreframe[k-1])
    seleorder = orderframe[scoreframe[k-1]==maximum]
    selescore = scoreframe[scoreframe[k-1]==maximum]
    if len(seleorder)>1:
        for i in range(1,k-1):
            colmax = max(selescore[i])
            neworder = seleorder[selescore[i]==colmax]
            newscore = selescore[selescore[i]==colmax]
            seleorder = neworder
            selescore = newscore
    pos_order = seleorder.values.tolist()
    return pos_order



def OrderByConnectivity(idxlist, ringsize, connectivity):
    """
    Input:

    idxlist:

    connectivity:

    Return:

    pos_order: list

    """
    k = int(ringsize/2+1) if ringsize%2==0 else int((ringsize+1)/2)

    pos_order = []
    return pos_order

def EnumerateAtomOrders(idxlist, ringsize, init_index):
    """
    Eumerate

    Return:

    all_order: list of lists
    """
    all_order = []
    idxarr = np.array(idxlist)
    for i in init_index:
        clock = np.mod(list(range(i,i+ringsize)),ringsize)
        anti = np.mod(list(range(i+1,i+1-ringsize,-1)),ringsize)
        clockorder = idxarr[clock]
        antiorder = idxarr[anti]
        all_order.append(clockorder.tolist())
        all_order.append(antiorder.tolist())
    return all_order

#################################
#### Ring Atom Rearrangement ####
#################################
def GetNeighbours(mol, idx):
    """
    Get Atom Neighbours.
    
    Input:

    mol: rdMol

    idx: Int

    Return:

    atomidx: list

    bonds: list
    """
    connected_atoms = mol.GetAtomWithIdx(idx).GetNeighbors()
    atomidx = [atom.GetIdx() for atom in connected_atoms]
    atomnicno = [atom.GetAtomicNum() for atom in connected_atoms]
    bonds = [int(mol.GetBondBetweenAtoms(idx,x).GetBondType()) for x in atomidx]
    return atomidx, bonds

def Rearrangement(mol, idxlist, order="default"):
    """
    Rearrange atom order (Ordering in Cremer Pople Parameters is important!)
    
    Rearragement order:

    Input

    mol: OBMol

    idx_list:

    order: str ("default","random")

    Return
    """
    # Start with first atom output from ring decomposition algorithm, i.e. highest degree
    ringsize = len(idxlist)
    endpoint = ringsize-1
    ringloop = [idxlist[0]]
    bondorder = []
    for i in range(endpoint):
        atomidx, bonds = GetNeighbours(mol, ringloop[i])
        checklist = list(filter(lambda x: x in idxlist and x not in ringloop, atomidx))
        nextatom  = checklist[0]
        ringloop.append(nextatom)
    # Random 
    if order=="random":
        output = ringloop
    if order=="default":
        # starting atom: triple bond >  double bond > aromatic bond > single bond 
        # If tie: we consider the element orders (Ascending Order), i.e. B>C>N>O>P>S
        # If tie still exist, then consider the exocyclic connectivity of atoms (Descending Order). Start with highest connectivity (number bonds linking to non-Hydrogen)
        # Excoyclic connectivity order (double bond substituents > 2 single bonds substituents > 1 single bond substituents > unsubstituted)
        # If tie still exist, then pick it at random (as it is possibly highly symmetric). E.g. cycloalkane.
        # Orientation: clockwise/anticlockwise (pick the next atoms with highest order as stated above)
        # If Ties happen, randomly pick the starting atoms and orientation.
        # compute half loop length
        k = int(ringsize/2+1) if ringsize%2==0 else int((ringsize+1)/2) 
        # Rbond order
        Rbonds = GetRingBonds(mol, ringloop)
        print(Rbonds)
        Rbonds_ = [1.5 if x==12 else x for x in Rbonds]
        rbondarray = np.array(Rbonds_)
        maxbond = max(Rbonds_)
        Relements = GetRingElement(mol,ringloop)
        relementarray = pd.DataFrame(Relements)[1].values
        connectivity = ComputeConnectivityScore(mol, ringloop)
        rconnectivity = np.array(connectivity)
        finalorder = []
        # Determine the ring ordering
        if maxbond==3.0: # Triple bond exist in Ring (usually in big cycles)
            init_indx = [x for y in np.argwhere(rbondarray==maxbond).tolist() for x in y]
            bondall = EnumerateAtomOrders(ringloop, ringsize, init_indx)
            bond_pos_order = CompareBondSum(bondall, ringsize, Rbonds)
            if len(bond_pos_order)>1: # Tie --> use element order next
                ele_pos_order = CompareElementSum(bond_pos_order, ringsize, Relements)
                if len(ele_pos_order)>1: # Tie again --> use connectivity order next
                    c_pos_order = CompareConnectivity(orderlist, ringsize, connectivity)
                    finalorder = c_pos_order[0] # pick the first output   <---- we terminate here, even c_pos_order could have more than one route
                else:
                    finalorder = ele_pos_order[0]
            else:
                finalorder = bond_pos_order[0]
        elif maxbond==2.0: # Double bond exist in Ring (highest order)
            init_indx = [x for y in np.argwhere(rbondarray==maxbond).tolist() for x in y]
            bondall = EnumerateAtomOrders(ringloop, ringsize, init_indx)
            bond_pos_order = CompareBondSum(bondall, ringsize, Rbonds)
            if len(bond_pos_order)>1: # Tie --> use element order next
                ele_pos_order = CompareElementSum(bond_pos_order, ringsize, Relements)
                if len(ele_pos_order)>1: # Tie again --> use connectivity order next
                    c_pos_order = CompareConnectivity(ele_pos_order, ringsize, connectivity)
                    finalorder = c_pos_order[0] # pick the first output   <---- we terminate here, even c_pos_order could have more than one route
                else:
                    finalorder = ele_pos_order[0]
            else:
                finalorder = bond_pos_order[0]
        elif maxbond==1.5: # Aromatic bond exist in Ring (highest order)
            init_indx = [x for y in np.argwhere(rbondarray==maxbond).tolist() for x in y]
            bondall = EnumerateAtomOrders(ringloop, ringsize, init_indx)
            bond_pos_order = CompareBondSum(bondall, ringsize, Rbonds)
            if len(bond_pos_order)>1: # Tie --> use element order next
                ele_pos_order = CompareElementSum(bond_pos_order, ringsize, Relements)
                if len(ele_pos_order)>1: # Tie again --> use connectivity order next
                    c_pos_order = CompareConnectivity(ele_pos_order, ringsize, connectivity)
                    finalorder = c_pos_order[0] # pick the first output   <---- we terminate here, even c_pos_order could have more than one route
                else:
                    finalorder = ele_pos_order[0]
            else:
                finalorder = bond_pos_order[0]
        else: # SINGLE BOND
            finalorder = []
            minele = min(relementarray) # MINIMUM ATOMIC NUMBER
            if len(rconnectivity)==0:
                rconnectivity = np.array([0.0]*len(ringloop))
            maxscore = np.max(rconnectivity) # MAXIMUM CONNECTIVITY 
            # Case 1: Homocycle (cycle with same elements)
            if np.all(relementarray==minele) & np.all(rconnectivity==maxscore): # Homocycle and all having the same connectivity
                finalorder = ringloop # use initial atom order
            elif np.all(relementarray==minele) & np.any(rconnectivity!=maxscore): # Homocycle with different connectivity    
               init_indx = [x for y in np.argwhere(rconnectivity==maxscore).tolist() for x in y]
               c_all = EnumerateAtomOrders(ringloop, ringsize, init_indx)
               c_pos_order = CompareConnectivity(c_all, ringsize, connectivity)
               finalorder = c_pos_order[0]  # pick the first output  <--- we terminate here, even c_pos_order could have more than one route
            # Case 2: Heterocycle (cycle with different elementes)
            else:
                init_indx = [x for y in np.argwhere(relementarray==minele).tolist() for x in y]
                e_all = EnumerateAtomOrders(ringloop, ringsize, init_indx)
                ele_pos_order = CompareElementSum(e_all, ringsize, Relements)
                if len(ele_pos_order)>1: # Tie again --> use connectivity order next
                    c_pos_order = CompareConnectivity(ele_pos_order, ringsize, connectivity)
                    finalorder = c_pos_order[0] # pick the first output   <---- we terminate here, even c_pos_order could have more than one route
                else:
                    finalorder = ele_pos_order[0]
    return finalorder




