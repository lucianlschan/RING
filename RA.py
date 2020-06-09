############################
#### Ring Analysis (RA) ####
############################

import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
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

def PuckeringToDisplacement(q, ang, ringsize):
    """
    Get displacement from puckering coordinates

    Input:

    q: list

    ang: list

    ringsize: int

    Output:

    displacement: list
    """
    displacement = []
    if (ringsize%2==0):
        t = int(ringsize/2 -1)
        for s in range(ringsize):
            coscomponent = np.cos([ang[m-2]+2*np.pi*m*s/ringsize for m in range(2,t+1)])
            qm = q[:-1]
            displacement.append(np.sqrt(2/ringsize)*np.dot(qm,coscomponent) + np.sqrt(1/ringsize)*np.array(q[-1])*(-1)**(s))
    else:
        t = int((ringsize-1)/2)
        for s in range(ringsize):
            coscomponent = np.cos([ang[m-2]+2*np.pi*m*s/ringsize for m in range(2,t+1)])
            displacement.append(np.sqrt(2/ringsize)*np.dot(q,coscomponent))
    return displacement



###############################
#### Substituents Analysis ####
###############################
def GetSubstituent(mol, ring, smarts):
    substruct = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(substruct)
    ring_atom_substituent_atom = []
    for match in matches:
        if match[0] in ring:
            ring_atom_substituent_atom.append(match)
    return ring_atom_substituent_atom

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
    R = np.array(substituent_coord_)[0]
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

    idxlist: list (all int)

    Return:

    atomicnum: list of tuples  [(ring atom idx, atomic number)]
    """
    atoms = [(node, mol.GetAtomWithIdx(node).GetAtomicNum()) for node in idxlist]
    return atoms


def GetRingBonds(mol, rp):
    """
    Input:

    mol: rdmol

    rp: ringpath (list)

    Return:

    bonds: list
    """
    N = len(rp)
    bondsorder = [float(mol.GetBondBetweenAtoms(rp[i], rp[(i+1)%N]).GetBondType()) for i in range(N)]
    bondorder_modified = [1.5 if b==12 else b for b in bondsorder]
    bondatom = [tuple(sorted([rp[i],rp[(i+1)%N]])) for i in range(N)]
    bonds = list(zip(bondatom,bondorder_modified))
    return bonds


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

def GetRingAtomType(mol, ringidx):
    """
    Input:
    
    mol: rdMol
    
    ringidx: list
    
    Return:
    
    outcome: list
    
    """
    ringatom = Chem.MolFromSmarts("[R]")
    monocyclicatom = Chem.MolFromSmarts("[R1]")
    bicyclicatom = Chem.MolFromSmarts("[R2]")
    ringatom_match = [a for x in mol.GetSubstructMatches(ringatom) for a in x]
    monocyclic_match = [a for x in mol.GetSubstructMatches(monocyclicatom) for a in x]
    bicyclic_match = [a for x in mol.GetSubstructMatches(bicyclicatom) for a in x]
    outcome = []
    for i in ringidx:
        if (i in ringatom_match) and (i in monocyclic_match):
            outcome.append("M")
        elif (i in ringatom_match) and (i in bicyclic_match):
            outcome.append("B")
        elif (i in ringatom_match) and (i not in monocyclic_match) and (i not in bicyclic_match):
            outcome.append("P")
        else:
            outcome.append("NA")
    return outcome

#################################
#### Ring Atom Rearrangement ####
#################################
def EnumerateAtomOrders(idxlist, ringsize, init_index):
    """
    Eumerate all possible ring path (including clockwise, anticlockwise)

    Return:

    allorder: list of lists
    """
    allorder = []
    idxarr = np.array(idxlist)
    for i in init_index:
        clock = np.mod(list(range(i,i+ringsize)),ringsize)
        anti = np.mod(list(range(i+1,i+1-ringsize,-1)),ringsize)
        clockorder = idxarr[clock]
        antiorder = idxarr[anti]
        allorder.append(clockorder.tolist())
        allorder.append(antiorder.tolist())
    return allorder

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
            if arr.max()==2:
                cscore.append((k,2.0))
            elif arr.max()==1 and len(arr)==2:
                cscore.append((k,1.5))
            elif arr.max()==1.0 and len(arr)==1:
                cscore.append((k,1.0))
            else:
                continue
    contained = [item[0] for item in cscore]
    fscore = cscore + [(k,0.0) for k in idxlist if k not in contained]
    return fscore

def Sorting(data,size):
    """

    """
    midpoint = int(size/2+1) if size%2==0 else int((size+1)/2)
    bonds = data.iloc[:,:midpoint]
    bmax = max(bonds.iloc[:,midpoint-1])
    seleb= data[data.iloc[:,midpoint-1]==bmax]
    selebc = seleb
    for i in range(1,midpoint-1):
        colmax = max(seleb.iloc[:,i])
        neworder = seleb[seleb.iloc[:,i]==colmax]
        seleb = neworder
    cframe = data.iloc[seleb.index,size:size+midpoint]
    cmax = max(cframe.iloc[:,midpoint-1])
    selec = cframe[cframe.iloc[:,midpoint-1]==cmax]
    selecc = selec
    for i in range(1,midpoint-1):
        colmax = max(selec.iloc[:,i])
        neworder = selec[selec.iloc[:,i]==colmax]
        selec = neworder
    eframe = data.iloc[selec.index, 2*size:2*size+midpoint]
    #emax = max(eframe.iloc[:,midpoint-1])
    emin = min(eframe.iloc[:,midpoint-1])
    selee = eframe[eframe.iloc[:,midpoint-1]==emin]
    seleec = selee
    for i in range(1,midpoint-1):
        #colmax = max(selee.iloc[:,i])
        colmin = min(selee.iloc[:,i])
        #neworder = selee[selee.iloc[:,i]==colmax]
        neworder = selee[selee.iloc[:,i]==colmin]
        selee = neworder
    idx = selee.index
    return idx

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
        # starting atom: triple bond > double bond > aromatic bond > single bond 
        # If tie exist, then consider the exocyclic connectivity of atoms (Descending Order). 
        # Start with highest connectivity (number bonds linking to non-Hydrogen)
        # Excoyclic connectivity order (double bond substituents) > 2 single bonds substituents > 1 single bond substituents > unsubstituted
        # If tie: we consider the element orders (Ascending Order), i.e. B>C>N>O>P>S
        # If tie still exist, then pick it at random (as it is possibly highly symmetric). E.g. cycloalkane.
        # Orientation: clockwise/anticlockwise (pick the next atoms with highest order as stated above)
        # If Ties happen, randomly pick the starting atoms and orientation.
        Rbonds = GetRingBonds(mol, ringloop)
        bondarray = np.array([b[1] for b in Rbonds])
        bonddict = dict(Rbonds)
        cdict = dict(ComputeConnectivityScore(mol, ringloop))
        eledict = dict(GetRingElement(mol, ringloop))
        maximum = max(bondarray)
        init_indx = [i for i,j in enumerate(bondarray) if j==maximum]
        orders = EnumerateAtomOrders(ringloop, ringsize, init_indx)
        bondf, connectivityf, elef = [],[],[]
        for order in orders:
            btmp = [tuple(sorted([order[x%ringsize],order[(x+1)%ringsize]])) for x in range(ringsize)] 
            bondf.append([bonddict.get(b) for b in btmp])
            connectivityf.append([cdict.get(atom) for atom in order])
            elef.append([eledict.get(atom) for atom in order])
        cframe = pd.DataFrame(connectivityf, columns=["C{}".format(x) for x in range(ringsize)]).cumsum(axis=1)
        eframe = pd.DataFrame(elef,columns=["E{}".format(x) for x in range(ringsize)]).cumsum(axis=1)
        bondframe = pd.DataFrame(bondf, columns=["B{}".format(x) for x in range(ringsize)]).cumsum(axis=1)
        dataframe = pd.concat([bondframe,cframe,eframe],axis=1)
        index = Sorting(dataframe, ringsize)
        if len(index)>=1:
            output = orders[index[0]]
        else:
            print("Please Check")
            output = ringloop
    return output

######################
#### Eccentricity ####
######################
def GetEccentricity(mol, ring):
    coord = [list(mol.GetConformer().GetAtomPosition(x)) for x in ring]
    ccoord = np.array(coord)-np.array(coord).mean(axis=0)
    pca = PCA(n_components=2)
    projection = pca.fit_transform(ccoord)
    b = np.ones_like(projection[:,0:1])
    X = projection[:,0:1]
    Y = projection[:,1:]
    M = np.hstack([np.square(X),X*Y,np.square(Y),X,Y,b])
    D = np.matmul(np.transpose(M),M)
    C = np.zeros([6,6])
    C[0,2] = 2
    C[1,1] = -1
    C[2,0] = 2
    inverse = np.linalg.inv(D)
    tmp = np.matmul(inverse,C)
    outcome = np.linalg.eig(tmp)
    positiveidx = [y for x in np.argwhere(outcome[0]>0.0).tolist() for y in x]
    a2,b2 = outcome[1][:,positiveidx[0]][0],outcome[1][:,positiveidx[0]][2]
    if np.abs(a2)>np.abs(b2):
        eccentricity = np.sqrt(1-b2/a2)
    else:
        eccentricity = np.sqrt(1-a2/b2)
    return eccentricity

##########################
#### Peptide Ordering ####
##########################
def GetAminoAcidType(mol, CA):
    """
    Get the amino acid single letter code

    Input:

    mol: rdMol

    CA: C alpha (int)

    Output:

    aa: string
    """
    alanine = Chem.MolFromSmarts("[C;R](N)[CH3X4]")
    aspartate = Chem.MolFromSmarts("[C;R](N)[CH2X4][CX3](=[OX1])[OH0-,OH]")
    cysteine = Chem.MolFromSmarts("[C;R](N)[CH2X4][SX2H,SX1H0-]")
    glutamate = Chem.MolFromSmarts("[C;R](N)[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]")
    glycine = Chem.MolFromSmarts("[C;R](N)[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]")
    histidine = Chem.MolFromSmarts("[C;R](N)[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1")
    leucine = Chem.MolFromSmarts("[C;R](N)[CH2X4][CHX4]([CH3X4])[CH3X4]")
    lysine = Chem.MolFromSmarts("[C;R](N)[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]")
    phenylalanine = Chem.MolFromSmarts("[C;R](N)[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1")
    serine = Chem.MolFromSmarts("[C;R](N)[CH2X4][OX2H]")
    threonine = Chem.MolFromSmarts("[C;R](N)[CHX4]([CH3X4])[OX2H]")
    tryptophan = Chem.MolFromSmarts("[C;R](N)[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12")
    tyrosine = Chem.MolFromSmarts("[C;R](N)[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1")
    valine = Chem.MolFromSmarts("[C;R](N)[CHX4]([CH3X4])[CH3X4]")
    sidechain = [tyrosine, tryptophan, phenylalanine, histidine, lysine, glutamate, aspartate, threonine, serine, cysteine,
             leucine, valine, alanine, glycine]
    order = list(range(14))
    aminoacid_rank = ["Y","W","F","H","K","E","D","T","S","C","L","V","A","G"]
    aminoacid = dict(list(zip(order, aminoacid_rank)))
    match = [mol.GetSubstructMatches(x) for x in sidechain]
    for idx, item in enumerate(match):
        if any(item):
            for mat in item:
                if CA==mat[0]:
                    aa = aminoacid.get(idx)
    cneighbors = [atom.GetAtomicNum()==1 for atom in mol.GetAtomWithIdx(CA).GetNeighbors()]
    c_implicit_value = mol.GetAtomWithIdx(CA).GetImplicitValence()
    if sum(cneighbors)==2 or c_implicit_value==2:
        aa = "G"
    return aa

def CTPOrder(mol, ring, n_res=4):
    """
    Get the ring ordering of cyclic peptides
    """
    order = list(range(14))
    aminoacid_rank = ["W","Y","F","K","L","H","V","E","T","D","C","S","A","G"]
    aminoacid = dict(list(zip(aminoacid_rank, order)))
    ringsize = len(ring)
    aatype = [GetAminoAcidType(mol, ring[i]) for i in [1,4,7,10]]
    sorting = [aminoacid_rank.index(a) for a in aatype]
    minidx = sorting.index(min(sorting))
    shift = (n_res-minidx)%n_res
    ringloopidx = [(i+shift*3)%ringsize for i in range(ringsize)]
    ringloop = [x for _, x in sorted(zip(ringloopidx,ring))]
    return ringloop

def GetSideChainFirstAtoms(mol, CA, aminoacid):
    alanine = Chem.MolFromSmarts("[C;R](N)[CH3X4]")
    aspartate = Chem.MolFromSmarts("[C;R](N)[CH2X4][CX3](=[OX1])[OH0-,OH]")
    cysteine = Chem.MolFromSmarts("[C;R](N)[CH2X4][SX2H,SX1H0-]")
    glutamate = Chem.MolFromSmarts("[C;R](N)[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]")
    glycine = Chem.MolFromSmarts("[C;R](N)[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]")
    histidine = Chem.MolFromSmarts("[C;R](N)[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1")
    leucine = Chem.MolFromSmarts("[C;R](N)[CH2X4][CHX4]([CH3X4])[CH3X4]")
    lysine = Chem.MolFromSmarts("[C;R](N)[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]")
    phenylalanine = Chem.MolFromSmarts("[C;R](N)[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1")
    serine = Chem.MolFromSmarts("[C;R](N)[CH2X4][OX2H]")
    threonine = Chem.MolFromSmarts("[C;R](N)[CHX4]([CH3X4])[OX2H]")
    tryptophan = Chem.MolFromSmarts("[C;R](N)[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12")
    tyrosine = Chem.MolFromSmarts("[C;R](N)[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1")
    valine = Chem.MolFromSmarts("[C;R](N)[CHX4]([CH3X4])[CH3X4]")
    aminoacid_list = ["W","Y","F","K","L","H","V","E","T","D","C","S","A","G"]
    smarts = [tryptophan,tyrosine,phenylalanine,lysine,leucine,histidine,valine,glutamate,threonine,aspartate,cysteine,serine,alanine,glycine]
    aminoacid_dict = dict(list(zip(aminoacid_list,smarts)))
    pattern = aminoacid_dict.get(aminoacid)
    matches = mol.GetSubstructMatches(pattern)
    if aminoacid=="G":
        outcome = "NA"
    else:
        for item in matches:
            if item[0]==CA:
                outcome = item[2]
    return outcome
    
