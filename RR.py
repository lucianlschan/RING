##################################
#### Ring Reconstruction (RR) ####
##################################

import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import py_rdl
import bondtable

###############################
########## Fix Zero ###########
###############################

def fixzero(x):
    x_ = np.array([0.0]) if np.allclose(0,x, rtol=1e-06, atol=1e-08) else x 
    return x_ 


###################################################
##### Ring Reconstruction for MONOCYCLIC ring #####
###################################################
def GetCoordinate(mol, ringpath):
    """
    Get molecule coordinates

    Input:

    mol: rdmol 

    ringpath: list 

    Return:

    coordinate: list
    """
    coordinate = [list(mol.GetConformer().GetAtomPosition(x)) for x in ringpath]
    return coordinate

def GetRingBondLength(mol, ringpath):
    """
    Get bond length of the ring bonds

    Input:

    mol: rdmol 

    ringidx: list 

    Return:

    bondlength: list
 
    """
    N = len(ringpath)
    ringbond = [[ringpath[i], ringpath[(i+1)%N]] for i in range(N)]
    molconf = mol.GetConformer()
    bondlength = [rdMolTransforms.GetBondLength(molconf, *b) for b in ringbond]
    return bondlength

def SetInitialRingBondLength(mol, ringpath):
    """
    Set inital ring bond length for the reference planar ring

    Input:

    mol: rdmol 

    ringidx: list 

    Return:

    bondlength: list
 
    """
    N = len(ringpath)
    pair = [[ringpath[i], ringpath[(i+1)%N]] for i in range(N)]
    first_ele = [mol.GetAtomWithIdx(x[0]).GetAtomicNum() for x in pair]
    second_ele = [mol.GetAtomWithIdx(x[1]).GetAtomicNum() for x in pair]
    sorted_ele = [sorted(p) for p in zip(first_ele,second_ele)]
    bond = [int(mol.GetBondBetweenAtoms(*x).GetBondType()) for x in pair]
    ele_bond = [(sorted_ele[i][0],sorted_ele[i][1], bond[i]) for i,j in enumerate(bond)]
    bondlength = [bondtable.BONDLENGTH_REF.get(eb,"Unknown bond length, please update BOND LENGTH table") 
            for eb in ele_bond]
    return bondlength

def GetRingBondAng(mol, ringpath):
    """
    Get bond angles of the ring

    Input:

    mol: rdmol 

    ringpath: list 

    Return:

    bondang: list (output in radian)
 
    """
    N = len(ringpath)
    atoms = [[ringpath[i], ringpath[(i+1)%N], ringpath[(i+2)%N]] for i in range(N)]
    molconf = mol.GetConformer()
    bondang =[rdMolTransforms.GetAngleRad(molconf, x[0], x[1], x[2]) for x in atoms]
    return bondang

def SetInitialRingBondAng(mol, ringpath):
    """
    Get bond angles of the ring

    Input:

    mol: rdmol 

    ringpath: list 

    Return:

    bondang: list (output in radian)
 
    """
    N = len(ringpath)
    atoms = [[ringpath[i], ringpath[(i+1)%N], ringpath[(i+2)%N]] for i in range(N)]
    first_ele = [mol.GetAtomWithIdx(x[0]).GetAtomicNum() for x in atoms]
    second_ele = [mol.GetAtomWithIdx(x[1]).GetAtomicNum() for x in atoms]
    third_ele = [mol.GetAtomWithIdx(x[2]).GetAtomicNum() for x in atoms]
    sorted_ele = list(zip(first_ele, second_ele, third_ele))
    bond = []
    for a in atoms:
        bond.append([int(mol.GetBondBetweenAtoms(a[i],a[i+1]).GetBondType()) for i in range(2)])
    elements_and_bond_order = [sorted_ele[i]+tuple(bond[i]) for i,j in enumerate(bond)]
    bondang = [bondtable.BONDANGLE_REF.get(x,"Unknown bond length, please update BOND ANGLE table") for x in elements_and_bond_order]
    return bondang

def GetZ(ringsize, q, ang):
    """
    Get Z position from desired puckering amplitude (q) and angle (ang). Angle is in radians

    Input:

    ringsize:

    q:

    ang:

    Return:

    z: list
    """
    qsize = len(q)
    angsize = len(ang)
    assert ((qsize+angsize)==(ringsize-3)) and ringsize>5 and ringsize<16, "Inappropriate Input"
    Z = []
    K = int((ringsize-1)/2)+1 if ringsize%2==1 else int(ringsize/2)
    if ringsize%2==1: # odd ring size
        for j in range(ringsize):
            tmp = np.sum([q[m-2]*np.cos(ang[m-2]+2*np.pi*m*j/ringsize) for m in range(2,K)])
            Z.append(np.sqrt(2/ringsize)*tmp)
    else:  # even ring size
        for j in range(ringsize):
            tmp = np.sqrt(2/ringsize)*np.sum([q[m-2]*np.cos(ang[m-2]+2*np.pi*m*j/ringsize) for m in range(2,K)])+np.sqrt(1/ringsize)*q[-1]*(-1)**j
            Z.append(tmp)
    return Z

def UpdateBondLength(ringsize, Z, init_bondl):
    """
    Update bond length given new z position 

    Input:

    ringsize: int

    z: list

    r: list

    Return:

    new_r: list
    """
    assert len(Z)==ringsize and len(init_bondl)==ringsize, "Inappropriate Input"
    N = ringsize
    nr = [np.sqrt(np.square(init_bondl[i])-np.square(Z[(i+1)%N]-Z[i])) for i in range(N)]
    return nr

def UpdateBeta(ringsize, Z, r, beta):
    """
    Update bond angle given new z position
    """
    assert len(Z)==ringsize and len(r)==ringsize and len(beta)==ringsize, "Inappropriate Input"
    N = ringsize
    r_new = UpdateBondLength(ringsize,Z,r)
    idxlist = [[i,(i+1)%N,(i+2)%N] for i in range(ringsize)]
    beta_new = []
    for i in idxlist:
        tmp_a = np.square(Z[i[2]]-Z[i[0]])-np.square(Z[i[1]]-Z[i[0]])-np.square(Z[i[2]]-Z[i[1]])
        tmp_b = 2*r[i[0]]*r[i[1]]*np.cos(beta[i[0]])
        tmp_c = 2*r_new[i[0]]*r_new[i[1]]
        beta_new.append(np.arccos((tmp_a+tmp_b)/tmp_c))
    return beta_new

def RotationMatrix(phi):
    rotation = np.array([(np.cos(phi), -np.sin(phi)), (np.sin(phi),np.cos(phi))])
    return rotation

def SegCoord(segment, r, beta):
    """
    
    Input:

    segment: list
    """
    coordinate = []
    segsize = len(segment)
    segment_r = [r[x] for x in segment]
    segment_beta = [beta[x] for index, x in enumerate(segment)]
    alpha = []
    gamma = []
    coordinate = [np.array((0,0)),np.array((segment_r[0],0))]
    slength = [segment_r[0]]
    if segsize>=3:
        for index in range(segsize-2):
            if index==0:
                x = slength[-1] + r[index+1]*np.cos(np.pi-beta[index])  
                y = r[index+1]*np.sin(np.pi-beta[index])
                coordinate.append(np.array((x,y)))
            else:
                x = slength[-1] + r[index+1]*np.cos(np.pi-alpha[index-1])
                y = r[index+1]*np.sin(np.pi-alpha[index-1])
                rota = RotationMatrix(gamma[index-1]) 
                coord = np.matmul(rota,np.array((x,y)))
                coordinate.append(coord)
            slength.append(np.linalg.norm(coordinate[-1]))
            alpha.append(segment_beta[index+1]-np.arcsin(np.sin(segment_beta[index])*slength[index]/slength[index+1]))
            gamma.append(np.arctan2(y,x))
    return coordinate

def RingPartition(ringsize, z, r, beta):
    """
    Initialize coordinates for three segments. 

    Input: 

    ringsize: int

    z: list

    r: list

    beta: list

    Return:
    """
    # divide the ring into segments and initialise the coordinate of the segments
    location = list(range(ringsize))
    assert ringsize<=16 and ringsize>=5, "Ring size greater than 16 or smaller than 5 is not supported"
    if ringsize<=7 and ringsize>=5:
        segment1 = location[0:3]
        segment2 = location[2:-1]
        segment2.reverse()
        segment3 = location[-2:]+[location[0]]
        segment3.reverse()
    elif ringsize<=10 and ringsize>=8:
        segment1 = location[0:4]
        segment2 = location[3:-2]
        segment2.reverse()
        segment3 = location[-3:]+[location[0]]
        segment3.reverse()
    elif ringsize<=13 and ringsize>=11:
        segment1 = location[0:5]
        segment2 = location[4:-3]
        segment2.reverse()
        segment3 = location[-4:]+[location[0]]
        segment3.reverse()
    else: #ringsize<=16 and ringsize>=14:
        segment1 = location[0:6]
        segment2 = location[5:-4]
        segment2.reverse()
        segment3 = location[-5:] + [location[0]]
        segment3.reverse()
    segcoord_1_init = SegCoord(segment1, r, beta)
    segcoord_2_init = SegCoord(segment2, r, beta)
    segcoord_3_init = SegCoord(segment3, r, beta)
    Reflection = np.array((-1,1))
    OPsq = np.inner(segcoord_1_init[-1], segcoord_1_init[-1])
    PQsq = np.inner(segcoord_2_init[-1], segcoord_2_init[-1])
    OQsq = np.inner(segcoord_3_init[-1], segcoord_3_init[-1])
    segcoord_1 = [Reflection*item for item in segcoord_1_init]
    segcoord_2 = [x + np.sqrt((OQsq,0)) for x in segcoord_2_init]
    segcoord_3 = [np.array(x) for x in segcoord_3_init]
    # Link segment together
    xp = (OPsq+OQsq-PQsq)/(2*np.sqrt(OQsq))
    yp = np.sqrt(OPsq-np.square(xp))
    phi1, phi2, phi3 = np.arctan2(segcoord_1[-1][1],segcoord_1[-1][0]), np.arctan2(segcoord_2[-1][1], segcoord_2[-1][0]-np.sqrt(OQsq)), np.arctan2(segcoord_3[-1][1], segcoord_3[-1][0])
    phiseg1, phiseg2 = np.arctan2(yp,xp), np.arctan2(yp,xp-np.sqrt(OQsq))
    sigma1, sigma2, sigma3 = np.abs(phi1-phiseg1), np.abs(phiseg2-phi2), np.abs(phi3)
    Rsigma1, Rsigma2, Rsigma3 = RotationMatrix(-sigma1), RotationMatrix(sigma2), RotationMatrix(-sigma3)
    coordinate_1 = [np.array((0,0))]
    seg1_size = len(segcoord_1)
    for i in range(1,seg1_size-1):
        coordinate_1.append(np.matmul(Rsigma1,segcoord_1[i]))
    coordinate_1.append(np.array((xp,yp)))
    #### Check Here ####
    coordinate_2 = []
    seg2_size = len(segcoord_2)
    for i in range(seg2_size-2,0,-1):
        tmp = np.sqrt((OQsq,0))
        coordinate_2.append(tmp + np.matmul(Rsigma2, (segcoord_2[i]-tmp)))
    coordinate_3 = [np.sqrt((OQsq,0))]
    seg3_size = len(segcoord_3)
    for i in range(seg3_size-2,0,-1):
        coordinate_3.append(np.matmul(Rsigma3, segcoord_3[i]))
    coordinate = coordinate_1 + coordinate_2 + coordinate_3
    Rg = np.sum(coordinate,axis=0)
    phig = np.arctan2(Rg[1],Rg[0]) + np.pi/2
    Rphig = RotationMatrix(-phig)    
    newcoord = [np.matmul(Rphig, coordinate[i]-Rg).tolist()+[z[i]] for i in range(ringsize)]
    origin = np.mean(newcoord,axis=0)
    finalcoord = np.array(newcoord)-origin
    return finalcoord

####################################################
#### Call this function to reconstruct the ring ####
####################################################
def SetRingPuckerCoords(mol, ringpath, amplitude, angle, init_bondl, init_bondang):
    """
    Reconstruct the ring from given puckering amplitude and puckering angle

    Input:

    mol: rdmol

    ringidx: list

    amplitdue: list

    angle: list


    Return:

    newcoord: ndarray 
    """
    molcenter = np.array(GetCoordinate(mol, ringpath)).mean(axis=0)
    N = len(ringpath)
    newZ = GetZ(N, amplitude, angle)
    new_bondl = UpdateBondLength(N, newZ, init_bondl)
    new_bondang = UpdateBeta(N, newZ, init_bondl, init_bondang)
    newcoord = np.array(RingPartition(N, newZ, new_bondl, new_bondang)) 
    return newcoord



########################################
########## Ring Substituents  ##########
########################################

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
    alpha=0
    beta=0
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

def SetRingSubstituentPosition(mol, ring, ring_substituent, alpha, beta):
    """
    Update ring subtituent position. Bond length is fixed.

    mol: rdmol

    ring: list  (ring index)

    ring_substituent: list (ring atom index, substituent index)

    alpha: float (0, np.pi)

    beta: float  (0,2*np.pi )

    Return:

    coordinate: list
    """
    coordinate = []
    molconformer = mol.GetConformer()
    bondlength = rdMolTransforms.GetBondLength(molconformer, *ring_substituent)
    ring_coord = [list(molconformer.GetAtomPosition(node)) for node in ring]
    substituent_coord = [list(molconformer.GetAtomPosition(node)) for node in ring_substituent]
    ringcenter = np.mean(ring_coord, axis=0)
    ring_coord_ = np.array(ring_coord) - ringcenter
    substituent_coord_ = np.array(substituent_coord) - ringcenter
    S = np.diff(substituent_coord) # vector from ring atom to substituent
    s = S/np.linalg.norm(S)  
    n = GetNormal(ring_coord_) # Normal vector 
    R = np.array(substituent_coord[0]) # ring atom 
    U = R - np.dot(R,n)*n
    u = U/np.linalg.norm(U)
    v = np.cross(n,u)
    x = fixzero(bondlength*np.sin(alpha)*np.cos(-beta))
    y = fixzero(bondlength*np.sin(alpha)*np.sin(-beta))
    z = fixzero(bondlength*np.cos(alpha))
    b = np.array([x,y,z])
    T = np.array([u,v,n]).T
    ring_substituent_pos = np.matmul(T,b) + np.array(substituent_coord)[0]
 
