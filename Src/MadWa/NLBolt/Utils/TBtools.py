import numpy as np
import numpy.linalg
import scipy as sp
import numba as nb

from . import tightbinding as tight
from ..Math import Gmath


coordZero=3e-1



@nb.njit
def cellVolume(cell, regime2D=True):
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]

    if regime2D:
        a12 = np.cross(a1,a2)
        cellV = np.sqrt(np.sum(a12*a12))
    else:
        a12 = np.cross(a1,a2)
        cellV = np.dot(a12,a3)
    return np.abs(cellV)

@nb.njit
def ReciprocalCell(cell):
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]
    V = cellVolume(cell, regime2D=False)

    b1 = 2*np.pi*(np.cross(a2,a3))/V
    b2 = 2*np.pi*(np.cross(a3,a1))/V
    b3 = 2*np.pi*(np.cross(a1,a2))/V

    #Rcell = np.array((b1,b2,b3), dtype=np.float64)
    Rcell = np.zeros((3,3), dtype=np.float64)
    Rcell[0] = b1
    Rcell[1] = b2
    Rcell[2] = b3
    
    return Rcell


def ReciprocalCell2(cell):
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]
    V = cellVolume(cell, regime2D=False)

    b1 = 2*np.pi*(np.cross(a2,a3))/V
    b2 = 2*np.pi*(np.cross(a3,a1))/V
    b3 = 2*np.pi*(np.cross(a1,a2))/V

    Rcell = np.array((b1,b2,b3), dtype=np.float64)
    
    return Rcell



@nb.njit
def realposition2(pos, cell):
    r"""
    Calculates real (Cartesian) from the relative postion "pos" and unti cell "cell"
    """
    vec = np.zeros(3)
    vec += pos[0]*cell[0]
    vec += pos[1]*cell[1]
    vec += pos[2]*cell[2]
    return vec

@nb.njit
def PutIntoCell2(cell, co, adjvector = np.zeros(3)):
    r"""
    puts "cartesian" coordinates "co" into unit cell "cell"
    returns new coordinates and (and) displacement vector in untic cell vectors
    """
    pos = np.linalg.solve(np.transpose(cell), co - adjvector)
    pos2 = np.array(( pos[0]%1, pos[1]%1, pos[2]%1  ))
    dv = np.array(( pos[0]//1, pos[1]//1, pos[2]//1  ), dtype = np.int32)
    co2 = realposition2(pos2, cell)
    return co2, dv    


def addElement(dic, rv, iw1, iw2, Hel, Nw):
    if not rv in dic.keys():
        dic[rv] = np.zeros((Nw, Nw), dtype=np.complex128)
    dic[rv][iw1,iw2] = Hel

def addElementPlus(dic, rv, iw1, iw2, Hel, Nw):
    if not rv in dic.keys():
        dic[rv] = np.zeros((Nw, Nw), dtype=np.complex128)
    dic[rv][iw1,iw2] += Hel




def putt_wf_into_cell(TB0, Dim=2, adjvector = np.zeros(3)):
    TB = tight.TBH()
    TB.Exist = True
    TB.Nw = TB0.Nw
    TB.cell = TB0.cell.copy()
    TB.cell2D = TB0.cell2D.copy()
    if hasattr(TB0, "at_names"): 
        TB.at_names = TB0.at_names.copy()
        TB.Nat = len(TB0.at_names)
    if hasattr(TB0, "at_coords"): 
        at_coords0 = TB0.at_coords.copy()
        Nat = len(at_coords0)
        at_coords = [atc-adjvector for atc in at_coords0]
        TB.at_coords = at_coords

    rvecs0u = np.zeros((TB0.Nw,3), dtype = np.int32)
    rvecs0d = np.zeros((TB0.Nw,3), dtype = np.int32)
    TB.coords1 = np.zeros((TB.Nw, 3))
    TB.coords2 = np.zeros((TB.Nw, 3))
    for iw in range(TB0.Nw):
        cooU2, dvU = PutIntoCell2(TB.cell, TB0.coords1[iw], adjvector = adjvector) 
        if (Dim==2):
            dvU[2]=0
        TB.coords1[iw] = cooU2
        rvecs0u[iw] = -dvU

        cooD2, dvD = PutIntoCell2(TB.cell, TB0.coords2[iw], adjvector = adjvector) 
        if (Dim==2):
            dvD[2]=0
        TB.coords2[iw] = cooD2
        rvecs0d[iw] = -dvD


    rvecs = []
    TB.dH1 = {}
    for rv0 in TB0.dH1.keys():
        rv0a = np.array(rv0)
        Hu = TB0.dH1[rv0]
        for iw1 in range(TB.Nw):
            for iw2 in range(TB.Nw):
                #print(rv0a, rvecs0u[iw2], rvecs0u[iw1])
                rvNewA =  rv0a + rvecs0u[iw2] - rvecs0u[iw1]
                rvN = tuple(rvNewA)
                if not rvN in rvecs:
                    rvecs.append(rvN)
                addElement(TB.dH1, rvN, iw1, iw2, Hu[iw1,iw2], TB.Nw)

    TB.dH2 = {}
    for rv0 in TB0.dH2.keys():
        rv0a = np.array(rv0)
        Hd = TB0.dH2[rv0]
        for iw1 in range(TB.Nw):
            for iw2 in range(TB.Nw):
                rvNewA =  rv0a + rvecs0d[iw2] - rvecs0d[iw1]
                rvN = tuple(rvNewA)
                if not rvN in rvecs:
                    rvecs.append(rvN)
                addElement(TB.dH2, rvN, iw1, iw2, Hd[iw1,iw2], TB.Nw)
    
    TB.rvecs = np.array(rvecs)
    return TB
    



def apply_symmetry_UD(TB0, symF, Nmax=3, Nzmax=0, Dim=2):
    Sanity = testSymmetry_atoms(TB0, symF, maxN=Nmax, maxNz=Nzmax, prin=False)
    if not Sanity:
        print('Warning: sanity check failed, review your symmetry finction and, probably, visit a specialist!')
    
    TB = tight.TBH()
    TB.Exist = True
    TB.Nw = TB0.Nw
    TB.cell = TB0.cell.copy()
    TB.cell2D = TB0.cell2D.copy()
    if hasattr(TB0, "at_names"): 
        TB.at_names = TB0.at_names.copy()
        TB.Nat = len(TB0.at_names)
    if hasattr(TB0, "at_coords"): 
        TB.at_coords = TB0.at_coords.copy()

    TB.dH1 = TB0.dH1.copy()
    TB.coords1 = TB0.coords1.copy()
    rvecs = []
    for rv in TB.dH1.keys():
        rvecs.append(rv)

    rvecs0d = np.zeros((TB.Nw,3), dtype = np.int32)
    TB.coords2 = np.zeros((TB.Nw, 3))
    for iw in range(TB.Nw):
        coo20 = symF(TB.coords1[iw])
        cooD, dvD = PutIntoCell2(TB.cell,coo20)
        TB.coords2[iw] = cooD
        rvecs0d[iw] = dvD

    TB.dH2 = {}
    for rv0 in TB.dH1.keys():
        Hin = TB.dH1[rv0]
        for iw1 in range(TB.Nw):
            for iw2 in range(TB.Nw):
                coo2in = TB.coords1[iw2] + rv0[0]*TB.cell[0] + rv0[1]*TB.cell[1] + rv0[2]*TB.cell[2]
                coo20 = symF(coo2in)
                coo2, dv2 = PutIntoCell2(TB.cell, coo20)
                if Dim==2:
                    dv2[2] = 0
                rvec2 = dv2
                
                rvNewA =  rvec2 - rvecs0d[iw1] 
                rvN = tuple(rvNewA)
                
                if not rvN in rvecs:
                    rvecs.append(rvN)
                addElementPlus(TB.dH2, rvN, iw1, iw2, Hin[iw1,iw2], TB.Nw)

    TB.rvecs = np.array(rvecs)
    return TB




def NVecSym(rv, symF):
    rv1 = symF(rv)
    zer1 = symF(np.zeros(3))
    rv2 = rv1-zer1
    Nrv2 = np.array([round(rv2[0]),  round(rv2[1]), round(rv2[2])])
    #print(rv2, Nrv2)
    return(Nrv2)


def apply_symmetry_UD_X(TB0, symF, Nmax=3, Nzmax=0, Dim=2):
    Sanity = testSymmetry_atoms(TB0, symF, maxN=Nmax, maxNz=Nzmax, prin=False)
    if not Sanity:
        print('Warning: sanity check failed, review your symmetry finction and, probably, visit a specialist!')
    
    TB = tight.TBH()
    TB.Exist = True
    TB.Nw = TB0.Nw
    TB.cell = TB0.cell.copy()
    TB.cell2D = TB0.cell2D.copy()
    if hasattr(TB0, "at_names"): 
        TB.at_names = TB0.at_names.copy()
        TB.Nat = len(TB0.at_names)
    if hasattr(TB0, "at_coords"): 
        TB.at_coords = TB0.at_coords.copy()

    TB.dH1 = TB0.dH1.copy()
    TB.coords1 = TB0.coords1.copy()
    rvecs = []
    for rv in TB.dH1.keys():
        rvecs.append(rv)

    rvecs0d = np.zeros((TB.Nw,3), dtype = np.int32)
    TB.coords2 = np.zeros((TB.Nw, 3))
    for iw in range(TB.Nw):
        coo20 = symF(TB.coords1[iw])
        #cooD, dvD = PutIntoCell2(TB.cell,coo20)
        cooD = coo20
        dvD = np.zeros(3)
        TB.coords2[iw] = cooD
        rvecs0d[iw] = dvD

    TB.dH2 = {}
    for rv0 in TB.dH1.keys():
        Hin = TB.dH1[rv0]
        rvNew = NVecSym(rv0, symF)
        rvN = tuple(rvNew)
        for iw1 in range(TB.Nw):
            for iw2 in range(TB.Nw):
                # coo2in = TB.coords1[iw2] + rv0[0]*TB.cell[0] + rv0[1]*TB.cell[1]
                # coo20 = symF(coo2in)
                # coo2, dv2 = PutIntoCell2(TB.cell, coo20)
                # if Dim==2:
                #     dv2[2] = 0
                # rvec2 = dv2
                
                #rvNewA =  rvec2 - rvecs0d[iw1] 
                #rvN = tuple(rvNewA)
                
                if not rvN in rvecs:
                    rvecs.append(rvN)
                addElementPlus(TB.dH2, rvN, iw1, iw2, Hin[iw1,iw2], TB.Nw)

    TB.rvecs = np.array(rvecs)
    return TB




def findWstate(TB, coord, maxN = 3, maxNz=None, precision=coordZero):
    exist = False
    if maxNz is None:
        maxNz = maxN
    for iw in range(TB.Nw):
        for ix in range(-maxN, maxN+1):
            for iy in range(-maxN, maxN+1):
                for iz in range(-maxNz, maxNz+1):
                    co1 = TB.coords[iw] + ix*TB.cell[0] + iy*TB.cell[1] + iz*TB.cell[2]
                    dist = np.linalg.norm(coord - co1)
                    if dist<precision:
                        exist = True
                        return exist, iw, (ix,iy,iz)
    return exist, 0, (0,0,0)
        
def testSymmetry(TB, symF, maxN, maxNz=None, precision=coordZero):
    Good = True
    goods = 0
    for coo1 in TB.coords:
        coo2 = symF(coo1)
        exist, iw, rv = findWstate(TB, coo2, maxN=maxN, maxNz=maxNz, precision=precision)
        Good = Good and exist
        if exist:
            goods += 1
    return Good, goods/TB.Nw
    

def findAtom(TB, coord, maxN = 3, maxNz=None, precision=coordZero):
    exist = False
    if maxNz is None:
        maxNz = maxN
    for iat in range(TB.Nat):
        for ix in range(-maxN, maxN+1):
            for iy in range(-maxN, maxN+1):
                for iz in range(-maxNz, maxNz+1):
                    co1 = TB.at_coords[iat] + ix*TB.cell[0] + iy*TB.cell[1] + iz*TB.cell[2]
                    dist = np.linalg.norm(coord - co1)
                    if dist<precision:
                        exist = True
                        return exist, iat, (ix,iy,iz)
    return exist, 0, (0,0,0)


def testSymmetry_atoms(TB, symF, maxN, maxNz=None, prin=False, precision=coordZero):
    Good = True
    for iat,coo1 in enumerate(TB.at_coords):
        coo2 = symF(coo1)
        exist, iat2, rv = findAtom(TB, coo2, maxN=maxN, maxNz=maxNz, precision=precision)
        Good = Good and exist
        if prin:
            print(TB.at_names[iat]+str(iat), ' ->', TB.at_names[iat2]+str(iat2), ' , ', rv, ' : ', exist)
    return Good

def testSymmetry_atoms2(TB, symF, maxN, maxN2 = 0, maxNz=None, prin=False, precision=coordZero):
    Good = True
    for ix0 in range(-maxN2, maxN2+1):
        for iy0 in range(-maxN2, maxN2+1):
            rve0 = np.array((ix0, iy0, 0))
            for iat,coo1 in enumerate(TB.at_coords):
                coo1a = coo1 + rve0[0]*TB.cell[0] + rve0[1]*TB.cell[1] + rve0[2]*TB.cell[2]
                coo2 = symF(coo1a)
                exist, iat2, rv = findAtom(TB, coo2, maxN=maxN, maxNz=maxNz, precision=precision)
                Good = Good and exist
                if prin:
                    print(TB.at_names[iat]+str(iat), rve0, ' ->', TB.at_names[iat2]+str(iat2), ' , ', rv, ' : ', exist)
    return Good










