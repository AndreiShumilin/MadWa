import numpy as np
import numba as nb

from ..Utils.utils import ex,ey,ez

from ..Tbasic import tbasic
from ..Tbasic import tbroutines 
from ..Utils import utils as ut

from . import loperators as loper

__all__ = ['s0','sx','sy','sz','makeRMap','readRMap','projections_from_TB','projections_from_out','Wud_from_proj_1','Wud_from_proj_All']

s0 = np.array(((1,0),(0,1)), dtype=np.complex128)
sx = np.array(((0,1),(1,0)), dtype=np.complex128)
sy = np.array(((0,-1j),(1j,0)), dtype=np.complex128)
sz = np.array(((1,0),(0,-1)), dtype=np.complex128)

@nb.njit
def makeRMap(Rvecs):
    Mx = np.max(np.abs(Rvecs[...,0]))
    My = np.max(np.abs(Rvecs[...,1]))
    Mz = np.max(np.abs(Rvecs[...,2]))
    Nx = 2*Mx+1
    Ny = 2*My+1
    Nz = 2*Mz+1
    Map = np.zeros((Nx, Ny, Nz), dtype=np.int32) - 1
    for ir, r in enumerate(Rvecs):
        ix,iy,iz = r
        Map[ix,iy,iz] = ir
    return Map

@nb.njit
def readRMap(Map, rv):
    Nx, Ny, Nz = Map.shape
    Mx = (Nx-1)//2
    My = (Ny-1)//2
    Mz = (Nz-1)//2
    if (np.abs(rv[0])>Mx) or (np.abs(rv[1])>My) or (np.abs(rv[2])>Mz):
        return -10
    return Map[rv[0],rv[1],rv[2]]


def projections_from_TB(TB, ax=ex, az=ez, r=1, zona=1.0):
    prj0 = TB.proj
    projDetails = []

    def addProjections(At, prj):
        coo = At[1]
        l = prj[2]
        Nm = 2*l+1
        for m in range(1, Nm+1):
            projD = {}
            projD['atom'] = At[0]
            projD['center'] = coo
            projD['l'] = l
            projD['mr'] = m
            projD['r'] = r
            projD['z-axis'] = az
            projD['x-axis'] = ax
            projD['zona'] =zona
            projDetails.append(projD)
    
    for p in prj0:
        atName = p[0]
        for at in TB.atoms:
            if at[0] == atName:
                addProjections(at, p)
    return projDetails


def projections_from_out(fname: str, cell=np.eye(3)):
    r"""
    !!!!!! currently does not read the names of atoms which is a critical issue !!!!!!
    """
    def extract_wannier_projections_out(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        proj_index = next((i for i, line in enumerate(lines) if "PROJECTIONS" in line.upper()), None)
        if proj_index is None:
            print("PROJECTIONS section not found.")
            return None
        border1_index = next((j for j in range(proj_index, len(lines)) if lines[j].strip().startswith("+")), None)
        if border1_index is None:
            print("First border line not found after PROJECTIONS.")
            return None
        header_index = border1_index + 1
        border2_index = header_index + 1
        if border2_index >= len(lines) or not lines[border2_index].strip().startswith("+"):
            print("Second border (after header) not found.")
            return None
        start_data = border2_index + 1
        end_data = next((k for k in range(start_data, len(lines)) if lines[k].strip().startswith("+")), len(lines))
        data_lines = [
            line.strip().strip("|").strip()
            for line in lines[start_data:end_data]
            if line.strip() and line.strip().startswith("|")
        ]
        return data_lines

    dlines = extract_wannier_projections_out(fname)
    if dlines is None:
        return None
    if len(dlines) == 0:
        return None
    projections = []
    for lin in dlines:
        prj = {}
        data = lin.split()
        centerA = float(data[0])*cell[0] + float(data[1])*cell[1] + float(data[2])*cell[2]
        prj['center'] = centerA
        prj['l'] = int(data[3])
        prj['mr'] = int(data[4])
        prj['r'] = int(data[5])
        prj['z-axis'] = np.array([float(data[6]), float(data[7]), float(data[8])])
        prj['x-axis'] = np.array([float(data[9]), float(data[10]), float(data[11])])
        prj['zona'] = float(data[12])
        projections.append(prj)
    return projections


@nb.njit
def Wud_from_proj_1(WTBu, WTBd, rv, drvecs, RMapU, RMapD):
    #rvA = np.array((int(rv[0]), int(rv[1]), int(rv[2])), dtype=np.int32)
    rvA = np.asarray(rv, dtype=np.int32)
    Nw = WTBu.shape[1]
    Nrv = WTBu.shape[0]
    WRes =  np.zeros((Nw, Nw), dtype = np.complex128)
    for rv1 in drvecs:
        rv2 = rv1 - rvA
        i1 = readRMap(RMapU, rv1)
        i2 = readRMap(RMapD, rv2)
        bad1 = (i1<0) or (i2<0)
        bad2 = (i1 >= Nrv) or (i2>=Nrv)
        if bad2:
            print('warning: incorrectMap')
        if not (bad1 or bad2): 
            W1 = WTBu[i1]
            W2 = WTBd[i2]
            Wcon = W1@np.conjugate(np.transpose(W2))
            WRes += Wcon
    return WRes

@nb.njit(parallel=True)
def Wud_from_proj_All(WTBu, WTBd, TargRvecs, drvecs, RMapU, RMapD):
    TNrv = len(TargRvecs)
    Nw = WTBu.shape[1]
    Nrv = WTBu.shape[0]
    
    Wud = np.zeros((TNrv, Nw, Nw), dtype=np.complex128)

    for irv, rv in enumerate(TargRvecs):
        Wud1 = Wud_from_proj_1(WTBu, WTBd, rv, drvecs, RMapU, RMapD)
        Wud[irv] = Wud1.copy()
    return Wud
