import numpy as np


__all__ = ['projDict','readWin']

projDict = {'s':0,
           'p':1,
           'd':2,
           'f':3}

proj_marker1 = 'begin projections'
proj_marker2 = 'end projections'

cell_marker1 = 'begin unit_cell_cart'
cell_marker2 = 'end unit_cell_cart'

atoms_marker1 = 'begin atoms_cart'
atoms_marker2 = 'end atoms_cart'

kpts_marker1 = 'begin kpoints'
kpts_marker2 = 'end kpoints'

windE_markerM = 'dis_win_max'
windE_markerm = 'dis_win_min'
windF_markerM = 'dis_froz_max'
windF_markerm = 'dis_froz_min'

nb_marker = 'num_bands'
nw_marker = 'num_wann'

kpth_marker1 = 'begin kpoint_path'
kpth_marker2 = 'end kpoint_path'

kmesh_marker = 'mp_grid'

def win_projections(lines):
    Nl = len(lines)
    projDict = {'s':0,
               'p':1,
               'd':2,
               'f':3}
    proj_marker1 = 'begin projections'
    proj_marker2 = 'end projections'
    iln = 0
    toread = False

    projL1 = []
    
    for iln in range(Nl):
        ln = lines[iln]
        if proj_marker1 in ln:
            toread = True
        elif proj_marker2 in ln:
            toread = False
        elif toread:
            L1a = ln.split()[0]
            L1b = L1a.split(':')
            if len(L1b) == 2:
                At = L1b[0]
                projT = L1b[1]
                if projT in projDict.keys():
                    projl = projDict[projT]
                else:
                    projl = -1000
                projL1.append( (At, projT, projl)  )      
    return projL1
        

def win_cell(lines):
    Nl = len(lines)
    cell_marker1 = 'begin unit_cell_cart'
    cell_marker2 = 'end unit_cell_cart'
    toread = False
    for iln in range(Nl):
        if cell_marker1 in lines[iln]:
            toread = True
            break  
    if toread:
        cell = np.loadtxt(lines[iln+1: iln+4], dtype=np.float64)
        return cell
    else:
        return None
        
def win_getPar(lines, Par, typ=np.float64):
    toread = False
    Nl = len(lines)
    for iln in range(Nl):
        if Par in lines[iln]:
            toread = True
            break  
    if toread:
        lns = lines[iln].split()
        if (lns[0] == Par) and (lns[1] == '='):
            a0 = lns[2]
            aA = np.array((a0,), dtype=typ)
            a1 = aA[0]
            return a1
    return None

def win_getAtoms(lines):
    Nl = len(lines)
    Atoms = []
    AtTypes = []
    AtCoords = []
    toread = False
    for iln in range(Nl):
        ln = lines[iln]
        if atoms_marker1 in ln:
            toread = True
        elif atoms_marker2 in ln:
            toread = False
            break
        elif toread:
            lns = ln.split()
            if len(lns)>=4:
                typ = lns[0]
                coords = np.loadtxt(lns[1:4], dtype=np.float64)
                Atoms.append( (typ, coords)  )
                AtTypes.append(typ)
                AtCoords.append(coords)
    AtCoords = np.array(AtCoords)
    return Atoms, AtTypes, AtCoords

def win_kpts(lines):
    Nl = len(lines)
    kpts = []
    toread = False
    for iln in range(Nl):
        ln = lines[iln]
        if kpts_marker1 in ln:
            toread = True
        elif kpts_marker2 in ln:
            toread = False
            break
        elif toread:
            lns = ln.split()
            if len(lns)>=3:
                k = np.loadtxt(lns, dtype=np.float64)
                kpts.append( k  )
    kpts = np.array(kpts)
    return kpts



def win_kpth(lines):
    Points = []
    Labels = []
    Points2 = []
    Labels2 = []
    Nl = len(lines)
    toread = False
    for iln in range(Nl):
        ln = lines[iln]
        if kpth_marker1 in ln:
            toread = True
        elif kpth_marker2 in ln:
            toread = False
            break
        elif toread:
            lns = ln.split()
            if len(lns)>=4:
                Label = lns[0]
                Label2 = lns[4]
                # p1 = np.loadtxt(lns[1:4], dtype=np.float64)
                # p2 = np.loadtxt(lns[5:8], dtype=np.float64)
                Points.append([float(x) for x in lns[1:4]]) 
                Points.append([float(x) for x in lns[5:8]])
                Labels.append(Label)
                Labels.append(Label2)
    Points2 = [Points[i] for i in  range(len(Points)) if i==0 or Points[i] != Points[i-1]]
    Labels2 = [Labels[i] for i in  range(len(Labels)) if i==0 or Labels[i] != Labels[i-1]]
    Points2 = np.array(Points2)
    return Points2, Labels2


def win_kmesh(lines):
    kmesh = []
    Nl = len(lines)
    for iln in range(Nl):
        line = lines[iln]
        if kmesh_marker in line.lower():
            parts = line.split()
            kmesh.append([int(x) for x in parts[2:5]])
    kmesh = np.array(kmesh)
    return kmesh


def readWin(winfile):
    winDict = {}
    with open(winfile, 'r') as f:
        lines = f.readlines()
    kpth, lbls = win_kpth(lines)
    winDict['kpath'] = kpth 

    winDict['mp_grid'] = win_kmesh(lines)
    
    winDict['nb'] = win_getPar(lines, nb_marker, typ=np.int32)
    winDict['nw'] = win_getPar(lines, nw_marker, typ=np.int32)
    
    winDict['eMax'] = win_getPar(lines, windE_markerM)
    winDict['eMin'] = win_getPar(lines, windE_markerm)
    winDict['eFMax'] = win_getPar(lines, windF_markerM)
    winDict['eFMin'] = win_getPar(lines, windF_markerm)
    
    winDict['cell'] = win_cell(lines)
    winDict['proj'] = win_projections(lines)

    at, atT, atCoo = win_getAtoms(lines)
    winDict['atoms'] = at
    winDict['atom_types'] = atT
    winDict['atom_coords'] = atCoo

    kpts = win_kpts(lines)
    Nk = len(kpts)
    winDict['kpts'] = kpts
    winDict['Nk'] = Nk
    
    return winDict