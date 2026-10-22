import numpy as np


def LastInstance(lines, mark):
    r"""
    finds the last instance of the string "mark"
    in a set of lines
    """
    ires = -1
    for iln,ln in enumerate(lines):
        if mark in ln:
            ires = iln
    return ires


def getByAtomInfo(lines, mark, shift=4):
    r"""
    gets by-atom info shown after the mark, like
             'magnetization (z)'
    """
    i1 = LastInstance(lines, mark)
    if i1 < 0:
        return np.array([[]])
    i1 = i1 + shift
    lnl = lines[i1].split()
    res = []
    while lnl[0].isdigit():
        N = len(lnl)
        l1 = [float(lnl[n]) for n in range(1,N)]
        res.append( np.array(l1) )
        i1 += 1
        lnl = lines[i1].split()
        while len(lnl)==0:
            i1 += 1
            lnl = lines[i1].split()
    return np.array(res)


def getXtot(ar, NNatoms = None):
    r"""
    gets the total from the by-atom info 
    (obtained by getByAtomInfo)

    NNatoms - a set of atoms under the interest.
            is set only these atoms would be taken into account
    """
    res = 0
    for iel, el in enumerate(ar):
        if (NNatoms is None) or ( (iel+1) in NNatoms):
            if len(el)>0:
                res += el[-1]
    return res

def MagnetizFromOUTCAR(fout, NNatoms=None, zero=1e-7):
    r"""
    calculates the magnetization from the OUTCAR file

    NNatoms - a set of atoms under the interest.
            is set only these atoms would be taken into account
    """
    markX = "magnetization (x)"
    markY = "magnetization (y)"
    markZ = "magnetization (z)"
    with open(fout, 'r') as f:
        lines = f.readlines()

    arx = getByAtomInfo(lines, markX)
    ary = getByAtomInfo(lines, markY)
    arz = getByAtomInfo(lines, markZ)

    mx = getXtot(arx, NNatoms=NNatoms)
    my = getXtot(ary, NNatoms=NNatoms)
    mz = getXtot(arz, NNatoms=NNatoms)
    
    mv = np.array((mx,my,mz))
    am = np.linalg.norm(mv)
    if am>zero:
        mdir = mv/am
    else:
        mdir = np.zeros(3)
    return mdir