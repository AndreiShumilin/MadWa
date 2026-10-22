import numpy as np
import numba as nb

from MadWa.Utils.utils import FindRv


@nb.njit
def PtoWann1(Op, Orv1, W1, rvects1, 
             W2=None, rvects2=None, 
             Trvects=None, RelRvects=None,
             RelDisRvects = None, Auto_full=False):
    r"""
    Transforms an operator from projection space to Wannier.
    (it is possible to consider different wannier bases for lhs and rhs wavefunctions)

    Intermediate step with Nw x Nw matrix (Op) of the term related to a single r-vector
    
    W1, rvects1 - W-matrix and the set of r-vectors for lhs wannier functions
    W2, rvects2 - W-matrix and the set of r-vectors for rhs wannier functions (None -> match lhs ones)
    Trvects - rvectors of the resulting operator (None->match rvects1)
    RelRvects - "relevant" r-vectros for the operator (everything not in RelRvects abut in Trvects automatically would be zero)
    RelDisRvects - "relevant" "displacements" between wannier state and projection

    Auto_full - controls the default values for RelRvects/RelDisRvects
    True: RelRvects and RelDisRvects match Trvects
    False: RelDisRvects = [(0,0,0),], which means that only relevant r-vector (RelRvects) is the vector Orv1

    (numba-optimized)
    """
    if W2 is None:
        W2 = W1
    if rvects2 is None:
        rvects2 = rvects1
    if Trvects is None:
        Trvects = rvects1
    if RelRvects is None:
        RelRvects = Trvects
    if RelDisRvects is None:
        if Auto_full:
            RelDisRvects = Trvects
        else:
            RelDisRvects = np.array(((0,0,0),), dtype=np.int32)
            RelRvects = np.array(((Orv1[0],Orv1[1],Orv1[2]),), dtype=np.int32)
    
    Nr1 = len(Trvects)
    Nw = W1.shape[1]
    Res = np.zeros((Nr1, Nw, Nw), dtype=np.complex128)
    for rv in RelRvects:
        irv, fou0 = FindRv(Trvects, rv)
        for dv1 in RelDisRvects:
            dv2 =  dv1 + Orv1 - rv
            idv2, fou = FindRv(RelDisRvects, dv2)
            if fou:
                idv1, fou_dv1 = FindRv(rvects1, dv1)
                idv2, fou_dv2 = FindRv(rvects2, dv2)
                all_ok = fou0 and fou_dv1 and fou_dv2
                if all_ok:
                    tW1 = W1[idv1]
                    tW2 = W2[idv2]
                    tW2C = np.conjugate(np.transpose(tW2))
                    R1 = tW1@Op@tW2C
                    Res[irv] += R1
    return Res
                
    
    
@nb.njit
def PtoWann(Op, Orvecs, W1, rvects1, 
             W2=None, rvects2=None, 
             Trvects=None, RelRvects=None,
             RelDisRvects = None, Auto_full=False):
    r"""
    transform to Wannier space a operator in proj.space with single r-vector
    """
    if Trvects is None:
        Trvects = rvects1
    Nr1 = len(Trvects)
    Nw = W1.shape[1]
    Res = np.zeros((Nr1, Nw, Nw), dtype=np.complex128)
    
    for iorv, orv in enumerate(Orvecs):
        Res1 = PtoWann1(Op[iorv], orv, W1, rvects1, 
             W2=W2, rvects2=rvects2,
             Trvects=Trvects, RelRvects=RelRvects,
             RelDisRvects = RelDisRvects, Auto_full=Auto_full)
        Res += Res1
    return Res



@nb.njit
def WtoProj1(Op1, Orv1, W1, rvects1, 
             W2=None, rvects2=None, 
             Trvects=None, RelRvects=None,
             RelDisRvects = None, Auto_full=False):
    r"""
    transform to Wannier space a operator in proj.space with single r-vector
    """
    if W2 is None:
        W2 = W1
    if rvects2 is None:
        rvects2 = rvects1
    if Trvects is None:
        Trvects = rvects1
    if RelRvects is None:
        RelRvects = Trvects
    if RelDisRvects is None:
        if Auto_full:
            RelDisRvects = Trvects
        else:
            RelDisRvects = np.array(((0,0,0),), dtype=np.int32)
    
    Nr1 = len(Trvects)
    Nw = W1.shape[1]
    Res = np.zeros((Nr1, Nw, Nw), dtype=np.complex128)
    for rv in RelRvects:
        irv, fou0 = FindRv(Trvects, rv)
        for dv1 in RelDisRvects:
            dv2 =  dv1 + Orv1 - rv
            idv2, fou = FindRv(RelDisRvects, dv2)
            if fou:
                idv1, fou_dv1 = FindRv(rvects1, dv1)
                idv2, fou_dv2 = FindRv(rvects2, dv2)
                all_ok = fou0 and fou_dv1 and fou_dv2
                if all_ok:
                    tW1 = W1[idv1]
                    tW2 = W2[idv2]
                    tW2C = np.conjugate(np.transpose(tW2))
                    R1 = tW1@Op1@tW2C
                    Res[irv] += R1
    return Res

@nb.njit
def WtoProj(Ow, Orvecs, W1, rvects1, 
             W2=None, rvects2=None, 
             Trvects=None, RelRvects=None,
             RelDisRvects = None, Auto_full=False):
    r"""
    transform to Wannier space a operator in proj.space with single r-vector
    """
    if W2 is None:
        W2 = W1
    if rvects2 is None:
        rvects2 = rvects1
    if Trvects is None:
        Trvects = rvects1
    if RelRvects is None:
        RelRvects = Trvects
    if RelDisRvects is None:
        if Auto_full:
            RelDisRvects = Trvects
        else:
            RelDisRvects = np.array(((0,0,0),), dtype=np.int32)

    Nr1 = len(Trvects)
    Nw = W1.shape[1]
    Res = np.zeros((Nr1, Nw, Nw), dtype=np.complex128)
    for iorv, orv in enumerate(Orvecs):
        Res1 =  WtoProj1(Ow[iorv], orv, W1, rvects1, 
             W2=W2, rvects2=rvects2, 
             Trvects=Trvects, RelRvects=RelRvects,
             RelDisRvects = RelDisRvects, Auto_full=Auto_full)
        Res += Res1
    return Res
        





