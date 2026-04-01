import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt
import numba as nb



ebasis = np.eye(3, dtype=np.float64)

boltzmann_maximum = 24.0


@nb.njit
def igeomS(q,n):
    if n<1:
        return 0
    res = (1-q**n)/(1-q)
    return round(res)

@nb.njit
def CORDindex(ipar, ord1):
    indset = np.zeros((ord1), dtype=np.int32)
    for i in range(0,ord1):
        ip1 = ipar//(3**(i))
        ip2 = ip1%3
        indset[i] = ip2
    return indset

@nb.njit
def SETtoIND(set1):
    ord1 = len(set1)
    i0 = igeomS(3,ord1)
    i1 = 0
    for i in range(ord1):
        n1 = set1[i]
        i1 += n1*(3**i)
    return i0 + i1

@nb.njit
def i_to_ord(i, Nmax=20):
    for n in range(Nmax):
        if igeomS(3,n)>i:
            return n-1
    
@nb.njit
def i_to_info(i):
    ord1 = i_to_ord(i)
    istart = igeomS(3,ord1)
    ic = i-istart
    #set1 = CORDindex(ic, ord1)
    return ord1, ic #, set1

@nb.jit(nopython=True)
def KGridLocal(Nk, k0, dk, dim=2):
    ak = np.linspace(-dk,dk,Nk)

    if dim==2:
        Gk = np.zeros((Nk,Nk,1,3))
        for ix in range(Nk):
            for iy in range(Nk):
                dkv = np.array((ak[ix],ak[iy],0))
                Gk[ix,iy,0] = k0 + dkv
    else:
        Gk = np.zeros((Nk,Nk,Nk,3))
        for ix in range(Nk):
            for iy in range(Nk):
                for iz in range(Nk):
                    dkv = np.array((ak[ix],ak[iy],ak[iz]))
                    Gk[ix,iy,iz] = k0 + dkv

    return Gk



def getGrids(TB, k0, dk, dim=2, Nk=9):
    Nw = TB.Nw
    Kgr = KGridLocal(Nk, k0, dk, dim=dim)
    Nx, Ny, Nz,_ = Kgr.shape
    EgridU = np.zeros((Nx, Ny, Nz, Nw))
    EgridD = np.zeros((Nx, Ny, Nz, Nw))
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                k1 = Kgr[ix,iy,iz]
                H1,H2 = TB.getHk(k1)
                eeU, evecU = np.linalg.eigh(H1)
                eeD, evecD = np.linalg.eigh(H2)
                EgridU[ix,iy,iz] = eeU
                EgridD[ix,iy,iz] = eeD
    return Kgr, EgridU, EgridD


################  part of the code to get numba version of getGrids ###########################
@nb.jit(nopython=True)
def HRtoHK(HRlist, rvlist, kv, Nw, cell):
    Hres = np.zeros((Nw, Nw), dtype = np.complex128)
    for ir,rv in enumerate(rvlist):
        rvreal = rv[0]*cell[0] + rv[1]*cell[1] + rv[2]*cell[2]
        kvr = (kv@rvreal)
        exp1 = np.exp(1j* kvr )
        H1 = HRlist[ir]
        Hres += H1*exp1
    return Hres


@nb.jit(nopython=True)
def NumbaGetGrids(ListHRU, ListHRD, rvlist, k0, dk, Nw, cell, dim=2, Nk=9):
    Kgr = KGridLocal(Nk, k0, dk, dim=dim)
    Nx, Ny, Nz,_ = Kgr.shape
    EgridU = np.zeros((Nx, Ny, Nz, Nw))
    EgridD = np.zeros((Nx, Ny, Nz, Nw))
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                k1 = Kgr[ix,iy,iz]
                #H1,H2 = TB.getHk(k1)
                H1 = HRtoHK(ListHRU, rvlist, k1, Nw, cell)
                H2 = HRtoHK(ListHRD, rvlist, k1, Nw, cell)
                eeU, evecU = np.linalg.eigh(H1)
                eeD, evecD = np.linalg.eigh(H2)
                EgridU[ix,iy,iz] = eeU
                EgridD[ix,iy,iz] = eeD
    return Kgr, EgridU, EgridD


################################################################################################


@nb.njit
def DerivArray(Kgr, Fgr, alpa):
    Nx, Ny, Nz, _ = Kgr.shape
    alpvec = ebasis[alpa]
    shp = (Nx,Ny,Nz)
    if shp[alpa]==1:
        return Kgr, np.zeros(shp, dtype=np.float64)

    shp2 = np.array(shp)
    shp2[alpa] -= 1
    Kgr2 = np.zeros( (shp2[0], shp2[1], shp2[2], 3 ), dtype=np.float64)
    dFgr = np.zeros( (shp2[0], shp2[1], shp2[2]), dtype=np.float64)
    for ix in range(shp2[0]):
        if alpa==0:
            ix1 = ix
            ix2 = ix+1
        else:
            ix1 = ix
            ix2 = ix
        for iy in range(shp2[1]):
            if alpa==1:
                iy1 = iy
                iy2 = iy+1
            else:
                iy1 = iy
                iy2 = iy
            for iz in range(shp2[2]):
                if alpa==2:
                    iz1 = iz
                    iz2 = iz+1
                else:
                    iz1 = iz
                    iz2 = iz
                k1 = Kgr[ix1, iy1, iz1]
                k2 = Kgr[ix2, iy2, iz2]
                f1 = Fgr[ix1, iy1, iz1]
                f2 = Fgr[ix2, iy2, iz2]
                newK = (k1+k2)/2
                Kgr2[ix,iy,iz] = newK
                dk = (k2-k1)@alpvec
                df = (f2-f1)/dk
                dFgr[ix,iy,iz] = df
    return Kgr2, dFgr

@nb.njit
def gaussAverage(Kgr, fGr, k0, dkG):
    K2 = np.sum((Kgr-k0)*(Kgr-k0),axis=3)
    wei = np.exp( -K2/(2*dkG*dkG)  )
    sw = np.sum(wei)
    return np.sum(fGr*wei)/sw


@nb.njit
def getDerivs(Kgr, Egr, k1, maxOrd, dkGau=0.01):
    maxOrd1 = maxOrd+1

    FunL = [Egr.copy(),]
    kgrL = [Kgr.copy(),]
    parL = [gaussAverage(Kgr, Egr, k1, dkGau),]
    
    ic = 1
    for iord in range (1,maxOrd1):
        Nind = 3**iord
        for ic in range(Nind):
            set1 = CORDindex(ic, iord)
            set2 = set1[:-1]
            alpa = set1[-1]
            oldInd = SETtoIND(set2)
            kgr0 = kgrL[oldInd]
            fgr0 = FunL[oldInd]

            kgr1, dfgr = DerivArray(kgr0, fgr0, alpa)
            kgrL.append(kgr1)
            FunL.append(dfgr)
            avres = gaussAverage(kgr1, dfgr, k1, dkGau)
            parL.append(avres)
    return np.array(parL)



###########################################################################################################

def partition(L):
    NL = len(L)
    if NL==1:
        return [[L,],]
    else:
        el1 = L[-1]
        parts0 = partition(L[:-1])
        Nparts = []
        for part in parts0:
            for is1,set1 in enumerate(part):
                newset = set1 + [el1,]
                newpart = [s2 for is2,s2 in enumerate(part) if is2 != is1]
                newpart.append(newset)
                Nparts.append(newpart)
            newpart = part + [[el1,]]
            Nparts.append(newpart)           
    return Nparts


def Npartition(N):
    L1 = [i for i in range(N)]
    return partition(L1)


def PartiArray(Parti, Nord=None):
    if Nord is None:
        Nord = len(Parti[0][0])
    P1 = Parti.copy()
    P2 = []
    for p in P1:
        for s in p:
            if len(s)<Nord:
                for i in range(Nord-len(s)):
                    s.append(-1)
        P2.append(np.array(p))
    return P2


@nb.njit
def BoltzmanDeriv(x, N):

    if N>0:
        if np.abs(x) > boltzmann_maximum: ###/N:
            return 0.0
        
    
    Deno = (1+np.exp(x))**(N+1)
    
    if N==0:
        Nom = 1
    elif N==1:
        Nom = -np.exp(x)
    elif N==2:
        Nom = np.exp(x)*(np.exp(x) - 1)
    elif N==3:
        Nom = -np.exp(x)*(1 - 4*np.exp(x) + np.exp(2*x))
    elif N==4:
        Nom = np.exp(x)*(-1 + 11*np.exp(x) - 11*np.exp(2*x) + np.exp(3*x) )
    elif N==5:
        Nom = -np.exp(x)*( 1 - 26*np.exp(x) + 66*np.exp(2*x) - 26*np.exp(3*x) + np.exp(4*x) )
    else:
        Nom = 0
    return Nom/Deno
    

@nb.njit
def allAlpha(N):
    Nvar = 3**N
    Res = np.zeros((Nvar, N), dtype=np.int32)
    for ivar in range(Nvar):
        xc = ivar
        for i in range(N):
            Res[ivar,i] = xc%3
            xc = xc // 3
    return Res

    
            