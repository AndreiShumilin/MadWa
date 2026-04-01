import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt
import numba as nb


@nb.njit
def generateSERet(con, gcont):
    N = len(con)
    SE = np.zeros((N,N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            SE[i,j] += np.conjugate(con[i])*con[j]
    SE *= 1j*np.pi*gcont
    return SE



@nb.jit(nopython=True)
def KGrid(Nkx, Nky, Nkz, cell, rKXmax=1, rKYmax=1, rKZmax=1, regime2D=True):
    Kgrid1 = np.zeros((Nkx,Nky,Nkz,3), dtype=np.float64)
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]
    b1 = 2*np.pi*np.cross(a2, a3)/ np.dot(a1, np.cross(a2, a3))
    b2 = 2*np.pi*np.cross(a3, a1)/ np.dot(a2, np.cross(a3, a1))
    b3 = 2*np.pi*np.cross(a1, a2)/ np.dot(a3, np.cross(a1, a2))

    if regime2D:
        rKZmax1 = 0.0
    else:
        rKZmax1 = rKZmax
    
    akx = np.linspace(-rKXmax,rKXmax,Nkx )
    aky = np.linspace(-rKYmax,rKYmax,Nky )
    akz = np.linspace(-rKZmax1,rKZmax1,Nkz )
    for ix in range(Nkx):
        for iy in range(Nky):
            for iz in range(Nkz):
                Kgrid1[ix,iy,iz] = akx[ix]*(b1/2) + aky[iy]*(b2/2) + akz[iz]*(b3/2)

    if regime2D:
        a12 = np.cross(a1,a2)
        Sa = np.sqrt(np.sum(a12*a12))
        b12 = np.cross(b1,b2)
        Sb = np.sqrt(np.sum(b12*b12))
        gK = Sa*Sb/(4*np.pi*np.pi*Nkx*Nky)
        gK *= rKXmax*rKYmax
    else:
        a12 = np.cross(a1,a2)
        Va = np.dot(a12,a3)
        b12 = np.cross(b1,b2)
        Vb = np.dot(b3,b12)
        gK = (Va*Vb)/((2*np.pi)**3 *Nkx *Nky *Nkz)
        gK *= rKXmax*rKYmax*rKZmax
    return Kgrid1, gK



@nb.njit
def Gtunn(H1, Lc, Rc, gcont=0.01, Ef=0):
    N = H1.shape[0]
    SRl = generateSERet(Lc, gcont)
    SRr = generateSERet(Rc, gcont)
    SAl = np.transpose(np.conjugate(SRl))
    SAr = np.transpose(np.conjugate(SRr))

    Ham1 = Ef*np.eye(N, dtype=np.complex128) - H1 - SRl - SRr

    GR1 = np.linalg.inv(Ham1)
    GA1 = np.transpose(np.conjugate(GR1))

    gammaL = 1j*(SRl - SAl)
    gammaR = 1j*(SRr - SAr)

    res1 = np.trace(gammaL@GR1@gammaR@GA1)

    return np.abs(res1)


@nb.njit
def simpleGtunn(ee, evec, Lc, Rc, gcont=0.01, Ef=0):
    r"""
    Simple formula, should work for "tunneling case"
    but seems to work always when the contacts are sufficiently separated
    """
    res = 0
    N1 = len(ee)
    for i in range(N1):
            e1 = ee[i]
            psi1 = evec[...,i].copy()
            tt1 = Lc@psi1
            tt1 *= np.conjugate(Rc@psi1)
            tt1 /= (Ef - e1)
            res += tt1
    res2 = np.abs(res*res)
    res2 *= 4*np.pi*np.pi*gcont*gcont
    return res2



@nb.njit
def simpleGtunnInt(ee, evec, Lc, Rc, Ef1, Ef2, gcont=0.01, delta=0.01, Nint=50):
    r"""
    Simple formula, should work for "tunneling case"
    but seems to work always when the contacts are sufficiently separated
    """
    res = 0
    N1 = len(ee)
    eef = np.linspace(Ef1,Ef2,Nint)
    deFer = eef[1]-eef[0]
    RRes = 0
    for Ef in eef:
        for i in range(N1):
                e1 = ee[i]
                psi1 = evec[...,i].copy()
                tt1 = Lc@psi1
                tt1 *= np.conjugate(Rc@psi1)
                tt1 /= (Ef - e1 - 1j*delta)
                res += tt1
        res2 = np.abs(res*res)
        res2 *= 4*np.pi*np.pi*gcont*gcont
        RRes += res2*deFer
    return RRes


@nb.njit
def bilin(cor1, cor2, fcorners):
    x1,y1 = cor1[0], cor1[1]
    x2,y2 = cor2[0], cor2[1]
    fvec = np.asarray(fcorners)
    Mat = np.array([
        (x2*y2, -x2*y1, -x1*y2, x1*y1),
        (-y2, y1, y2, -y1),
        (-x2, x2, x1, -x1),
        (1, -1, -1, 1)
    ], dtype=np.complex128)
    Mat = Mat/( (x2-x1)*(y2-y1) )
    avec = Mat@fvec  
    return avec




###############################################################

integralZero = 1e-6

@nb.njit
def analiticKint0(kc1,kc2, avec):
    a,b,c,d = avec
    x1,y1 = kc1[0], kc1[1]
    x2,y2 = kc2[0], kc2[1]

    Mul1 = (0.25)*(y2-y1)*(x2-x1)
    T1 = 4*a + 2*b*(x1+x2) + (2*c+d*(x1+x2))*(y1+y2)
    Res = Mul1*T1    
    return Res

@nb.njit
def analiticKintX(kc1,kc2, avec, Rx):
    if np.abs(Rx)<integralZero:
        return analiticKint0(kc1,kc2, avec)
    a,b,c,d = avec
    x1,y1 = kc1[0], kc1[1]
    x2,y2 = kc2[0], kc2[1]

    Mul1 = 1j*(y2-y1)/(2*Rx*Rx)

    T1 = 2*a*Rx + 2*b*(1j+Rx*x1) + (1j*d +c*Rx + d*Rx*x1)*(y1+y2)
    T1 = T1*np.exp(1j*Rx*x1)

    T2 = 2*a*Rx + 2*b*(1j+Rx*x2) + (1j*d + c*Rx + d*Rx*x2)*(y1+y2)
    T2 = T2*np.exp(1j*Rx*x2)

    Res = Mul1*(T1-T2)
    return Res

@nb.njit
def analiticKintY(kc1,kc2, avec, Ry):
    a,b,c,d = avec
    avecY = (a,c,b,d)
    kc1Y = (kc1[1], kc1[0])
    kc2Y = (kc2[1], kc2[0])
    return analiticKintX(kc1Y,kc2Y, avecY, Ry)



@nb.njit
def analiticKint(kc1,kc2, fcorners, Rv):
    f11, f12, f21, f22 = fcorners
    a,b,c,d = bilin(kc1,kc2, fcorners)
    avec = (a,b,c,d)
    x1,y1 = kc1[0], kc1[1]
    x2,y2 = kc2[0], kc2[1]
    Rx,Ry = Rv[0], Rv[1]

    if np.abs(Rx)<integralZero:
        return analiticKintY(kc1,kc2, avec, Ry)
    elif np.abs(Ry)<integralZero:
        return analiticKintX(kc1,kc2, avec, Rx)
    
    Mul1 = 1j*np.exp(1j*Ry*y1)/(Rx*Rx*Ry*Ry)
    
    T1 = np.exp(1j*Rx*x1)*(b-1j*a*Rx -1j*b*Rx*x1) + 1j*np.exp(1j*Rx*x2) *(1j*b + a*Rx + b*Rx*x2)
    T1 = T1 * Ry * (np.exp(1j*Ry*(y2-y1)) -1)  

    T2 = -1 + 1j*Ry*y1 + np.exp(1j*Ry*(y2-y1))*(1-1j*Ry*y2)
    T2 = T2* c*Rx*( np.exp(1j*Rx*x1) - np.exp(1j*Rx*x2) )

    T3 = -1 + 1j*Ry*y1 + np.exp(1j*Ry*(y2-y1)) * (1-1j*Ry*y2)
    T3a = (1j + Rx*x1)*np.exp(1j*Rx*x1) - (1j + Rx*x2)*np.exp(1j*Rx*x2)
    T3 = T3*T3a*d

    Res = Mul1*(T1 + T2 + T3)
    return Res




