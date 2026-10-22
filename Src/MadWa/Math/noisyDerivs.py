import numpy as np
import numba as nb


@nb.njit
def LeastSQFit1(dma, ya, ws=None):
    r"""
    makes a fit with multi-dimensional linear dependence
    dma [Nmes x Ndim] array of magnetization directions differences from the "reference one" (m_a)
    ya  [Nmes] array of the "results" - typically the Hamiltonian terms

    results are the coefficients of the linear dependence and the estimated "noise"
    """
    Nmes, Ndm = dma.shape
    
    if ws is None:
        ws = np.ones(Nmes)

    M = Ndm+1
    X = np.zeros((Nmes, M), dtype = np.float64)
    for j in range(Nmes):
        X[j,0] = 1.0 * ws[j]
        for i in range(1,Ndm+1):
            imod = i-1
            X[j,i] = dma[j,imod] * ws[j]
        
    ymod = ya*ws
    lstResR = np.linalg.lstsq(X, np.real(ymod), rcond=-1)
    lstResI = np.linalg.lstsq(X, np.imag(ymod), rcond=-1)
    coefR = lstResR[0]
    coefI = lstResI[0]

    if len(lstResR[1])>=1:
        noiR = lstResR[1][0]
    else:
        noiR = 0.0

    if len(lstResI[1])>=1:
        noiI = lstResI[1][0]
    else:
        noiI = 0.0
    Nexc = Nmes -Ndm 
    
    noise = np.sqrt((noiR + noiI)/Nexc)
    coef = coefR.astype(np.complex128) + 1j*coefI.astype(np.complex128)
    return coef, noise



@nb.njit
def Iorder2(Ndm):
    M1 = int(Ndm*(Ndm+1) // 2)
    res = np.zeros((M1,2), dtype=np.int32)
    ii = 0
    for i in range(Ndm):
        for j in range(i, Ndm):
            res[ii,0] = i
            res[ii,1] = j
            ii += 1
    return res

@nb.njit
def LeastSQFit3(dma, ya, ws=None):
    Nmes, Ndm = dma.shape
    if ws is None:
        ws = np.ones(Nmes)
    M1 = int(Ndm*(Ndm+1) // 2)
    M = 1 + Ndm + M1 + Ndm
    X = np.zeros((Nmes, M), dtype=np.float64)
    II = Iorder2(Ndm)
    Nexc = Nmes - M
    for j in range(Nmes):
        X[j,0] = 1.0  * ws[j]
        for i in range(1,Ndm+1):
            imod = i-1
            X[j,i] = dma[j,imod] * ws[j]
        for i in range(Ndm+1, Ndm+M1+1):
            imod = i - Ndm -1
            i1,i2 = II[imod]
            X[j,i] = dma[j,i1]*dma[j,i2] * ws[j]
        for i in range(Ndm+M1+1, M):
            imod = i - Ndm - M1 -1
            X[j,i] = (dma[j,imod]**3) * ws[j]
    ymod = ya*ws
    lstResR = np.linalg.lstsq(X, np.real(ymod), rcond=-1)
    lstResI = np.linalg.lstsq(X, np.imag(ymod), rcond=-1)
    coefR = lstResR[0]
    coefI = lstResI[0]
    noise = np.sqrt((lstResR[1][0] + lstResI[1][0])/Nexc)
    coef = coefR.astype(np.complex128) + 1j*coefI.astype(np.complex128)
    return coef, noise



def NoisyDerivative3(dma, ya, logFoo=None, MinExc=1):
    
    def log1(s):
        if not logFoo is None:
            logFoo(s)

    Nmes, Ndm = dma.shape
    Msq = int(Ndm*(Ndm+1) // 2)
    M3 = Ndm + Msq + Ndm
    if Nmes-M3<MinExc:
        log1('Insuficient measurements for 3rd-order procedure \n')
        return None

    Amabs = np.array( [np.linalg.norm(dma[j]) for j in range(Nmes)] )
    Amin = np.min(Amabs)
    #print(Amabs)
    typMA = np.mean(Amabs)
    coef0, noise0 = LeastSQFit3(dma, ya)
    coef0_2 = coef0[Ndm+1:Ndm+Msq+1]
    coef0_3 = coef0[Ndm+Msq+1:]
    Ac2 = np.linalg.norm(coef0_2)/np.sqrt(Msq)
    Ac3 = np.linalg.norm(coef0_3)/np.sqrt(Ndm)
    wm  = np.sqrt( noise0/(Ac2 + Ac3*typMA) )
    wei1 = np.exp(-Amabs*Amabs/(2*wm*wm) + Amin*Amin/(2*wm*wm))
    coef1, noise1 = LeastSQFit1(dma, ya, ws=wei1)
    return coef1, noise1
    


    
            
