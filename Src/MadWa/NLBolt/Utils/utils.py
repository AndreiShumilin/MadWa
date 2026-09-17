import numpy as np
import scipy as sp


eElectron = 1.602176634e-19
K_to_EV = 0.8617333262145177e-4
dfzero = 1e-9
ex_cutof = 30

eV = 1.602176634e-19
hbar = 1.054571e-34

const_v = eV/(hbar*1e10)
const_D = const_v*const_v*1e-15



def Kpath(Points, Nst = 30):
    r"""
    generates Kpath based on the set of Points
    """
    Npoi0 = len(Points)

    kpoi = [Points[0],]
    Xmarks = [0.,]
    xx = [0,]
    cnt = 0
    len1 = 0.0
    for i in range(Npoi0-1):
        for i1 in range(Nst):
            cnt += 1
            k1 = ((Nst-i1-1)/Nst) * Points[i] + ((i1+1)/Nst)* Points[i+1]
            dk = k1-kpoi[-1]
            adk = np.linalg.norm(dk)
            len1 += adk
            kpoi.append(k1)
            xx.append(len1)
        Xmarks.append(len1)
    return kpoi, xx, Xmarks



def dFmu(E, mu, T):
    r"""
    calculates the derivative of Fermi function on chemical potential
    E - energy, mu-chemical potential, T - temperature in enegy units
    """
    t = (E-mu)/T
    if np.abs(t)>ex_cutof:
        return 0.0
    else:
        r = np.exp(t)/( (np.exp(t) + 1)**2 )
        r = r/T
        return r

def getDtensor(Dos, TDF, Vcell, mu=0, TK=300):
    r"""
    calculates Diffusion tensor in [m^2/s] based on energy-resolved diffusion tensors
    and densities of states, They can be read (np.loadtxt works) from results ob BoltzWann:
    wannier90.1_tdf.dat
    wannier90.1_boltzdos.dat
    (respectively)

    Vcell - unit cell volume (should be provided)
    mu - chemical potential
    TK - temperature [K]
    """
    T = TK*K_to_EV
    Dt = np.zeros((3,3))
    Ndos = Dos.shape[0]
    Ntdf = TDF.shape[0]
    for i in range(Ntdf-1):
        e1 = TDF[i][0]
        df1 = dFmu(e1, mu, T)
        dEtdf = TDF[i+1][0] - TDF[i][0]
        if df1 > dfzero:
            Dt[0,0] += df1*TDF[i][1] * dEtdf
            Dt[0,1] += df1*TDF[i][2] * dEtdf
            Dt[1,0] += df1*TDF[i][2] * dEtdf
            Dt[1,1] += df1*TDF[i][3] * dEtdf
            Dt[0,2] += df1*TDF[i][4] * dEtdf
            Dt[2,0] += df1*TDF[i][4] * dEtdf
            Dt[1,2] += df1*TDF[i][5] * dEtdf
            Dt[2,1] += df1*TDF[i][5] * dEtdf
            Dt[2,2] += df1*TDF[i][6] * dEtdf
    gg = 0
    for i in range(Ndos-1):
        e2 = Dos[i][0]
        dEdos = Dos[i+1][0] - Dos[i][0]
        df2 = dFmu(e2, mu, T)
        if df2 > dfzero:
            gg += ( df2*Dos[i][1]/Vcell ) * dEdos
    return const_D*Dt/gg






def SpinAccumulationParams(tenDu, tenDd, tenSu, tenSd, tauS, eE, ePerp, Efield=1e6, L=1e-6):
    r"""
    tenDu, tenDd - diffustion tensors (u/d - spin), expected in [m^2/s]
    tenSu, tenSd - conductivity tensors (u/d - spin)
    expected in [A/V] in 2D and [A/Vm] in 3D
    tauS - spin relaxation time, expected in [s]
    eE, ePep - directions of electric field and the one perpendicular to it
    Efiled - electric field [V/m]
    L - perpendicular sample size [m] (sample is from -L/2 to L/2)

    result assumes that the spin accumulation follows the law:
    S0 * sinh(y/ls)
    S0 - coefficient in [1/m^2] in 2D or in [1/m^3] in 3D
    ls - spin relaxation length [m]
    result is S0, ls
    """
    sxyU = ePerp@tenSu@eE
    sxyD = ePerp@tenSd@eE
    Du  =  ePerp@tenDu@ePerp
    Dd  =  ePerp@tenDd@ePerp
    D = 2*Du*Dd/(Du + Dd)
    lamS = np.sqrt(D*tauS)

    C1 = Efield*(sxyU-sxyD)*lamS
    C1 /= (eElectron * D)
    C1 /= np.cosh(L/(2*lamS) )
    return C1, lamS, L

def SpinAccumulation_PlotData(pars, N=100):
    r"""
    calculates the data for a plot of spin accumulation based on the resutls of 
    SpinAccumulationParams()
    """
    C, lams, L = pars
    yy = np.linspace(-L/2, L/2, N)
    res = np.zeros((N,N))
    for i in range(N):
        ss = C*np.sinh(yy[i]/lams)
        ss *= Scell
        res[...,i] = ss
    return res
    



