import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np

def curveLine(p0, p1, crv, Np=30):
    len1 = np.linalg.norm( p1 - p0 )
    tt = np.linspace(0,1,Np)

    vmove = p1 - p0
    ez = np.array((0,0,1), dtype=np.float64)
    shftV = np.cross(ez,vmove)*crv
    
    xl = []
    yl = []
    for t in tt:
        p = p0*(1-t) + t*p1
        pSh = p + shftV*t*(1-t)
        xl.append(pSh[0])
        yl.append(pSh[1])
    xar = np.array(xl)
    yar = np.array(yl)
    return xar, yar


def DrawTBHam2D(rvecs, Htb, cen, cell=np.eye(3), MaxNx = 10000, MaxNy=10000, curMax=0.3, 
                cmapS=pl.cm.viridis, cmapLn=pl.cm.bwr, absMax=0.01, fileF=None, AtCoords=None, AtSize=10):
    Nrv = rvecs.shape[0]
    Nw = Htb.shape[1]

    i0 = -1
    for irv,rv in enumerate(rvecs):
        if ((rv[0]==0) and (rv[1]==0) and (rv[2]==0) ):
            print(rv, irv)
            i0 = irv
    if i0<0:
        print('Pozor: no zero r-vector!')
        return None
    print(i0, rvecs[i0])

    H0 = Htb[i0]
    Ens = np.real(np.diagonal(H0))
    Emax = np.max(Ens)
    Emin = np.min(Ens)
    print(Emin, Emax)

    H0nd = H0.copy()
    for i in range(Nw):
        H0nd[i,i]=0.0
    HElMax = np.max(np.abs(H0nd))
    print(HElMax)

    plt.figure(figsize=(15,15))
    for irv,rv in enumerate(rvecs):
        if np.abs(rv[0]) < MaxNx:
            if np.abs(rv[1]) < MaxNy:
                for iw in range(Nw):
                    r1 = cen[iw] + rv[0]*cell[0] + rv[1]*cell[1]
                    colN = 2*(Ens[iw] - Emin)/(Emax-Emin) -1
                    col1 = cmapS(colN)
                    plt.plot((r1[0],),(r1[1],), 'o', color=col1)

    for irv,rv in enumerate(rvecs):
        if np.abs(rv[0]) < MaxNx:
            if np.abs(rv[1]) < MaxNy:
                for iw1 in range(Nw):
                    for iw2 in range(Nw):
                        p0 = cen[iw1]*np.array((1.0,1.0,0.0))
                        p1 = (cen[iw2]  + rv[0]*cell[0] + rv[1]*cell[1]  )*np.array((1.0,1.0,0.0))
                        Hel = Htb[irv,iw1,iw2]
                        if np.abs(Hel)/HElMax > absMax:
    
                            crv = (1 - np.abs(Hel)/HElMax)*curMax
                            crv * np.sign(np.real(Hel))
                            #crv = 0.1
                            xx1, yy1 = curveLine(p0, p1, crv)
    
                            colN = np.abs(Hel)/HElMax*np.sign(np.real(Hel))
                            
                            col2 = cmapLn(colN)
                            
                            plt.plot(xx1,yy1,'-', color=col2, linewidth=0.5)


                           
    if not AtCoords is None:
        for atc in AtCoords:
            xl = (atc[0],)
            yl = (atc[1],)
            plt.plot(xl,yl,'bx',markersize=AtSize)
        
    if not fileF is None:
        plt.savefig(fileF, bbox_inches='tight')
    plt.show()
