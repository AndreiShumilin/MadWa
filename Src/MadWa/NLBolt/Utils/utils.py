import numpy as np
import scipy as sp



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