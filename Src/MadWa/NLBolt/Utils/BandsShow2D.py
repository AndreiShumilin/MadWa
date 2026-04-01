import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt

from ..Utils import tightbinding as tight


r'''
a set of tool to analize bands for 2D TB models
'''


def Bands2DAk(TB, ak, Nphi=250):
    r'''
    Calculates bands for a given absolute value of k-vector (ak)
    dependent on the polar angel phi
    '''
    phis = np.linspace(0,2*np.pi,Nphi)
    kk = np.array([(ak*np.cos(p), ak*np.sin(p),0) for p in phis ])
    eup = []
    edown = []
    for k1 in kk:
        H1,H2 = TB.getHk( k1 )
        ee1 = np.linalg.eigh(H1)[0]
        ee2 = np.linalg.eigh(H2)[0]
        eup.append(ee1)
        edown.append(ee2)
    eup = np.array(eup)
    edown = np.array(edown)
    return phis, eup, edown

def Bands2DDirection(TB, phi0, AkMax=1, Nk=250):
    r'''
    Calculates bands for a given polar angle phi0 dependent 
    on the absolute value of k-vector  from 0 to AkMax
    '''
    aak = np.linspace(0, AkMax, Nk)
    kk = np.array([(ak*np.cos(phi0), ak*np.sin(phi0),0) for ak in aak ])
    eup = []
    edown = []
    for k1 in kk:
        H1,H2 = TB.getHk( k1 )
        ee1 = np.linalg.eigh(H1)[0]
        ee2 = np.linalg.eigh(H2)[0]
        eup.append(ee1)
        edown.append(ee2)
    eup = np.array(eup)
    edown = np.array(edown)
    return aak, eup, edown

def AnglePlot_Af( bands, emin,emax, styl1='b-', styl2='r-'):
    r'''
    procedure to automatically make a plot for the bands as calculated from Bands2DAk
    '''
    phis, ee1, ee2 = bands
    Np, Ne = ee1.shape
    for i in range(Ne):
        plt.plot(phis, ee1[...,i], styl1)
        plt.plot(phis, ee2[...,i], styl2)
    plt.ylim(emin,emax)
    
    yy = (emin,emax)
    
    dotAngles = [n * np.pi/4 for n in range(9) ]
    Labels1 = ['0', r'$\pi/4$', r'$2\pi/4$',r'$3\pi/4$',r'$4\pi/4$',r'$5\pi/4$',r'$6\pi/4$',r'$7\pi/4$',r'$8\pi/4$']
    
    for dag in dotAngles:
        xx = (dag,dag)
        plt.plot(xx,yy, 'k--')
    
    plt.xticks(dotAngles, Labels1)
    plt.xlim(0,2*np.pi)
