import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt
import numba as nb


r'''
Set of functions that apply symmetry operations on atom coordinates
'''


def Conserve(coord):
    return coord


def Inverse(coord, centr=np.zeros(3)):
    return centr - (coord-centr)


def InverseXY(coord, centr=np.zeros(3)):
    coord1 = centr - (coord-centr)
    coord1[2] = coord[2]
    return coord1


def MirrorXY(coord, cen1,cen2):
    exy = np.array((1,1,0))
    ez = np.array((0,0,1))

    cocen = coord - (cen1+cen2)/2
    coordXY = cocen*exy
    eline = (cen1-cen2)*exy
    eline = eline/np.linalg.norm(eline)
    coordXY1 = eline*(coordXY@eline)
    coordXY2 = coordXY - coordXY1
    ncooXY = coordXY1 - coordXY2
    ncoo = ncooXY + cocen[2]*ez  + (cen1+cen2)/2
    return ncoo


def rotateC4(coord, centr=np.zeros(3)):
    co1 = coord - centr
    co2 = np.array((-co1[1],co1[0], co1[2]))
    return centr + co2