import numpy as np
import scipy as sp
import numba as nb
from numba import int32, float32, float64    # import the types


@nb.jit(nopython=True)
def KGrid(Nkx, Nky, Nkz,  cell, rKXmax=1, rKYmax=1, rKZmax=1, regime2D=True):
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
    ###gK = (cell[0,0]*rKXmax/np.pi)*(cell[1,1]*rKYmax/np.pi)/(Nkx*Nky)   ####???????  old version  ????????????????

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
def coarse_kgrid_cells(Nkx, Nky, Nkz,  cell, Dim2D=False ):
    """
    Calculates the cell and the centers and the weights. For each center you hace 4 vertices and a weight associated to each cell
    """
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]
    b1 = 2*np.pi*np.cross(a2, a3)/ np.dot(a1, np.cross(a2, a3))
    b2 = 2*np.pi*np.cross(a3, a1)/ np.dot(a2, np.cross(a3, a1))
    b3 = 2*np.pi*np.cross(a1, a2)/ np.dot(a3, np.cross(a1, a2))

    if Dim2D:

        Nkz = 1
    ds1 = 1.0 / Nkx
    ds2 = 1.0 / Nky
    ds3 = 0.0  if Dim2D else  1.0/Nkz

    Nk = Nkx * Nky * Nkz

    centers = np.zeros((Nk, 3), dtype=np.float64)
    if Dim2D:
        vertices = np.zeros((Nk, 4, 3), dtype=np.float64)
    else:
        vertices = np.zeros((Nk, 8, 3), dtype=np.float64)
    weights = np.zeros(Nk, dtype=np.float64)
    counter = 0
    for ix in range(Nkx):
        s1_min = ix * ds1 
        s1_max = s1_min + ds1
        s1_center = (ix + 0.5)*ds1
        
        for iy in range(Nky):
            s2_min = iy * ds2
            s2_max = s2_min + ds2
            s2_center = (iy + 0.5)*ds2
            
            for iz in range(Nkz):
                s3_min = iz * ds3
                s3_max = s3_min + ds3
                s3_center = (iz + 0.5)*ds3
                # Cell center
                center = (s1_center * b1 + s2_center * b2 + s3_center * b3)
                
                centers[counter] = center
                # Weight
                weights[counter] = 1.0 / Nk

                if Dim2D:

                    v0 = s1_min * b1 + s2_min * b2
                    v1 = s1_max * b1 + s2_min * b2
                    v2 = s1_min * b1 + s2_max * b2
                    v3 = s1_max * b1 + s2_max * b2

                    vertices[counter, 0] = v0
                    vertices[counter, 1] = v1
                    vertices[counter, 2] = v2
                    vertices[counter, 3] = v3

                else:
                    verts = np.zeros((8, 3), dtype=np.float64)
                    idx = 0
                    for sx in (s1_min, s1_max):
                        for sy in (s2_min, s2_max):
                            for sz in (s3_min, s3_max):
                                verts[idx] = (sx * b1 + sy * b2 + sz * b3)
                                idx += 1
                    vertices[counter] = verts
                counter += 1
    return centers, vertices, weights
    
@nb.njit
def subdivide_cells(tag_cell, vertices, centers, weigths, nx= 2, ny=2, nz=2, Dim2D=False):
    """
    Routine subdivide the target cell in a nx ny nz size grid. Recalculate the vertices and the centers and the weights for the new cells
    """
    parent_centre = centers[tag_cell]
    parent_vertice =  vertices[tag_cell]
    parent_weight = weigths[tag_cell]
    #dim =  len(tag_vertice)
    if Dim2D:
        nz = 1
        
    nk = nx * ny * nz
    child_centers = np.zeros((nk,3), dtype = np.float64)
    child_weights = np.full(nk, parent_weight/nk, dtype = np.float64)
    
    if Dim2D:
        child_vertices = np.zeros((nk, 4, 3), dtype=np.float64)
        vo =  parent_vertice[0]
        e1 =  parent_vertice[1] - vo
        e2 =  parent_vertice[2] - vo 
        #e3 = np.zeros((1,3), dtype=np.float64)
        e3 = np.zeros(3, dtype=np.float64)
    else:
        child_vertices = np.zeros((nk, 8, 3), dtype=np.float64)
        vo =  parent_vertice[0]
        e1 =  parent_vertice[1] - vo
        e2 =  parent_vertice[2] - vo 
        e3 =  parent_vertice[4] - vo
  
    counter = 0
    for ix in range(nx):
        u0 = ix/nx
        u1 = (ix + 1)/nx

        for iy in range(ny):
                v0 = iy/ny
                v1 = (iy + 1)/ny

                for iz in range(nz):
                    w0 = iz/nz
                    w1 = (iz + 1)/nz

                    if Dim2D:
                        c00 = vo + u0*e1 + v0*e2
                        c10 = vo + u1*e1 + v0*e2
                        c01 = vo + u0*e1 + v1*e2
                        c11 = vo + u1*e1 + v1*e2

                
                        child_vertices[counter,0] = c00
                        child_vertices[counter,1] = c10
                        child_vertices[counter,2] = c01
                        child_vertices[counter,3] = c11

                        child_centers[counter] = (c00 + c10 + c01 + c11)/4
                       
                    else:
                        verts = np.zeros((8,3), dtype = np.float64)
                        idx = 0 
                        for uu in (u0,u1):
                            for vv in (v0,v1):
                                for ww in (w0,w1):
                                    verts[idx] =  vo + uu * e1 + vv * e2 + ww * e3
                                    idx += 1
                        child_vertices[counter] = verts

                        for i in range(len(verts)):
                            child_centers[counter] += verts[i]
                        child_centers[counter] = child_centers[counter]/len(verts)

                    counter = counter + 1 

    return child_vertices, child_centers, child_weights

