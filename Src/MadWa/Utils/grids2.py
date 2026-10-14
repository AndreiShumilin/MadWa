import numpy as np
import numba as nb

import matplotlib.pyplot as plt

import MadWa.Tbasic.tbroutines as tbroutines


# Important note: order of the verticies:
# 2D: 0:(00), 1:(01), 2:(10), 3:(11)
# 3D: 0: (000), 1:(001), 2:(010), 3:(011), 4:(100),
#     5:(101), 6:(110), 7:(111)

@nb.njit
def makecell(dim, corner, vx, vy, vz = np.zeros(3, dtype=np.float64)):
    r"""
    creates verticies and the center of a cell based on its left-down-bottom "corner"
    and side vectors vx,vy,vz
    
    dim - dimensionality
    """
    if dim ==2:
        verts = np.zeros((4, 3), dtype=np.float64)
        verts[0] = corner   
        verts[1] = corner + vy
        verts[2] = corner + vx
        verts[3] = corner + vx + vy
        center = corner + vx/2 + vy/2
    else:
        verts = np.zeros((8,3), dtype=np.float64)
        ind = 0
        for dx in (0.0,1.0):
            for dy in (0.0,1.0):
                for dz in (0.0,1.0):
                    verts[ind] = corner + dx*vx + dy*vy + dz*vz
                    ind += 1
        center = corner + vx/2 + vy/2 + vz/2
    return verts, center

def dividecell(dim, Parent_vert, nx, ny, nz, ParentWeight=1.0):
    r"""
    divides the cell based on its verticies Parent_vert
    into nx x ny x nz children cells

    dim - dimensionality
    """
    corner = Parent_vert[0].copy()

    childverts = []
    childcenters = []
    
    nk = nx*ny*nz
    wei2 = ParentWeight/nk
    childweights = [wei2 for i in range(nk)]
    
    if dim==2:
        vx = Parent_vert[2] - Parent_vert[0]
        vy = Parent_vert[1] - Parent_vert[0]
        vz = np.zeros(3, dtype=np.float64)
        nz = 1
    else:
        vx = Parent_vert[4] - Parent_vert[0]
        vy = Parent_vert[2] - Parent_vert[0]
        vz = Parent_vert[1] - Parent_vert[0]

    dvx = vx/nx
    dvy = vy/ny
    dvz = vz/nz
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                corn1 = corner + ix*dvx + iy*dvy + iz*dvz
                #print(dvx, dvy, dvz)
                #print(ix,iy,iz, corn1)
                vert1, cen1 = makecell(dim, corn1, dvx, dvy, dvz)
                childverts.append(vert1)
                childcenters.append(cen1)
    return childverts, childcenters, childweights


@nb.njit
def getEgrid(H, rvecs, deg, centers, cell):
    Nk, N3 = centers.shape
    Nw = H.shape[1]
    Egr = np.zeros((Nk, Nw), dtype=np.float64)
    for ik in range(Nk):
        kv = centers[ik]
        Hk = tbroutines.Hk_njit(H, rvecs, deg, kv, cell)
        evals = np.linalg.eigvalsh(Hk)
        Egr[ik] = evals
    return Egr



###=============================================================================================================           


class KGrid:
    r"""
    The class operating with k-grids including the non-uniform ones
    """
    def __init__(self, dim, cell=np.eye(3)):
        self.dim = dim
        self.cell = cell
        self.N = 0
        self.ready = False
        
        self.Lvertices = []
        self.Lcenters = []
        self.Lweights = []

        self.center = np.zeros((1,3))
        self.weights = np.zeros(1)

    def makeArrays(self):
        r"""
        Prepears centers and weight as numpy arrays usable by other parts of the code
        """
        self.centers = np.array(self.Lcenters)
        self.weights = np.array(self.Lweights)
        self.ready = True

    

    def refineGrid1(self, Criterium, nx=2, ny=2, nz=2, skip_indicies=[], final=True):
        r"""
        refines the k-grid based on 'Criterium' (once)
        Criterium - a user-provided function that takes verticies and center as an argument 
                    and returns True/False. True means that the cell should be refined
        nx, ny, nz - the number of child-cells in each direction
        skip_indicies - indicies to skip (probably to previous information that they do not need refinement)
        final - if True, will make arrays of centers and weights

        return the number of refined cells
        """
        if self.dim==2:
            nz = 1
        nk = nx * ny * nz

        Remove_indicies = []
        Skip_indiciesT1 = []
        child_vertices = []
        child_centers = []
        child_weights = []
        
        Nmodified = 0
        for ik in range(self.N):
            if not (ik in skip_indicies):
                vert1 = self.Lvertices[ik]
                cent1 = self.Lcenters[ik]
                wei1 = self.Lweights[ik]
                ToRef = Criterium(vert1, cent1)
                if ToRef:
                    Lvert2, Lcent2, Lwei2 = dividecell(self.dim, vert1, nx, ny, nz, ParentWeight=wei1)
                    Remove_indicies.append(ik)
                    child_vertices = child_vertices + Lvert2
                    child_centers = child_centers + Lcent2
                    child_weights = child_weights + Lwei2
                    Nmodified += 1
                
        if (Nmodified > 0):
            SkipIndicies2 = []
            Lvertices2 = []
            Lcenters2 = []
            Lweights2 = []
            i2 = 0
            for i in range(self.N):
                if not i in Remove_indicies:
                    Lvertices2.append(self.Lvertices[i])
                    Lcenters2.append(self.Lcenters[i])
                    Lweights2.append(self.Lweights[i])
                    SkipIndicies2.append(i2)
                    i2 += 1

            self.Lvertices = Lvertices2 + child_vertices
            self.Lcenters = Lcenters2 + child_centers
            self.Lweights = Lweights2 + child_weights
            self.N = len(self.Lweights)
        else:
            SkipIndicies2 = [i for i in range(self.N)]
        if final:
            self.makeArrays()
        return Nmodified, SkipIndicies2 

    def refineGridLoop(self, Criterium, nx=2, ny=2, nz=2, MaxIt = 100, LogFun=print):
        r"""
        refines the k-grid untill the 'Criterium' is fulfilled for all the Cells or
        up to MaxIt iterations
        
        Criterium - a user-provided function that takes verticies and center as an argument 
                    and returns True/False. True means that the cell should be refined
        nx, ny, nz - the number of child-cells in each direction

        if LogFun is not None, procedure would make a log if MaxInt is reached
        """
        Iskip = []
        for iit in range(MaxIt):
            Nmod, Iskip = self.refineGrid1(Criterium, nx=nx, ny=ny, nz=nz, skip_indicies=Iskip, final=False)
            if Nmod==0:
                break
        if (Nmod>0) and not(LogFun is None):
            text1 = 'Warning: after '+str(MaxInt)+' interatio, refine criterium is not met \n'
        self.makeArrays()



###=============================================================================================================

def Uniform_Krid(Nkx, Nky, Nkz,  cell=np.eye(3), dim=3 ):
    """
    Makes a uniform KGrid  with a size Nkx x Nky x Nkz
    based on the unit cell "cell"
    Dim2D - True if the problem is in two dimensions
    """
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]
    b1 = 2*np.pi*np.cross(a2, a3)/ np.dot(a1, np.cross(a2, a3))
    b2 = 2*np.pi*np.cross(a3, a1)/ np.dot(a2, np.cross(a3, a1))
    b3 = 2*np.pi*np.cross(a1, a2)/ np.dot(a3, np.cross(a1, a2))

    if (dim==2):
        Nkz = 1
    ds1 = 1.0 / Nkx
    ds2 = 1.0 / Nky
    ds3 = 0.0  if (dim==2) else  1.0/Nkz

    Nk = Nkx * Nky * Nkz

    counter = 0
    vertices = []
    weights = []
    centers = []
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
                center = (s1_center * b1 + s2_center * b2) - (b1+b2)/2
                if (dim==3):
                    center += s3_center * b3 - b3/2
                
                centers.append(center)
                weights.append( 1.0 / Nk)

                if  (dim==2):
                    verts = np.zeros((4, 3), dtype=np.float64)
                    v0 = s1_min * b1 + s2_min * b2  - (b1+b2)/2
                    v1 = s1_min * b1 + s2_max * b2  - (b1+b2)/2
                    v2 = s1_max * b1 + s2_min * b2  - (b1+b2)/2
                    v3 = s1_max * b1 + s2_max * b2  - (b1+b2)/2

                    verts[0] = v0
                    verts[1] = v1
                    verts[2] = v2
                    verts[3] = v3

                else:
                    verts = np.zeros((8, 3), dtype=np.float64)
                    idx = 0
                    for sx in (s1_min, s1_max):
                        for sy in (s2_min, s2_max):
                            for sz in (s3_min, s3_max):
                                verts[idx] = (sx * b1 + sy * b2 + sz * b3) - (b1+b2+b3)/2
                                idx += 1
                vertices.append(verts)
                counter += 1
    #print(counter, Nk)

    Kgr1 = KGrid(dim,cell)
    Kgr1.N = Nk
    Kgr1.Lvertices = vertices
    Kgr1.Lcenters = centers
    Kgr1.Lweights = weights
    Kgr1.makeArrays()
    
    return Kgr1



##############################\
##   Grid Plotting

def plotCell(vert, lw=0.3):
    xx1 = [vert[0][0], vert[2][0]]
    xx2 = [vert[2][0], vert[3][0]]
    xx3 = [vert[3][0], vert[1][0]]
    xx4 = [vert[1][0], vert[0][0]]

    yy1 = [vert[0][1], vert[2][1]]
    yy2 = [vert[2][1], vert[3][1]]
    yy3 = [vert[3][1], vert[1][1]]
    yy4 = [vert[1][1], vert[0][1]]
    plt.plot(xx1, yy1, 'k-', linewidth=lw)
    plt.plot(xx2, yy2, 'k-', linewidth=lw)
    plt.plot(xx3, yy3, 'k-', linewidth=lw)
    plt.plot(xx4, yy4, 'k-', linewidth=lw)


def PlotGrid2D(Gr, plotcells=False, msize=0.75):
    plt.figure(figsize=(15,15))
    x = Gr.centers[...,0]
    y = Gr.centers[...,1]
    plt.plot(x,y,'D', ms=msize)
    if plotcells:
        for i in range(Gr.N):
            plotCell(Gr.Lvertices[i])
    plt.show()



