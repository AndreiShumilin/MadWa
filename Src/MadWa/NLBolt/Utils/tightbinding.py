import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt


from ..Io import Nassima_reads as io ### some library to read wannier files made by Nassima
import wannier90io as w90io     ###### existing library to deal with wannier90 files, sometimes quite useful
from ..Math import Gmath
from . import TBtools as tools

__all__ = ['TBH', 'ReadInputFiles']



def bestRvec(r1, r2, lst, cell):
    guess = False
    bestR = 0.0
    besti = None
    for irv,rv in enumerate(lst):
        rRv = rv[0]*cell[0] + rv[1]*cell[1] +rv[2]*cell[2] 
        R = r2 +rRv - r1
        aR = np.linalg.norm(R)
        #print(rv, aR)
        if (not guess) or (aR<bestR):
            guess = True
            bestR = aR
            besti = irv
    return lst[besti]


class TBH:
    ### the class tailored to contain all relevant information on 
    ### a tight-binding Namiltonian
    ### currently always includes two Hmiltonians with spins up and down because it is Tailored for
    ### altermagnets
    def __init__(self):
        ### I supposed to use different procedures to initialize the class, so __init__ is blank
        self.Exist = False


    def fromFiles(self, fileHU, fileHD, inFile=None, centerFile1=None, centerFile2=None, Dim=2):
        r"""
        the most basic procedure to generate it from wannier90
        fileHU, fileHD - _hr files for spins up and down
        inFile --- wannier_in file (to get the cell but also contains projections)
         centerFile1, centerFile2  wannier_centers.xyz files for spin up and down
        Dim - system dimensionality
        """
        ### (*) --- this mark will show the parameters of the object that are used by the code latter
        ### if we write another procedures to initialize the object (for example from WannierBerry)
        ### we will need to set these fields
        
        self.Exist = True
        
        self.Dim = Dim  #(*) dimensionality: some procedures work differently in 2D and 3D
        
        NH1, N = io.read_hr(fileHU)  ## I use a procedure made by Nassima to read _hr files, but maybe this should be remade
        NH2, N = io.read_hr(fileHD)
        self.Nw = N    #(*) number of Wannier functions

        rvecAll = NH1.r_vectors + NH2.r_vectors
        rvecAll = np.array(rvecAll, dtype=np.int32)
        Nnas1 = len(NH1.r_vectors)
        Nnas2 = len(NH2.r_vectors)
        xm, xM = np.min(rvecAll[...,0]) , np.max(rvecAll[...,0])
        ym, yM = np.min(rvecAll[...,1]) , np.max(rvecAll[...,1])
        if self.Dim==3:
            zm, zM = np.min(rvecAll[...,2]) , np.max(rvecAll[...,2])
            
        self.xM = np.max( np.array((np.abs(xm), np.abs(xM))) )        
        self.yM = np.max( np.array((np.abs(ym), np.abs(yM))) )      
        if self.Dim==3:
            self.zM = np.max( np.array((np.abs(zm), np.abs(zM))) )      
        ####  (*) xM,yM,zM - maximum displacements in unit cells along x,y,z axes

        rvecs = []
        if Dim==3:
            for ix in range(xm,xM+1):
                for iy in range(ym,yM+1):
                    for iz in range(ym,yM+1):
                        rvecs.append((ix,iy,iz))
        else:
            for i in range(xm,xM+1):
                for j in range(ym,yM+1):
                    rvecs.append((i,j,0))
        rvecs = np.array(rvecs, dtype=np.int32)
        self.rvecs = rvecs    ### (*)  set of displacement vectors in Hamiltonian

        self.dH1 = {}     ### (*) dictionary for the Hamiltonian for spin up
                         ### sytucture: r-vector (for example (1,0,0)) -> Comlex matrix
        for i in range(Nnas1):
            rv = NH1.r_vectors[i]
            self.dH1[rv] = NH1.hr_matrix[i] 

        self.dH2 = {}
        for i in range(Nnas2):
            rv = NH2.r_vectors[i]
            self.dH2[rv] = NH2.hr_matrix[i] 

        if inFile is None:
            self.cell = np.eye(3)
            self.rcell = tools.ReciprocalCell(self.cell)
        else:
            with open(inFile, 'r') as fh:
                parsed_win = w90io.parse_win_raw(fh.read())   ###!!!!! very useful procedure - can get information from wannier_in files
            a1 = np.array(parsed_win['unit_cell_cart']['a1'])
            a2 = np.array(parsed_win['unit_cell_cart']['a2'])
            a3 = np.array(parsed_win['unit_cell_cart']['a3'])
            cell = np.array((a1,a2,a3))
            self.cell = cell               #(*) unit cell of TB model
            self.rcell = tools.ReciprocalCell(self.cell)
        self.cell2D = self.cell[:2,:2]

        if centerFile1 is None:
            self.coords = np.zeros((self.Nw,3))
            self.coords1 = np.zeros((self.Nw,3))  ##coords1, coords2 - coordinates of the centers of wannier90 functions
            self.coords2 = np.zeros((self.Nw,3))
        else:
            self.coords1 = io.read_centers(centerFile1, self.Nw)
            self.coords2 = io.read_centers(centerFile2, self.Nw)
            self.coords = (self.coords1 + self.coords2)/2    #this is currently not used
            self.at_names, self.at_coords = io.read_centers_atoms(centerFile1)
            #(*) at_names, self.at_coords - names of the atoms and their coordinates
            self.Nat = len(self.at_names)  #(*) nubmer of atoms
    

##-------------------------------------------------------------------------------------------------------------
    def fromWanRes(self, WanU, WanD, Dim=2):
        r"""
        from the MadWa reads of Wannier90
        """
        self.Dim = Dim
        
        self.Nw = WanU.Nw
        
        rvecAll = list(WanU.rvecs) + list(WanD.rvecs)
        rvecAll = np.array(rvecAll, dtype=np.int32)
        xm, xM = np.min(rvecAll[...,0]) , np.max(rvecAll[...,0])
        ym, yM = np.min(rvecAll[...,1]) , np.max(rvecAll[...,1])
        if self.Dim==3:
            zm, zM = np.min(rvecAll[...,2]) , np.max(rvecAll[...,2])
        self.xM = np.max( np.array((np.abs(xm), np.abs(xM))) )        
        self.yM = np.max( np.array((np.abs(ym), np.abs(yM))) )      
        if self.Dim==3:
            self.zM = np.max( np.array((np.abs(zm), np.abs(zM))) )  
        
        rvecs = []
        if Dim==3:
            for ix in range(xm,xM+1):
                for iy in range(ym,yM+1):
                    for iz in range(ym,yM+1):
                        rvecs.append((ix,iy,iz))
        else:
            for i in range(xm,xM+1):
                for j in range(ym,yM+1):
                    rvecs.append((i,j,0))
        rvecs = np.array(rvecs, dtype=np.int32)
        self.rvecs = rvecs    ### (*)  set of displacement vectors in Hamiltonian

        self.dH1 = {}     ### (*) dictionary for the Hamiltonian for spin up
                         ### sytucture: r-vector (for example (1,0,0)) -> Comlex matrix
        for i in range(WanU.Nrv):
            rv = tuple(WanU.rvecs[i])
            H1 = WanU.Hr[i]
            self.dH1[rv] = H1

        self.dH2 = {}
        for i in range(WanD.Nrv):
            rv = tuple(WanD.rvecs[i])
            H2 = WanD.Hr[i]
            self.dH2[rv] = H2

        self.cell = WanU.cell
        self.rcell = tools.ReciprocalCell(self.cell)
        self.cell2D = self.cell[:2,:2]

        ### temporary solution
        self.coords = np.zeros((self.Nw,3))
        self.coords1 = np.zeros((self.Nw,3))  ##coords1, coords2 - coordinates of the centers of wannier90 functions
        self.coords2 = np.zeros((self.Nw,3))
        #(*) at_names, self.at_coords - names of the atoms and their coordinates
        


##-------------------------------------------------------------------------------------------------------------    
    
    # def fromFiles(self, fileHU, fileHD, inFile=None, centerFile1=None, centerFile2=None, Dim=2):
    #     r"""
    #     the most basic procedure to generate it from wannier90
    #     fileHU, fileHD - _hr files for spins up and down
    #     inFile --- wannier_in file (to get the cell but also contains projections)
    #      centerFile1, centerFile2  wannier_centers.xyz files for spin up and down
    #     Dim - system dimensionality
    #     """
    #     ### (*) --- this mark will show the parameters of the object that are used by the code latter
    #     ### if we write another procedures to initialize the object (for example from WannierBerry)
    #     ### we will need to set these fields
        
    #     self.Exist = True
        
    #     self.Dim = Dim  #(*) dimensionality: some procedures work differently in 2D and 3D
        
    #     NH1, N = io.read_hr(fileHU)  ## I use a procedure made by Nassima to read _hr files, but maybe this should be remade
    #     NH2, N = io.read_hr(fileHD)
    #     self.Nw = N    #(*) number of Wannier functions

    #     rvecAll = NH1.r_vectors + NH2.r_vectors
    #     rvecAll = np.array(rvecAll, dtype=np.int32)
    #     Nnas1 = len(NH1.r_vectors)
    #     Nnas2 = len(NH2.r_vectors)
    #     xm, xM = np.min(rvecAll[...,0]) , np.max(rvecAll[...,0])
    #     ym, yM = np.min(rvecAll[...,1]) , np.max(rvecAll[...,1])
    #     if self.Dim==3:
    #         zm, zM = np.min(rvecAll[...,2]) , np.max(rvecAll[...,2])
            
    #     self.xM = np.max( np.array((np.abs(xm), np.abs(xM))) )        
    #     self.yM = np.max( np.array((np.abs(ym), np.abs(yM))) )      
    #     if self.Dim==3:
    #         self.zM = np.max( np.array((np.abs(zm), np.abs(zM))) )      
    #     ####  (*) xM,yM,zM - maximum displacements in unit cells along x,y,z axes

    #     rvecs = []
    #     if Dim==3:
    #         for ix in range(xm,xM+1):
    #             for iy in range(ym,yM+1):
    #                 for iz in range(ym,yM+1):
    #                     rvecs.append((ix,iy,iz))
    #     else:
    #         for i in range(xm,xM+1):
    #             for j in range(ym,yM+1):
    #                 rvecs.append((i,j,0))
    #     rvecs = np.array(rvecs, dtype=np.int32)
    #     self.rvecs = rvecs    ### (*)  set of displacement vectors in Hamiltonian

    #     self.dH1 = {}     ### (*) dictionary for the Hamiltonian for spin up
    #                      ### sytucture: r-vector (for example (1,0,0)) -> Comlex matrix
    #     for i in range(Nnas1):
    #         rv = NH1.r_vectors[i]
    #         self.dH1[rv] = NH1.hr_matrix[i] 

    #     self.dH2 = {}
    #     for i in range(Nnas2):
    #         rv = NH2.r_vectors[i]
    #         self.dH2[rv] = NH2.hr_matrix[i] 

    #     if inFile is None:
    #         self.cell = np.eye(3)
    #     else:
    #         with open(inFile, 'r') as fh:
    #             parsed_win = w90io.parse_win_raw(fh.read())   ###!!!!! very useful procedure - can get information from wannier_in files
    #         a1 = np.array(parsed_win['unit_cell_cart']['a1'])
    #         a2 = np.array(parsed_win['unit_cell_cart']['a2'])
    #         a3 = np.array(parsed_win['unit_cell_cart']['a3'])
    #         cell = np.array((a1,a2,a3))
    #         self.cell = cell               #(*) unit cell of TB model
    #     self.cell2D = self.cell[:2,:2]

    #     if centerFile1 is None:
    #         self.coords = np.zeros((self.Nw,3))
    #         self.coords1 = np.zeros((self.Nw,3))  ##coords1, coords2 - coordinates of the centers of wannier90 functions
    #         self.coords2 = np.zeros((self.Nw,3))
    #     else:
    #         self.coords1 = io.read_centers(centerFile1, self.Nw)
    #         self.coords2 = io.read_centers(centerFile2, self.Nw)
    #         self.coords = (self.coords1 + self.coords2)/2    #this is currently not used
    #         self.at_names, self.at_coords = io.read_centers_atoms(centerFile1)
    #         #(*) at_names, self.at_coords - names of the atoms and their coordinates
    #         self.Nat = len(self.at_names)  #(*) nubmer of atoms


    def fromFiles2(self, fileHU, fileHD, inFile, centerFile1, centerFile2, Dim=2, reqLists = []):
        r"""
        Version of the initialization that tries to reconstructed "un-archived" Hamiltonian
        from an "archived" one with "degeneracy" in Wigner-Seitz vectors
        requires good wavefunction centers
        """
        self.Exist = True
        
        self.Dim = Dim
        
        NH1, N = io.read_hr(fileHU)
        NH2, N = io.read_hr(fileHD)
        self.Nw = N

        ######## reading information from .in and .xyz fles
        ####### they are required in this version
        with open(inFile, 'r') as fh:
            parsed_win = w90io.parse_win_raw(fh.read())
        a1 = np.array(parsed_win['unit_cell_cart']['a1'])
        a2 = np.array(parsed_win['unit_cell_cart']['a2'])
        a3 = np.array(parsed_win['unit_cell_cart']['a3'])
        cell = np.array((a1,a2,a3))
        self.cell = cell
        self.rcell = tools.ReciprocalCell(self.cell)
        self.cell2D = self.cell[:2,:2]

        self.coords1 = io.read_centers(centerFile1, self.Nw)
        self.coords2 = io.read_centers(centerFile2, self.Nw)
        self.coords = (self.coords1 + self.coords2)/2    ###### Bad!!!!!!!
        self.at_names, self.at_coords = io.read_centers_atoms(centerFile1)
        self.Nat = len(self.at_names)
        ######## ------------------------------------------------

        rvecAll = NH1.r_vectors + NH2.r_vectors
        rvecAll = np.array(rvecAll, dtype=np.int32)
        Nnas1 = len(NH1.r_vectors)
        Nnas2 = len(NH2.r_vectors)
        xm, xM = np.min(rvecAll[...,0]) , np.max(rvecAll[...,0])
        ym, yM = np.min(rvecAll[...,1]) , np.max(rvecAll[...,1])
        if self.Dim==3:
            zm, zM = np.min(rvecAll[...,2]) , np.max(rvecAll[...,2])
            
        self.xM = np.max( np.array((np.abs(xm), np.abs(xM))) )        
        self.yM = np.max( np.array((np.abs(ym), np.abs(yM))) )      
        if self.Dim==3:
            self.zM = np.max( np.array((np.abs(zm), np.abs(zM))) )      

        rvecs = []
        if Dim==3:
            for ix in range(xm,xM+1):
                for iy in range(ym,yM+1):
                    for iz in range(ym,yM+1):
                        rvecs.append((ix,iy,iz))
        else:
            for i in range(xm,xM+1):
                for j in range(ym,yM+1):
                    rvecs.append((i,j,0))
        rvecs = np.array(rvecs, dtype=np.int32)
        self.rvecs = rvecs

        self.dH1 = {}
        self.dH2 = {}
        for rv in rvecs:
            rvt = tuple(rv)
            self.dH1[rvt] = np.zeros((self.Nw, self.Nw), dtype=np.complex128)
            self.dH2[rvt] = np.zeros((self.Nw, self.Nw), dtype=np.complex128)

        
        for i in range(Nnas1):
            rv = NH1.r_vectors[i]
            rvt = tuple(rv)
            H1 = NH1.hr_matrix[i] 

            lst = None
            for ls1 in reqLists:
                if rvt in ls1:
                    lst = ls1
            if lst is None:
                self.dH1[rvt] = H1
            else:
                for i1 in range(self.Nw):
                    for i2 in range(self.Nw):
                        r1 = self.coords1[i1]
                        r2 = self.coords1[i2]
                        rv2 = bestRvec(r1, r2, lst, self.cell)
                        rvt2 = tuple(rv2)
                        self.dH1[rvt2][i1,i2] = H1[i1,i2]

        for i in range(Nnas2):
            rv = NH2.r_vectors[i]
            rvt = tuple(rv)
            H2 = NH2.hr_matrix[i] 

            lst = None
            for ls1 in reqLists:
                if rvt in ls1:
                    lst = ls1
            if lst is None:
                self.dH2[rvt] = H2
            else:
                for i1 in range(self.Nw):
                    for i2 in range(self.Nw):
                        r1 = self.coords2[i1]
                        r2 = self.coords2[i2]
                        rv2 = bestRvec(r1, r2, lst, self.cell)
                        rvt2 = tuple(rv2)
                        self.dH2[rvt2][i1,i2] = H2[i1,i2]
        return self
    

    def getH(self,rv):
        r"""
        gets up and down Hamiltonians based on (integer) r-vector rv
        returns zero matrix of correct shape ir rv in not in the set of r-vectors
        """
        rvt = tuple(rv)
        if rvt in self.dH1.keys():
            H1 = self.dH1[rvt]
        else:
            H1 = np.zeros((self.Nw, self.Nw))

        if rvt in self.dH2.keys():
            H2 = self.dH2[rvt]
        else:
            H2 = np.zeros((self.Nw, self.Nw))
        return H1,H2


    def getHk(self, kv):
        r"""
        calculates Hamiltonians for up and down spins in k-space
        """
        Hres1 = np.zeros((self.Nw, self.Nw), dtype = np.complex128)
        Hres2 = np.zeros((self.Nw, self.Nw), dtype = np.complex128)
        for rv in self.rvecs:
            rvreal = rv[0]*self.cell[0] + rv[1]*self.cell[1] + rv[2]*self.cell[2]
            exp1 = np.exp(1j* (kv@rvreal) )
            H1, H2 = self.getH(rv)
            Hres1 += H1*exp1
            Hres2 += H2*exp1
        return Hres1, Hres2

    def ManyStates(self, N=16):
        r"""
        calculates all the states in a NxN (or NxNxN in 3D)  grid.
        slow proceduere, not using numba
        """
        if self.Dim==3:
            Nz = N
        else:
            Nz = 1
        
        Gk, gk = Gmath.KGrid(N, N, Nz, self.cell)
        statesU = []
        statesD = []
        for ix in range(N):
            for iy in range(N):
                for iz in range(Nz):
                    k = Gk[ix,iy,0]
                    Hu,Hd = self.getHk(k)
                    eeU, evecU = np.linalg.eigh(Hu)
                    eeD, evecD = np.linalg.eigh(Hd)
                    statesU = statesU + list(eeU)
                    statesD = statesD + list(eeD)
        statesU = np.array(statesU)
        statesD = np.array(statesD)
        return statesU, statesD
        


def ReadInputFiles(fileHU, fileHD, inFile=None, centerFile1=None, centerFile2=None, Dim=2):
    TBH1 = TBH()
    TBH1.fromFiles(fileHU, fileHD, inFile=inFile, centerFile1=centerFile1, centerFile2=centerFile2, Dim=Dim)
    return TBH1


def ReadInputFilesDG(fileHU, fileHD, inFile, centerFile1, centerFile2, reqLists, Dim=2):
    TBH1 = TBH()
    TBH1.fromFiles2(fileHU, fileHD, inFile=inFile, centerFile1=centerFile1, centerFile2=centerFile2, reqLists=reqLists, Dim=Dim)
    return TBH1




