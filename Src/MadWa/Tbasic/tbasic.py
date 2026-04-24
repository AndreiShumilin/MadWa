# Import libraries
#The objective is to build an object to perform basic calculations from Wannier90 files. Provide a type of postprocessing tool that would be compatible with Andrei's work
# file: tbasic.py

import numpy as np

import numba as nb
from .. import wannierIO as wIO
from .tbroutines import *
from ..Utils import utils as ut

__all__ = ['TBasic']

class TBasic:
     # This class only works for a spin-specie of Hamiltonian with spin orbit coupling. For Separated spin channel calculations ask Andrei :D
    """
    Basic Tight-Binding container for Wannier90 Hamiltonians.
    """
    def __init__(self, hr_file = None, win_file=None, tb_file = None):
        self.hr_file = hr_file
        self.tb_file = tb_file
        self.win_file = win_file
        
        # Initialize placeholders (populated by read_file)
        self.H_ij = None
        self.rvects = None
        self.Irvects = None
        self.deg = None
        self.num_wann = 0
        self.cell = None
        self.points = None
        self.kpath = None
        self.kpts = None
 #       self.labels = None
        self.num_rvec = 0
        self.recip_cell = None
        self.ready = False
        self.projections_exist = False

        self.fromTB = False
        
        # Automatically load the data when the object is created
        if not ( (self.hr_file is None) or (self.win_file is None)):
            self.read_file()

        # Automatically load the data if the object is created with tb-file
        if not ( (self.tb_file is None) or (self.win_file is None)):
            self.read_file_tb()

  
    #--------------------------------------------------------------------------------#
    #HR Hamiltonian and its attributes
    #--------------------------------------------------------------------------------#
    def read_file(self, hr_file = None, win_file=None):
        if not hr_file is None:
            self.hr_file = hr_file
        if not win_file is None:
            self.win_file = win_file
        H_ij, rvects, deg, Par = wIO.readHR(self.hr_file)
        self.H_ij = H_ij
        self.rvects = rvects.astype(np.float64)
        self.Irvects = rvects.astype(np.int32)   ### Integer version or rvects, lets keep it
        
        self.deg = deg
        self.num_wann = Par[0]
        self.num_rvec = Par[1]

        # index of (0,0,0) r-vector
        self.iZeroR, ex1 = self.findVecNumber((0,0,0))

        win_data = wIO.readWin(self.win_file)
        self.cell = win_data['cell'].astype(np.float64)
        self.kpts = win_data['kpts'] 
        self.kpath = win_data['kpath'] 
        
 
        self.recip_cell = self._get_reciprocal_lattice()
        self.ready = True

    
    def read_file_tb(self, tb_file = None, win_file=None):
        r"""
        Version of the initialization by reading tb-file of wannier90
        """
        if not tb_file is None:
            self.tb_file = tb_file
        if not win_file is None:
            self.win_file = win_file
            
        rvects, H_ij, r_mat, deg, Par = wIO.read_tb_file(self.tb_file)

        self.H_ij = H_ij
        self.rvects = rvects.astype(np.float64)
        self.Irvects = rvects.astype(np.int32)   ### Integer version or rvects, lets keep it
        
        self.deg = deg
        self.num_wann = Par[0]
        self.num_rvec = Par[1]

        # index of (0,0,0) r-vector
        self.iZeroR, ex1 = self.findVecNumber((0,0,0))

        win_data = wIO.readWin(self.win_file)
        self.cell = win_data['cell'].astype(np.float64)
        self.kpts = win_data['kpts'] 
        self.kpath = win_data['kpath'] 
        
 
        self.recip_cell = self._get_reciprocal_lattice()
        self.fromTB = True
        self.ready = True
    

    def manual(self, H_ij, rvects, deg=None, cell=np.eye(3), kpts=None, kpath=None):
        r"""
        Manualy initiates the object from the user-provided information
        """
        self.H_ij = H_ij.astype(np.complex128)
        self.rvects = rvects.astype(np.float64)
        self.Irvects = rvects.astype(np.int32)
        self.num_wann = H_ij.shape[1]
        self.num_rvec = len(rvects)

        if deg is None:
            self.deg = [1 for i in range(self.num_rvec)]
        else:
            self.deg = deg

        # index of (0,0,0) r-vector
        self.iZeroR, ex1 = self.findVecNumber((0,0,0))

        self.cell = cell
        self.kpts = kpts
        self.kpath = kpath

        self.recip_cell = self._get_reciprocal_lattice()
        self.ready = True

    # ------------------------------------------------------------
    # LATTICE
    # ------------------------------------------------------------
    def _get_reciprocal_lattice(self):
        cell = self.cell
        a1 = cell[0]
        a2 = cell[1]
        a3 = cell[2]
        V = cellVolume(self.cell, Dim2D = False)

        b1 = 2*np.pi*(np.cross(a2,a3))/V
        b2 = 2*np.pi*(np.cross(a3,a1))/V
        b3 = 2*np.pi*(np.cross(a1,a2))/V

        #Kcell = np.array([b1,b2,b3], dtype=np.float64)
        Kcell = np.zeros((3,3), dtype=np.float64)
        Kcell[0] = b1
        Kcell[1] = b2
        Kcell[2] = b3
    
        return Kcell
    # ------------------------------------------------------------
    # get_Hk wrapper
    # ------------------------------------------------------------
    def get_Hk(self, kvec):
        """Wrapper for the njit Hk function."""

        kvec1 = np.asarray(kvec, dtype=np.float64)
        if not self.ready:
            return None
           
        return Hk_njit(self.H_ij, self.rvects, self.deg, kvec1, self.cell)

    # ------------------------------------------------------------
    # Hamiltonian by rvect
    # ------------------------------------------------------------
    def get_Hr(self, rvec):
        """Gets the Hamiltonian for integer r-vector."""
        if not self.ready:
            return None

        rvec1 = np.asarray(rvec, dtype=np.int32)
        for i in range(self.num_rvec):
            if (self.Irvects[i] == rvec1).all():
                return self.H_ij[i]
                
        zerroH = np.zeros((self.Nw, self.Nw), dtype=np.complex128)
        return zerroH
        
    # ------------------------------------------------------------
    # get_bands  wrapper
    # ------------------------------------------------------------
    def get_bands_w90(self, Nst):
        """Wrapper for the njit Bands function."""
        if (not self.ready) or (self.kpath is None):
            return None
        else:
            return bands_w90(self.kpath, self.recip_cell, self.num_wann, self.H_ij, self.rvects, self.deg, self.cell, Nst)

    # ------------------------------------------------------------
    # a little bit generalized version of bands
    # ------------------------------------------------------------
    def bands(self, Nst, Kpath = None):
        """Wrapper for the njit Bands function."""
        if (Kpath is None) and (not(self.kpath is None)):
            Kpath = self.kpath
            
        if (not self.ready) or (Kpath is None):
            return None
        else:
            bands = bands_w90(Kpath, self.recip_cell, self.num_wann, self.H_ij, self.rvects, self.deg, self.cell, Nst)
            kpoi, xx, Xmarks= get_kpath(Kpath, self.recip_cell, Nst)
            return xx, bands, Xmarks

   

    def readProjections(self, seedname, readTB=None):
        """
        reads information on the wannier projections and construct the matrix W relaing wannier Functions 
        to the projections 
        """
        if readTB is None:
            readTB = self.fromTB
        
        WR = wIO.WanRes(seedname=seedname, Short=False, readTB=readTB)
        
        
        self.atoms = WR.fullwinD['atoms']
        self.window = (WR.Emin, WR.Emax)
        self.W = WR.W
                # Reminder:
                # W[ir, iw, ip] = < W_{iw} | P_{ip}(R[ir]) >
                # W_{iw} --- wannier function iw
                # P_{ip}(R[ir]) --- projection ip in the unit cell displaced by vector R[ir]
        self.proj = WR.fullwinD['proj'] 
        self.projections_exist = True

    def manualProjections(self, atoms, proj, W, window=None):
        """
        manualy set projections for Toy models
        """
        self.atoms = atoms
        self.proj = proj
        self.W = W
        if window is None:
            self.window = (-100.0,100.0)
        else:
            window = window
        self.projections_exist = True

    def findVecNumber(self, rv):
        """
        find the number of vector rv and checks if it exists
        rv - integer vector
        """
        arv = np.asarray(rv)
        exi = False
        Inum = -1
        for irv, rv1 in enumerate(self.Irvects):
            if (rv1==arv).all():
                exi = True
                Inum = irv
                #return Inum, exi
        return Inum, exi

    def EFermi_to_Charge(self, Kgr, Ef, D=3):
        r"""
        Calculates the total number of electrons based on Fermi level position Ef
        and k-grid Kgr. 
        D --- system dimension
        """
        cell = self.cell
        rcell = self.recip_cell
        Va = ut.cellVolume(cell, Dim2D=(D==2) )
        Vb = ut.cellVolume(rcell, Dim2D=(D==2) )
        Nkx, Nky, Nkz, _ = Kgr.shape
        gK = (Va*Vb)/((2*np.pi)**D *Nkx *Nky *Nkz)
        
        charge = 0
        for ix in range(Nkx):
            for iy in range(Nky):
                for iz in range(Nkz):
                    k = Kgr[ix,iy,iz]
                    Hk = self.get_Hk(k)
                    ens = np.linalg.eigh(Hk)[0]
                    N1 = np.sum(ens < Ef)
                    charge += N1 * gK
        return charge

    def Total_energy(self, Kgr, Ef, D=3):
        r"""
        calculates total energy based no K-grid Kgr and 
        Fermi energy Ef
        D --- system dimension
        """
        cell = self.cell
        rcell = self.recip_cell
        Va = ut.cellVolume(cell, Dim2D=(D==2) )
        Vb = ut.cellVolume(rcell, Dim2D=(D==2) )
        Nkx, Nky, Nkz, _ = Kgr.shape
        gK = (Va*Vb)/((2*np.pi)**D *Nkx *Nky *Nkz)
        
        charge = 0
        for ix in range(Nkx):
            for iy in range(Nky):
                for iz in range(Nkz):
                    k = Kgr[ix,iy,iz]
                    Hk = self.get_Hk(k)
                    ens = np.linalg.eigh(Hk)[0]
                    ens1 = ens*(ens < Ef)
                    N1 = np.sum(ens1)
                    charge += N1 * gK
        return charge
        
        

        
 

