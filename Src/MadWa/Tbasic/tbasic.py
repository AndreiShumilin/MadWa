# Import libraries
#The objective is to build an object to perform basic calculations from Wannier90 files. Provide a type of postprocessing tool that would be compatible with Andrei's work
# file: tbasic.py

import numpy as np

import numba as nb
from ..wannierIO import *
from .tbroutines import *

__all__ = ['TBasic']

class TBasic:
     # This class only works for a spin-specie of Hamiltonian with spin orbit coupling. For Separated spin channel calculations ask Andrei :D
    """
    Basic Tight-Binding container for Wannier90 Hamiltonians.
    """
    def __init__(self, hr_file = None, win_file=None):
        self.hr_file = hr_file
        self.win_file = win_file
        
        # Initialize placeholders (populated by read_file)
        self.H_ij = None
        self.rvects = None
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
        
        # Automatically load the data when the object is created
        if not ( (self.hr_file is None) or (self.win_file is None)):
            self.read_file()

  
    #--------------------------------------------------------------------------------#
    #HR Hamiltonian and its attributes
    #--------------------------------------------------------------------------------#
    def read_file(self, hr_file = None, win_file=None):
        if not hr_file is None:
            self.hr_file = hr_file
        if not win_file is None:
            self.win_file = win_file
        H_ij, rvects, deg, Par = readHR(self.hr_file)
        self.H_ij = H_ij
        self.rvects = rvects.astype(np.float64)
        self.Irvects = rvects.astype(np.int32)   ### Integer version or rvects, lets keep it
        self.deg = deg
        self.num_wann = Par[0]
        self.num_rvec = Par[1]


        win_data = readWin(self.win_file)
        self.cell = win_data['cell'].astype(np.float64)
        self.kpts = win_data['kpts'] 
        self.kpath = win_data['kpath'] 
        
 
        self.recip_cell = self._get_reciprocal_lattice()
        self.ready = True

    def manual(self, H_ij, rvects, deg, cell=np.eye(3), kpts=None, kpath=None):
        r"""
        Manualy initiates the object from the user-provided information
        """
        self.H_ij = H_ij.astype(np.complex128)
        self.rvects = rvects.astype(np.float64)
        self.Irvects = rvects.astype(np.int32)
        self.deg = deg
        self.num_wann = H_ij.shape[1]
        self.num_rvec = len(rvects)

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
        if not self.ready:
            return None
            
        return Hk_njit(self.H_ij, self.rvects, self.deg, kvec, self.cell)

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



    def readProjections(self, seedname):
        infile = seedname + '.win'
        amnfile = seedname + '.amn'
        Ufile = seedname + '_u.mat'
        UDISfile = seedname + '_u_dis.mat'
        eigfile = seedname + '.eig'
        
        winD = read.readWin(infile)
        
        self.atoms = winD['atoms']
        

        
 

