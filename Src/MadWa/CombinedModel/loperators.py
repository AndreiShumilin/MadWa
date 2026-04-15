"""
loperators.py

Module for defining angular momentum operators and computing L operators for spin-orbit coupling (SOC).
"""

import numpy as np
import typing

# Angular momentum operator definitions
l_sizes_dict = {0: 1, 1: 3, 2: 5, 3: 7}
l_goodlist = [0, 1, 2, 3]

s = 1 / np.sqrt(2)
Tl1 = np.array([[0, 1j * s, s], [-1j, 0, 0], [0, -1j * s, s]])
Tl2 = np.array([[0, 0, 0, -s, 1j * s], [0, s, -1j * s, 0, 0], [-1, 0, 0, 0, 0],
                [0, -s, -1j * s, 0, 0], [0, 0, 0, -s, -1j * s]])
Tl3 = np.array([[0, 0, 0, 0, 0, -1j * s, -s], [0, 0, 0, 1j * s, s, 0, 0],
                [0, -1j * s, -s, 0, 0, 0, 0], [1j, 0, 0, 0, 0, 0, 0],
                [0, 1j * s, -s, 0, 0, 0, 0], [0, 0, 0, 1j * s, -s, 0, 0],
                [0, 0, 0, 0, 0, 1j * s, -s]])

FirstTransform = {0: np.array([[1]]), 1: Tl1, 2: Tl2, 3: Tl3}

def Lz00(l: int) -> np.ndarray:
    """
    Compute the Lz operator for angular momentum l.

    Args:
        l (int): Angular momentum quantum number.

    Returns:
        np.ndarray: Lz operator matrix.
    """
    N = 2 * l + 1
    a = np.zeros((N, N), dtype=np.complex128)
    for im, m in enumerate(range(l, -l - 1, -1)):
        a[im, im] = m
    return a

def Lx00(l: int) -> np.ndarray:
    """
    Compute the Lx operator for angular momentum l.

    Args:
        l (int): Angular momentum quantum number.

    Returns:
        np.ndarray: Lx operator matrix.
    """
    N = 2 * l + 1
    def im(m): return l - m
    a = np.zeros((N, N), dtype=np.complex128)
    for m in range(l, -l, -1):
        a[im(m), im(m - 1)] = 0.5 * np.sqrt((l + m) * (l - m + 1))
        a[im(m - 1), im(m)] = 0.5 * np.sqrt((l + m) * (l - m + 1))
    return a

def Ly00(l: int) -> np.ndarray:
    """
    Compute the Ly operator for angular momentum l.

    Args:
        l (int): Angular momentum quantum number.

    Returns:
        np.ndarray: Ly operator matrix.
    """
    N = 2 * l + 1
    def im(m): return l - m
    a = np.zeros((N, N), dtype=np.complex128)
    for m in range(l, -l, -1):
        a[im(m), im(m - 1)] = (-0.5j) * np.sqrt((l + m) * (l - m + 1))
        a[im(m - 1), im(m)] = (0.5j) * np.sqrt((l + m) * (l - m + 1))
    return a

def hermconj(A: np.ndarray) -> np.ndarray:
    """
    Compute the Hermitian conjugate of a matrix.

    Args:
        A (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Hermitian conjugate of the matrix.
    """
    return np.conjugate(np.transpose(A))

def comparePos(pos1: np.ndarray, pos2: np.ndarray, dr: float = 1e-4) -> bool:
    """
    Compare two positions within a tolerance.

    Args:
        pos1 (np.ndarray): First position vector.
        pos2 (np.ndarray): Second position vector.
        dr (float): Tolerance for comparison.

    Returns:
        bool: True if positions are within tolerance, False otherwise.
    """
    dpos = np.sqrt(np.sum((pos1 - pos2) ** 2))
    return dpos < dr

class Block:
    """
    Class to handle angular momentum blocks for SOC calculations.
    """
    def __init__(self, atom_name: str, l: int, position: np.ndarray, axis_x: np.ndarray, axis_z: np.ndarray):
        self.atom_name = atom_name
        self.l = l
        self.position = position
        self.axis_x = axis_x
        self.axis_z = axis_z
        self.axis_y = np.cross(self.axis_z, self.axis_x)
        self.N = l_sizes_dict[self.l]
        self.psiN = np.zeros(self.N, dtype=int) - 1000
        self.complete = False

    def addPsi(self, ipsi: int, mr: int):
        """
        Add a wavefunction index to the block.

        Args:
            ipsi (int): Index of the wavefunction.
            mr (int): Magnetic quantum number.
        """
        imr = mr - 1
        self.psiN[imr] = ipsi
        if np.min(self.psiN) >= 0:
            self.complete = True

    def calcL(self):
        """
        Calculate the Lx, Ly, and Lz operators for the block.
        """
        if not self.complete:
            print(f'L-operators: Warning! Missing wavefunctions for block {self.atom_name}')
            return
        if self.l == 0:
            L0 = np.zeros((1, 1), dtype=np.complex128)
            self.Lx = self.Ly = self.Lz = L0
        elif self.l in l_goodlist:
            T = FirstTransform[self.l]
            Lx1 = hermconj(T) @ Lx00(self.l) @ T
            Ly1 = hermconj(T) @ Ly00(self.l) @ T
            Lz1 = hermconj(T) @ Lz00(self.l) @ T
            self.Lx = (self.axis_x[0] * Lx1 + self.axis_y[0] * Ly1 + self.axis_z[0] * Lz1)
            self.Ly = (self.axis_x[1] * Lx1 + self.axis_y[1] * Ly1 + self.axis_z[1] * Lz1)
            self.Lz = (self.axis_x[2] * Lx1 + self.axis_y[2] * Ly1 + self.axis_z[2] * Lz1)
        else:
            print(f'L-operators: Warning! Unknown basis l={self.l} in block {self.atom_name}')
            self.Lx = self.Ly = self.Lz = np.zeros((self.N, self.N), dtype=np.complex128)

def makeblocks(proj: typing.List[typing.Dict]) -> typing.List[Block]:
    """
    Create angular momentum blocks from Wannier projections.

    Args:
        proj (List[Dict]): List of projection dictionaries from Wannier90 .wout file.

    Returns:
        List[Block]: List of Block objects containing L operators.
    """
    Lblocks = []
    for iprj, prj in enumerate(proj):
        pos = prj['center']
        l = prj['l']
        mr = prj['mr']
        found = False
        for i, blk in enumerate(Lblocks):
            if comparePos(pos, blk.position) and l == blk.l:
                found = True
                Lblocks[i].addPsi(iprj, mr)
                break
        if not found:
            if 'atom' in prj.keys():
                name = prj['atom']
            else:
                name = f'Atom'
            blk_new = Block(name, l, pos, prj['x-axis'], prj['z-axis'])
            blk_new.addPsi(iprj, mr)
            Lblocks.append(blk_new)
    for bl in Lblocks:
        bl.calcL()
    return Lblocks

def L_BlocksToMatr(Nf: int, lst: typing.List[Block], xiDict) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert block L operators to full matrices.

    Args:
        Nf (int): Total number of Wannier functions.
        lst (List[Block]): List of Block objects.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Lx, Ly, Lz matrices.
    """
    Lx = np.zeros((Nf, Nf), dtype=np.complex128)
    Ly = np.zeros((Nf, Nf), dtype=np.complex128)
    Lz = np.zeros((Nf, Nf), dtype=np.complex128)
    for blk in lst:
        if not hasattr(blk, 'Lx'):
            continue
        aname = blk.atom_name
        info = (aname, blk.l, blk.l)
        if info in xiDict.keys():
            power = xiDict[info]
        else:
            power = 1
            
        for i in range(blk.N):
            for j in range(blk.N):
                i1 = blk.psiN[i]
                i2 = blk.psiN[j]
                Lx[i1, i2] = blk.Lx[i, j] * power
                Ly[i1, i2] = blk.Ly[i, j] * power
                Lz[i1, i2] = blk.Lz[i, j] * power
    return Lx, Ly, Lz