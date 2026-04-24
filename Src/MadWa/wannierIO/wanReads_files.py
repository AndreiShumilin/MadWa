import numpy as np
from .wanReads_win import readWin


__all__ = ['readWin','readAmn','readU','readEig','readXYZ','readHR','read_tb_file']

def readAmn(fileAmn):
    r"""
    Read Amn matrix from .amn files of Wannier90
    returns the matrix and the set of (Nb, Nk, Nw)
    Nb - number of Kohn-Sham bands
    Nk - number of k-poiints
    Nw - number of Wannier functions
    """
    with open(fileAmn, 'r') as f:
        lines = f.readlines()

    ln1 = lines[1].split()
    if len(ln1)<3:
        print('readAnm : incorrect line 1')
        return None

    Nb = int(ln1[0])
    Nk = int(ln1[1])
    Nw = int(ln1[2])
    #print(Nb, Nk, Nw)

    Amn = np.zeros((Nk, Nb, Nw), dtype=np.complex128)
    for ili,line in enumerate(lines[2:]):
        lns = line.split()
        if len(lns) != 5:
            print('readAnm : incorrect line ', str(2+ili))
            continue
        ib = int(lns[0])-1
        ik = int(lns[2])-1
        iw = int(lns[1])-1
        Re = float(lns[3])
        Im = float(lns[4])
        val = Re + 1j*Im
        Amn[ik, ib, iw] = val
    return Amn, (Nb, Nk, Nw)


def readU(fileU):
    r"""
    reads files of .mat format from wannier90, such as
    
    seedname_u.mat
    seedname_u_dis.mat
    
    Result is: (U, kar)
    U : Nkpt x N1 x N2 matrix, where
    Nkpt - num number of k-points
    N1, N2 - matrix dimensionality

    kar : Nkptx3 array of k-points in relative (Direct) representation
    """
    with open(fileU, 'r') as f:
        lines = f.readlines()
    ln = lines[1].split()
    Nkpt, N1, N2 = int(ln[0]), int(ln[1]), int(ln[2])
    iln = 3
    kar = np.zeros((Nkpt,3), dtype=np.float64)
    Umat = np.zeros((Nkpt, N2, N1), dtype=np.complex128)
    for ik in range(Nkpt):
        ln = lines[iln].split()
        kar[ik] = float(ln[0]), float(ln[1]), float(ln[2])
        U1 = np.loadtxt(lines[iln+1: iln+1+N1*N2], max_rows=(N1 * N2)).view(complex).reshape((N2, N1), order='F')
        Umat[ik] = U1
        iln += N1*N2 + 2
    return Umat, kar

def readEig(fileEig, nkpts, nb):
    r"""
    reads files of .eig format from wannier90 (fileEig)
    should be provided with number of k-points nkpt
    and number of Kohn-Sham bands nb
    """
    with open(fileEig, 'r') as f:
        lines = f.readlines()

    eigs = np.zeros((nkpts, nb), dtype=np.float64)
    for ln in lines:
        lns = ln.split()
        if len(lns) >= 3:
            ib = int(lns[0])-1
            ik = int(lns[1])-1
            en = float(lns[2])
            eigs[ik,ib] = en
    return eigs


def readXYZ(filexyz):
    with open(filexyz, 'r') as f:
        lines = f.readlines()
    Nw = int(lines[0])
    cen = np.loadtxt(lines[2:2+Nw], usecols=[1,2,3], dtype=np.float64)
    return cen


def readHR(hr_file): 
    r"""
    reads files of .hr format from wannier90 (fileEig)
    rturns:

    H : Hamiltonian, [num_rvec x num_wann x num_wann] complex matrix
    rvecs : [num_rvec x 3] int matrix
    deg - list of degeneracies
    pars = (num_wann, num_rvec)
    num_wann - number of Wannier functions
    num_rvec - number of r-vectros
    """
    with open(hr_file, 'r') as f:
            lines = f.readlines()
    num_wann = int(lines[1]) #Number of wannier functions
    num_rvec = int(lines[2]) #number of R vectors
    size_deg = int(np.ceil(num_rvec/15)) 
    deg = []
    for i in range(size_deg):
        deg.extend(list(map(int,lines[i+3].split()))) # Degeneracies
    dim = num_wann
    rvects = np.zeros((num_rvec,3), dtype =int) #Note that the R vector is reapeted num_wann**2 times in the file. This can be improve by having just the non-   equal R vectors 
    #indexes = np.zeros((num_rvec*(num_wann**2),2)) # i and j indexes
    H_ij = np.zeros((num_rvec, dim, dim), dtype=np.complex128) # matrix elements Hamiltonian
    start = 3 + size_deg
    idx = 0
    #------------ Adding the elements to each array -------------------------
    for r in range(num_rvec):
        for n in range(num_wann**2):
            line=lines[start+idx].split()
            R = list(map(int,line[:3]))
            rvects[r] = R
            i = int(line[3]) - 1
            j = int(line[4]) - 1
            re = float(line[5])
            im = float(line[6])
            H_ij[r, i, j] = re + 1j * im
            idx += 1
    return H_ij, rvects, deg, (num_wann, num_rvec)



def read_tb_file(tb_file):
    """
    Read the tb.dat file with or without soc
    tb_file = seedname_tb.dat file
    Returns the same as the latter function plus the r_mat that are the projections for all the rvects
    """
    with open(tb_file, 'r') as f:
        # Create an iterator to pull lines one by one
        lines = (line for line in f)
        
        # Skip header and lattice vectors (4 lines)
        for _ in range(4): next(lines)
        
        num_wann = int(next(lines).strip())
        num_rvec = int(next(lines).strip())
        
        # Read degeneracies (handle multiple lines)
        deg = []
        while len(deg) < num_rvec:
            deg.extend(map(int, next(lines).split()))
            
        rvects = np.zeros((num_rvec, 3), dtype=int)
        h_ij = np.zeros((num_rvec, num_wann, num_wann), dtype=np.complex128)
        # Position matrix elements: r_mat[R_index, m, n, direction(x,y,z)]
        r_mat = np.zeros((num_rvec, num_wann, num_wann, 3), dtype=np.complex128)

        # --- Read Hamiltonian Block ---
        for r_idx in range(num_rvec):
            # Skip any leading blank lines before the R-vector
            line = next(lines).strip()
            while not line:
                line = next(lines).strip()
            
            # Now we are at the R-vector line
            rvects[r_idx] = list(map(int, line.split()))
            
            for _ in range(num_wann**2):
                data = next(lines).split()
                i, j = int(data[0]) - 1, int(data[1]) - 1
                h_ij[r_idx, i, j] = float(data[2]) + 1j * float(data[3])

        # --- Read Position Matrix Block ---
        # The position block follows the Hamiltonian block
        for r_idx in range(num_rvec):
            # Skip blanks before the R-vector in the position block
            try:
                line = next(lines).strip()
                while not line:
                    line = next(lines).strip()
            except StopIteration:
                break # End of file reached
                
            # The R-vector here should match the one in the Ham block
            # Logic: Read N_wann**2 lines, each containing 6 floats (Re/Im for X, Y, Z)
            for _ in range(num_wann**2):
                data = next(lines).split()
                i, j = int(data[0]) - 1, int(data[1]) - 1
                # X projection
                r_mat[r_idx, i, j, 0] = float(data[2]) + 1j * float(data[3])
                # Y projection
                r_mat[r_idx, i, j, 1] = float(data[4]) + 1j * float(data[5])
                # Z projection
                r_mat[r_idx, i, j, 2] = float(data[6]) + 1j * float(data[7])
                
    return rvects, h_ij, r_mat, deg, (num_wann, num_rvec)

