"""
reads.py

Module for reading and parsing input files (Wannier90, VASP, etc.) for Hamiltonian calculations.
"""

import numpy as np
import pymatgen.electronic_structure.core as pmg_spin
from pymatgen.io.vasp import Vasprun, Kpoints
import wannier90io as w90io
import typing
import os

def read_u0(stream: typing.TextIO) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Read U matrices from a Wannier90 .mat file.

    Args:
        stream (TextIO): File stream of the .mat file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: k-points and U matrices.
    """
    stream.readline()  # Skip header
    nkpt, num_wann, num_bands = np.fromstring(stream.readline(), sep=' ', dtype=int)
    u_matrices = np.zeros((nkpt, num_bands, num_wann), dtype=complex)
    kpoints = []
    for ikpt in range(nkpt):
        empty = stream.readline()
        assert not empty.strip(), f"Expected empty line but found: '{empty}'"
        kpoint = np.fromstring(stream.readline(), sep=' ', dtype=float)
        assert len(kpoint) == 3
        kpoints.append(kpoint)
        u_matrices[ikpt, :, :] = np.loadtxt(stream, max_rows=(num_wann * num_bands)).view(complex).reshape((num_bands, num_wann), order='F')
    return np.array(kpoints), u_matrices

def read_u(fname: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to read U matrices from a file.

    Args:
        fname (str): Path to the .mat file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: k-points and U matrices.
    """
    with open(fname, "r") as f:
        return read_u0(f)


def read_hr(filename: str):
    """
    Read Wannier90 *_hr.dat, including ndegen, and return a symmetrized HR set:
      - hr_data.r_vectors: list of all R present after completion
      - hr_data.hr_matrix: array [nR, num_wann, num_wann] with H(R)/ndegen[R]
    """
    import numpy as np, os
    with open(filename, 'r') as f:
        # keep blank lines; we'll parse by position according to format
        raw = f.readlines()

    # 1: header (title), 2: num_wann, 3: nrpts
    title = raw[0].rstrip("\n")
    num_wann = int(raw[1].strip())
    nrpts = int(raw[2].strip())

    # next lines contain nrpts integers, we skip them
    degens = 0
    idx = 3
    while (degens < nrpts):
        ar1 = [int(x) for x in raw[idx].split()]
        degens += len(ar1)
        idx += 1
    # Now idx points to the first data line after ndegen list

    # Data lines: one line per matrix element:
    # R1 R2 R3  i  j  Re  Im   (1-based i,j)
    Hdict = {}              # {R: num_wann x num_wann complex}
    Rvecs = []

    valid_lines = 0
    for line in raw[idx:]:
        t = line.split()
        if len(t) != 7:
            print("Warning: incorrect line", t)
            # ignore empty or malformed lines silently
            continue
        R = (int(t[0]), int(t[1]), int(t[2]))
        i = int(t[3]) - 1
        j = int(t[4]) - 1
        re = float(t[5]); im = float(t[6])

        if R not in Hdict:
            Hdict[R] = np.zeros((num_wann, num_wann), dtype=np.complex128)
            Rvecs.append(R)
        Hdict[R][i, j] = re + 1j * im
        valid_lines += 1

    #print(f"Processed {valid_lines} valid HR data lines in {os.path.basename(filename)}")
    if not len(Rvecs) == nrpts:
        print('Warning : ', nrpts, ' r-vectors expected, ', len(Rvecs), ' was found')

    # Pack into hr_data object
    class HRData: pass
    hr_data = HRData()
    hr_data.r_vectors = Rvecs
    hr_data.hr_matrix = np.array([Hdict[R] for R in Rvecs], dtype=np.complex128)
    return hr_data, num_wann


def old_read_hr(filename: str) -> typing.Tuple[typing.Any, int]:
    """
    Read real-space Hamiltonian from Wannier90 _hr.dat file.

    Args:
        filename (str): Path to the _hr.dat file.

    Returns:
        Tuple[Any, int]: HRData object containing R-vectors and Hamiltonian matrices, and number of Wannier functions.
    """
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
#     print(f"First 10 lines of {filename}:")
# #    for i, line in enumerate(lines[:10]):
#        print(f"Line {i+1}: {line}")
    num_wann = int(lines[1])
    num_R = int(lines[2])
    data_lines = lines[3 + num_R:]
    Rvecs, Hdict = [], {}
    valid_lines = 0
    outlier_lines = []
    for idx, line in enumerate(data_lines):
        tokens = line.split()
        if len(tokens) != 7:
            print(f"[WARN] Skipping malformed HR line {idx + 4 + num_R}: {line}")
            continue
        try:
            R = tuple(map(int, tokens[0:3]))
            i, j = int(tokens[3]) - 1, int(tokens[4]) - 1
            re, im = float(tokens[5]), float(tokens[6])
            if i < 0 or j < 0 or i >= num_wann or j >= num_wann:
                print(f"[WARN] Invalid indices in line {idx + 4 + num_R}: {line}")
                continue
            if R == (-4, -4, 0):
                outlier_lines.append((idx + 4 + num_R, line))
            if R not in Hdict:
                Hdict[R] = np.zeros((num_wann, num_wann), dtype=complex)
                Rvecs.append(R)
            Hdict[R][i, j] = re + 1j * im
            valid_lines += 1
        except (ValueError, IndexError):
            print(f"[WARN] Skipping malformed HR line {idx + 4 + num_R}: {line}")
            continue
    print(f"Processed {valid_lines} valid HR data lines in {filename}")
    if outlier_lines and "wannier90.1_hr.dat" in filename:
        print(f"Lines for R = (-4, -4, 0) in {filename}:")
        for line_num, line in outlier_lines:
            print(f"Line {line_num}: {line}")
    class HRData: pass
    hr_data = HRData()
    hr_data.r_vectors = Rvecs
    hr_data.hr_matrix = np.array([Hdict[R] for R in Rvecs])
    return hr_data, num_wann

def read_Rlist_from_hr(hr_files: typing.List[str], required_Rs: list = [(0,0,0), (1,0,0)]) -> typing.List[tuple]:
    """
    Extract R-vectors from HR files.

    Args:
        hr_files (List[str]): List of paths to _hr.dat files.
        required_Rs (list): List of required R-vectors.

    Returns:
        List[tuple]: Sorted list of R-vectors.
    """
    Rset = set(required_Rs)
    for hr_file in hr_files:
        if os.path.exists(hr_file):
            with open(hr_file, 'r') as f:
                while True:
                    line = f.readline()
                    if line.strip().isdigit():
                        num_wann = int(line.strip())
                        break
                num_Rpts = int(f.readline())
                for _ in range(num_Rpts):
                    f.readline()
                for _ in range(num_Rpts * num_wann * num_wann):
                    line = f.readline().split()
                    if len(line) >= 3:
                        R = tuple(map(int, line[:3]))
                        Rset.add(R)
        else:
            print(f"⚠️ {hr_file} not found. Using only required R vectors: {required_Rs}")
    return sorted(Rset)

def parse_amn(filename: str = "wannier90.1.amn", nk_cut: int = None) -> typing.Tuple[typing.List[np.ndarray], int]:
    """
    Parse AMN file for Wannier projections.

    Args:
        filename (str): Path to the AMN file.
        nk_cut (int, optional): Number of k-points to parse.

    Returns:
        Tuple[List[np.ndarray], int]: List of A(k) matrices and number of bands.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    entries = []
    max_i = max_j = max_k = -1
    for line in lines:
        toks = line.strip().split()
        if len(toks) != 5:
            continue
        i, j, k = int(toks[0]) - 1, int(toks[1]) - 1, int(toks[2]) - 1
        re, im = float(toks[3]), float(toks[4])
        entries.append((i, j, k, re + 1j * im))
        max_i = max(max_i, i)
        max_j = max(max_j, j)
        max_k = max(max_k, k)
    nb, nw, nk_full = max_i + 1, max_j + 1, max_k + 1
    nk = nk_cut if nk_cut else nk_full
    A_k_list = [np.zeros((nb, nw), dtype=complex) for _ in range(nk)]
    for i, j, k, val in entries:
        if k < nk:
            A_k_list[k][i, j] = val
    print(f"✅ Parsed AMN: nb={nb}, nw={nw}, nk_used={nk}")
    return A_k_list, nb

def read_out(fname: str) -> typing.List[typing.Dict]:
    """
    Read Wannier projection data from .wout file.

    Args:
        fname (str): Path to the .wout file.

    Returns:
        List[Dict]: List of projection dictionaries.
    """
    def extract_wannier_projections_out(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        proj_index = next((i for i, line in enumerate(lines) if "PROJECTIONS" in line.upper()), None)
        if proj_index is None:
            raise ValueError("PROJECTIONS section not found.")
        border1_index = next((j for j in range(proj_index, len(lines)) if lines[j].strip().startswith("+")), None)
        if border1_index is None:
            raise ValueError("First border line not found after PROJECTIONS.")
        header_index = border1_index + 1
        border2_index = header_index + 1
        if border2_index >= len(lines) or not lines[border2_index].strip().startswith("+"):
            raise ValueError("Second border (after header) not found.")
        start_data = border2_index + 1
        end_data = next((k for k in range(start_data, len(lines)) if lines[k].strip().startswith("+")), len(lines))
        data_lines = [
            line.strip().strip("|").strip()
            for line in lines[start_data:end_data]
            if line.strip() and line.strip().startswith("|")
        ]
        return data_lines

    dlines = extract_wannier_projections_out(fname)
    projections = []
    for lin in dlines:
        prj = {}
        data = lin.split()
        prj['center'] = np.array([float(data[0]), float(data[1]), float(data[2])])
        prj['l'] = int(data[3])
        prj['mr'] = int(data[4])
        prj['r'] = int(data[5])
        prj['z-axis'] = np.array([float(data[6]), float(data[7]), float(data[8])])
        prj['x-axis'] = np.array([float(data[9]), float(data[10]), float(data[11])])
        prj['zona'] = float(data[12])
        projections.append(prj)
    return projections

def shifttomin(v: np.ndarray, emin: float) -> np.ndarray:
    """
    Shift eigenvalues to align with minimum energy.

    Args:
        v (np.ndarray): Eigenvalues.
        emin (float): Minimum energy threshold.

    Returns:
        np.ndarray: Shifted eigenvalues.
    """
    N = len(v)
    n1 = 0
    for i in range(N):
        if v[i] < emin:
            n1 += 1
        else:
            break
    return np.roll(v, -n1)

def findVec(A: np.ndarray, v: np.ndarray, crit: float = 1e-4) -> int:
    """
    Find index of vector v in array A within tolerance.

    Args:
        A (np.ndarray): Array of vectors.
        v (np.ndarray): Target vector.
        crit (float): Tolerance for comparison.

    Returns:
        int: Index of the vector, or -1 if not found.
    """
    shA = A.shape
    N = shA[0]
    for i in range(N):
        v1 = A[i]
        cr = np.max(np.abs(v1 - v))
        if cr < crit:
            return i
    return -1


def read_centers(file, Nw = None):
    with open(file, 'r') as f:
        num1 = int(f.readline())
        f.readline()

        colines = []
        if not Nw is None:
            for i in range(Nw):
                line1 = f.readline()
                line2 = line1.split()
                colines.append(line2)
        else:
            Nw = 0
            line1 = f.readline()
            line2 = line1.split()
            while line2[0] == 'X':
                Nw += 1
                colines.append(line2)
                line1 = f.readline()
                line2 = line1.split()
    coords = np.zeros((Nw, 3))
    for i in range(Nw):
        coords[i,0] = float(colines[i][1])
        coords[i,1] = float(colines[i][2])
        coords[i,2] = float(colines[i][3])
    return coords


def read_centers_atoms(file):
    at_names = []
    at_centers = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for ln in lines[2:]:
            ln2 = ln.split()
            if ln2[0] != 'X':
                at_names.append( ln2[0] )
                coo = np.zeros(3)
                coo[0] = float(ln2[1])
                coo[1] = float(ln2[2])
                coo[2] = float(ln2[3])
                at_centers.append( coo )
    return at_names, at_centers

def build_Hk_from_vasprun2(xml_file: str, wan_in: str, ibzkp: str, emin: float = -9) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray], typing.List[np.ndarray], int]:
    """
    Build k-space Hamiltonians from VASP vasprun.xml.

    Args:
        xml_file (str): Path to vasprun.xml.
        wan_in (str): Path to Wannier90 .win file.
        ibzkp (str): Path to IBZKPT file.
        emin (float): Minimum energy for eigenvalue shifting.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], int]: Spin-up, spin-down, and non-magnetic Hamiltonians, and number of k-points.
    """
    vrun = Vasprun(xml_file, parse_projected_eigen=True)
    eigvals_up = vrun.eigenvalues[pmg_spin.Spin.up]
    eigvals_down = vrun.eigenvalues[pmg_spin.Spin.down]
    nk = eigvals_up.shape[0]
    nb = eigvals_up.shape[1]
    assert eigvals_up.shape == eigvals_down.shape, "Spin-up and spin-down eigenvalue shapes mismatch"
    
    Hk_up_list, Hk_down_list, Hk0_list = [], [], []
    for ik in range(nk):
        evals_up = shifttomin(eigvals_up[ik, :, 0], emin)  #### should not be in the final version
        evals_down = shifttomin(eigvals_down[ik, :, 0], emin)
        Hk_up = np.diag(evals_up.astype(complex))
        Hk_down = np.diag(evals_down.astype(complex))
        Hk0 = (Hk_up + Hk_down) / 2.0
        Hk_up_list.append(Hk_up)
        Hk_down_list.append(Hk_down)
        Hk0_list.append(Hk0)
    
    vasp_kp0 = Kpoints.from_file(ibzkp)
    vasp_kp = np.array(vasp_kp0.kpts)
    with open(wan_in, 'r') as fh:
        parsed_win = w90io.parse_win_raw(fh.read())
    wan_kp = np.array(parsed_win['kpoints']['kpoints'])
    
    Hkw_up_list, Hkw_down_list, Hkw0_list = [], [], []
    for i in range(wan_kp.shape[0]):      #   modification 25 kpoints in VASP ------ 64 k-points of wannier
        iVasp = findVec(vasp_kp, np.abs(wan_kp[i]))   # !!! Should not be used in the final version !!!!!!!!!
        Hkw_up_list.append(Hk_up_list[iVasp])
        Hkw_down_list.append(Hk_down_list[iVasp])
        Hkw0_list.append(Hk0_list[iVasp])
    
    return Hkw_up_list, Hkw_down_list, Hkw0_list, wan_kp.shape[0]