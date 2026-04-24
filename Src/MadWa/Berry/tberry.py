import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import numba as nb
import MadWa.Tbasic.tbroutines as tbroutines

##################################################################
# This returns the Berry curvature in a specific kpoint
##################################################################
@nb.njit
def TBerry(H_ij, rvects, deg, kvec, cell, fermi_energy):
    dim = H_ij.shape[1]
    Hk = np.zeros((dim, dim), dtype=np.complex128)
    dH_dkx = np.zeros((dim, dim), dtype=np.complex128)
    dH_dky = np.zeros((dim, dim), dtype=np.complex128)
    dH_dkz = np.zeros((dim, dim), dtype=np.complex128)
    # Fourier Transform for H and its Derivatives
    for i in range(len(rvects)):
        R = rvects[i]
        rvreal = R[0]*cell[0] + R[1]*cell[1] + R[2]*cell[2]      
        phase = np.exp(1j * np.dot(kvec, rvreal))
        term = (H_ij[i] / deg[i]) * phase
        Hk += term
        # Analytical derivative: d/dk [exp(ikR)] = iR * exp(ikR)
        dH_dkx += 1j * rvreal[0] * term
        dH_dky += 1j * rvreal[1] * term
        dH_dkz += 1j * rvreal[2] * term
    E, U = la.eigh(Hk) # U columns are eigenvectors |n>
    # This calculates the matrix elements <n|dH/dk|m> 
    v_x = U.conj().T @ dH_dkx @ U
    v_y = U.conj().T @ dH_dky @ U
    v_z = U.conj().T @ dH_dkz @ U
    total_omg_xy = 0.0
    total_omg_xz = 0.0
    total_omg_yz = 0.0

    Omega_matrix_xy = np.zeros((dim,dim), dtype=np.complex128)
    Omega_matrix_xz = np.zeros((dim,dim), dtype=np.complex128)
    Omega_matrix_yz = np.zeros((dim,dim), dtype=np.complex128)

    for n in range(dim):
        if E[n] > fermi_energy: 
            continue # Only calculate for occupied bands  
        for m in range(dim):
            if n == m: 
                continue        
            dE = E[n] - E[m]
            if np.abs(dE) < 1e-14: 
                continue # Avoid division by zero
            term_xy = (v_x[n, m] * v_y[m, n] - v_y[n, m] * v_x[m, n])
            term_xz = (v_x[n, m] * v_z[m, n] - v_z[n, m] * v_x[m, n])
            term_yz = (v_y[n, m] * v_z[m, n] - v_z[n, m] * v_y[m, n])
            total_omg_xy += (1j * term_xy).real / (dE**2)
            total_omg_xz += (1j * term_xz).real / (dE**2)
            total_omg_yz += (1j * term_yz).real / (dE**2)

    return E, total_omg_xy, total_omg_xz, total_omg_yz
    
###############################################################
# This is a normalization function that is usefull for the 2D map
#################################################################
def apply_custom_log(data):
    # Mask for values with magnitude > 10
    mask = np.abs(data) > 10
    #transformed = 0
    transformed = np.zeros_like(data)
    transformed[mask] = np.sign(data[mask]) * np.log10(np.abs(data[mask]))
    transformed[~mask] = data[~mask] / 10.0
    return transformed

##################################################################
# This returns the Berry curvature in a 2Dkmesh
##################################################################
@nb.njit
def TBerry_2D(H_ij,rvects,deg, cell, corner, b1, b2, k_mesh, fermi_energy, recip_cell):

    dim = H_ij.shape[1]
    kx = k_mesh[0]
    ky = k_mesh[1] 
    # Initialize the grid for the heatmap
    b1 = tbroutines.fractional_to_cartesian(b1, recip_cell)
    b2 = tbroutines.fractional_to_cartesian(b2, recip_cell)
    berry_map_xy = np.zeros((kx,ky), dtype=np.float64)
    berry_map_xz = np.zeros((kx,ky), dtype=np.float64)
    berry_map_yz = np.zeros((kx,ky), dtype=np.float64)
    band_grid = np.zeros((dim, kx, ky), dtype = np.float64 )
    s1 = np.linspace(0, 1, kx)
    s2 = np.linspace(0, 1, ky)
    omg_xy = 0j
    omg_xz = 0j
    omg_yz = 0j 
    for i in range(kx):
        for j in range(ky):
            # Calculate the 3D k-point in crystal coordinates
            kvec_crystal = corner + s1[i] * b1 + s2[j] * b2

            E, omg_xy, omg_xz, omg_yz = TBerry(H_ij, rvects, deg, kvec_crystal, cell, fermi_energy)
            berry_map_xy[i,j] = omg_xy
            berry_map_xz[i,j] = omg_xz
            berry_map_yz[i,j] = omg_yz

    return band_grid,  berry_map_xy,  berry_map_xz,  berry_map_yz

def save_berry_plot(s1, s2, map_data, band_grid, fermi_energy, component_name):
    """
    Saves a specific Berry curvature component map with Fermi lines.
    component_name: string like 'xy', 'xz', or 'yz'
    """
    mn = int(np.floor(map_data.min()))
    mx = int(np.ceil(map_data.max()))
    ticks = range(mn, mx + 1) # +1 to include the max value

    plt.figure(figsize=(8, 7))

    cp = plt.contourf(s1, s2, map_data.T, levels=ticks, origin='lower', extend='both')

    # 3. Overlay Fermi Surface
    dim = band_grid.shape[0]
    for n in range(dim):
        if np.min(band_grid[n]) < fermi_energy < np.max(band_grid[n]):
            plt.contour(s1, s2, band_grid[n].T, levels=[fermi_energy], 
                        colors='black', linewidths=1.0, antialiased=True)

    ticklabels = []
    for n in ticks:
        if n < 0: 
            ticklabels.append(r'$-10^{%d}$' % abs(n))
        elif n == 0:
            ticklabels.append('0')
        else:
            ticklabels.append(r'$10^{%d}$' % n)

    cbar = plt.colorbar(cp, orientation='vertical')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label(rf'$\Omega_{{{component_name}}}$ (Log Scale)')

    plt.title(f'Berry Curvature $\Omega_{{{component_name}}}$ with Fermi Surface')
    plt.axis('off')
    filename = f"Omega_{component_name}_with_fermi_lines.pdf"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close() 


######################################################################
# This plots the Berry curvature in a 2D section defined by the user
######################################################################    
def berry_2Dmap(H_ij,rvects,deg, cell, corner, b1, b2, k_mesh, fermi_energy, recip_cell):
    
    kx = k_mesh[0]
    ky = k_mesh[1]
    s1 = np.linspace(0, 1, kx)
    s2 = np.linspace(0, 1, ky)
    band_grid, omg_xy_2D, omg_xz_2D, omg_yz_2D = TBerry_2D(H_ij,rvects,deg, cell, corner, b1, b2, k_mesh, fermi_energy, recip_cell)
    berry_map_xy = -apply_custom_log(omg_xy_2D)
    berry_map_xz = -apply_custom_log(omg_xz_2D)
    berry_map_yz = -apply_custom_log(omg_yz_2D)
    
    save_berry_plot(s1, s2,  berry_map_xy, band_grid, fermi_energy, 'xy')
    save_berry_plot(s1, s2,  berry_map_xz, band_grid, fermi_energy, 'xz')
    save_berry_plot(s1, s2,  berry_map_yz, band_grid, fermi_energy, 'yz')
    print('Saving 2D maps for the Berry curvature')
    header = f"{'Omega_yz':>19} {'Omega_xz':>19} {'Omega_xy':>19}"
    combined = np.column_stack((berry_map_yz.flatten(), berry_map_xz.flatten(), berry_map_xy.flatten()))
    np.savetxt('TBerry_curv_kslice.dat',combined, fmt='%20e', header = header )
    print('Data saved in TBerry_curv_kslice.dat file!')
    
@nb.njit
def AHC(H_ij, rvects, deg, cell, kmesh, fermi_energy, recip_cell,  Dim2D = False):
    V = tbroutines.cellVolume(cell, Dim2D)
    sigma_xy = 0
    sigma_xz = 0
    sigma_yz = 0
    recip_cell =np.ascontiguousarray(recip_cell)
    kvect = np.zeros((1,3), dtype=np.float64)
    ## Factor to get the conductivity in units of S/cm
    e2_h = 3.874e-5  # S (quantum of conductance)
    to_S_cm = - (e2_h / (2 * np.pi)) * 1e8
    if Dim2D:
        b1 = recip_cell[0]
        b2 = recip_cell[1]
        Nk = kmesh[0]* kmesh[1]
        dx = np.linspace(0,1,kmesh[0])
        dy = np.linspace(0,1,kmesh[1])
        for i in range(kmesh[0]):
            for j in range(kmesh[1]):
                kvect = b1*dx[i]+b2*dy[j]
                E, omega_xy, omega_xz, omega_yz = TBerry(H_ij, rvects, deg, kvect, cell, fermi_energy)
                sigma_xy +=  omega_xy
                sigma_xz +=  omega_xz
                sigma_yz +=  omega_yz
        sigma_xy = sigma_xy/(Nk*V)
        sigma_xz = sigma_xz/(Nk*V)
        sigma_yz = sigma_yz/(Nk*V)
    else:
        b1 = recip_cell[0]
        b2 = recip_cell[1]
        b3 = recip_cell[2]
        Nk = kmesh[0]* kmesh[1]*kmesh[2]
        dx = np.linspace(0,1,kmesh[0])
        dy = np.linspace(0,1,kmesh[1])
        dz = np.linspace(0,1,kmesh[2])
        for i in range(kmesh[0]):
            for j in range(kmesh[1]):
                for k in range(kmesh[2]):
                    kvect = b1*dx[i]+b2*dy[j]+b3*dz[k]
                    #Aqui la suma de todos los k
                    E, omega_xy, omega_xz, omega_yz = TBerry(H_ij, rvects, deg, kvect, cell, fermi_energy)
                    #E, omega_xy, omega_xz, omega_yz = TBerry2(H_ij, rvects, deg, kvect, cell)
                    sigma_xy +=  omega_xy
                    sigma_xz +=  omega_xz
                    sigma_yz +=  omega_yz
        sigma_xy = sigma_xy*to_S_cm/(Nk*V)
        sigma_xz = sigma_xz*to_S_cm/(Nk*V)
        sigma_yz = sigma_yz*to_S_cm/(Nk*V)
    return sigma_xy, sigma_xz, sigma_yz
    