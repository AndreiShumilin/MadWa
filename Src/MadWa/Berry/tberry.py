import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import numba as nb
import MadWa.Tbasic.tbroutines as tbroutines
import MadWa.Utils.grids as grids

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
            #berry_map_yz[i,j] = omg_yz
            berry_map_xy[i,j] = omg_xy
            band_grid[:,i,j]  = E
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

    #Overlay Fermi Surface
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
    #plt.axis('off')
    plt.show()
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
    #combined = np.column_stack((berry_map_yz.flatten(), berry_map_xz.flatten(), berry_map_xy.flatten()))
    combined = np.column_stack((omg_yz_2D.flatten(), omg_xz_2D.flatten(), omg_xy_2D.flatten()))
    
    np.savetxt('TBerry_curv_kslice.dat',combined, fmt='%20e', header = header )
    print('Data saved in TBerry_curv_kslice.dat file!')

@nb.njit
def berry_coarse(centers, H_ij, rvects, deg, cell, fermi_energy):
    
    n = len(centers)
    vect_omg_xy = np.zeros(n, dtype = np.float64)
    vect_omg_xz = np.zeros(n, dtype = np.float64)
    vect_omg_yz = np.zeros(n, dtype = np.float64)
    
    for c in range(n):
        kvect =  centers[c]
        _, omega_xy, omega_xz, omega_yz = TBerry(H_ij, rvects, deg, kvect, cell, fermi_energy)
        vect_omg_xy[c]=omega_xy
        vect_omg_xz[c]=omega_xz
        vect_omg_yz[c]=omega_yz

    return vect_omg_xy, vect_omg_xz, vect_omg_yz

@nb.njit
def AHC_ref(omg_xy, omg_yz, omg_xz, weights, cell, Dim2D = False):
    e2 = (1.602176634e-19)**2
    h_bar = 1.054571817e-34
    e2_h = e2/h_bar  # Siemens (S)
    to_S_cm =  -e2_h * 1e8 
    s_xy = 0.0
    s_yz = 0.0
    s_xz = 0.0

    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]
    
    if Dim2D:
  
        a12 = np.cross(a1,a2)
        Va = np.sqrt(np.sum(a12*a12))
    else:

        a12 = np.cross(a1,a2)
        Va = abs(np.dot(a12,a3))
    
    N= len(weights)
    print(f"total cells: {N}")

    for i in range(N):
        w = weights[i]
        s_xy += omg_xy[i]*w
        s_xz += omg_xz[i]*w
        s_yz += omg_yz[i]*w
        
    sigma_xy = s_xy*to_S_cm/Va
    sigma_xz = s_xz*to_S_cm/Va
    sigma_yz = s_yz*to_S_cm/Va

    return sigma_xy, sigma_xz, sigma_yz

def refine_mesh(Nkx, Nky, Nkz,H_ij, rvects, deg, cell, fermi_energy, tol, nx,ny,nz, Dim2D = False, max_level=2):
    centers, vertices, weights = grids.coarse_kgrid_cells(Nkx, Nky, Nkz,  cell, Dim2D)
    vect_omg_xy, vect_omg_xz, vect_omg_yz = berry_coarse(centers, H_ij, rvects, deg, cell, fermi_energy)
    # Track refinement level per cell
    levels = np.zeros(len(weights), dtype=np.int64)

    for level in range(max_level):
        n_cells = len(weights)
        refine_ids = []
        # ERROR ESTIMATOR
        for i in range(n_cells):

            local_error = max(abs(vect_omg_xy[i]), abs(vect_omg_xz[i]), abs(vect_omg_yz[i]))
            if local_error > tol:
                refine_ids.append(i)

                #print(local_error)
        # STOP CONDITION
        if len(refine_ids) == 0:

            print(f"Converged at level {level}")
            break

        keep_mask = np.ones(n_cells, dtype=bool)

        for idx in refine_ids:
            keep_mask[idx] = False

        new_centers = list(centers[keep_mask])
        new_vertices = list(vertices[keep_mask])
        new_weights = list(weights[keep_mask])

        new_omg_xy = list(vect_omg_xy[keep_mask])
        new_omg_xz = list(vect_omg_xz[keep_mask])
        new_omg_yz = list(vect_omg_yz[keep_mask])

        new_levels = list(levels[keep_mask])

        #Refinement of the target cells:

        for idx in refine_ids:
            child_vertices, child_centers, child_weights = grids.subdivide_cells(idx, vertices, centers, weights, nx, ny, nz, Dim2D)
            n_child = len(child_weights)
            # Compute Berry CUrvature in the children cells
            child_omg_xy = np.zeros(n_child)
            child_omg_xz = np.zeros(n_child)
            child_omg_yz = np.zeros(n_child)

            for c in range(n_child):

                kvect = child_centers[c]
                _, omg_xy, omg_xz, omg_yz = TBerry(H_ij, rvects, deg, kvect, cell, fermi_energy)

                child_omg_xy[c] = omg_xy
                child_omg_xz[c] = omg_xz
                child_omg_yz[c] = omg_yz

            #Join parents and childrens
            for c in range(n_child):
                new_centers.append(child_centers[c])
                new_vertices.append(child_vertices[c])
                new_weights.append(child_weights[c])
                
                new_omg_xy.append(child_omg_xy[c])
                new_omg_xz.append(child_omg_xz[c])
                new_omg_yz.append(child_omg_yz[c])
                
                new_levels.append(levels[idx] + 1)

        centers = np.asarray(new_centers)
        vertices = np.asarray(new_vertices)
        weights = np.asarray(new_weights)
        
        vect_omg_xy = np.asarray(new_omg_xy)
        vect_omg_xz = np.asarray(new_omg_xz)
        vect_omg_yz = np.asarray(new_omg_yz)

        levels = np.asarray(new_levels)

        print(f"Level = {level + 1}")
        print(f"Total cells = {len(weights)}")
        print(f"Refined cells = {len(refine_ids)}")
        print(f"Total weight = {np.sum(weights)}")
        
    sigma_xy, sigma_xz, sigma_yz = AHC_ref(vect_omg_xy, vect_omg_yz, vect_omg_xz, weights,cell, Dim2D)
    
    return sigma_xy, sigma_xz, sigma_yz  
    



