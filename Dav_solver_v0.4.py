#!/usr/bin/env python
# tested on python/3.8.6 py-numpy/1.18.5 py-scipy/1.5.3

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import subprocess
import time
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

#### input and output files ####
# input files for siesta for running a scf calculation on the relaxed atomic coordinates to get the density matrix (*.DM) file
fil_siesta_relaxed = "tbg-input-relaxed.fdf"
# input files for siesta for calculating the force for atomic coordinates
fil_siesta_force = "tbg-input-force.fdf"
# relaxed atomic coordinates, in siesta format and Angstrom units (if changed, also need to change the input file fil_siesta_relaxed)
fil_coord_relaxed = "coord-relaxed.fdf"
# siesta pseudopotential file(s)
fil_pseudo = "*.psml *.psf"
# SystemLabel in siesta input file
SystemLabel = "tbg"
# initial phonon vector (only needed if vector_initial_source = "file")
fil_vector_initial = "u-diag.dat"
# save file for intermediate steps (will be read if restart = True)
fil_saves = "dav_saves.npy"
# converged phonon eigenvector
fil_eigv_out = "eigv.dat"

#### control parameters for the iterative solver ####
# atomic displacement (in Ang) used for force evaluation
displ_size = 0.02
# whether displ_size is defined as the 2-norm ["norm"] or the largest element ["max"] of the vector
displ_size_definition = "max" 
# convergence threshold for the angle between u and Du 
Davidson_conv_thr = 0.02
Davidson_max_step = 100
Davidson_recompute_step = 999

#### initial vector, cell parameters and other settings ####
# whether to skip the initial scf calculation on relaxed atomic coordinates
skip_relaxed = True 
# if true, restart an interrupted calculation
restart = False 
# how we get the parameters for the model dynamical matrix: options are ["random"] and ["input"]. 
# ["random"]: from least-square fitting of the forces induced by a random atomic displacement; model_DM_para act as the initial guess 
# ["input"]: directly use the values given here by model_DM_para
model_DM_source = "random" 
#model_DM_para = np.array([12.1, 4.5, 21.5])
# whether we symmetrize the force computed from Siesta using known symmetries of the system
symmetrize_force = True
# how we get the initial phonon vector: options are ["file"] or ["function"]. If ["function"], use the function defined here
vector_initial_source = "function" 
def vector_initial_func(coord):
    # set initial phonon vector as a function of the relaxed atomic coordinates, given by [coord] 
    n_atoms = len(coord)
    disp_init = 0.0*coord
    n_per_layer = int(n_atoms/2)
    b = 4*np.pi/np.sqrt(3)/cell_para[0,0]
    for i in range(n_per_layer):
        # layer breathing mode of tBG, centered at AA
        (x, y) = coord[i,0:2] 
        disp_init[i,2] = 0.5 + (np.cos(b*y) + np.cos(b*(0.5*y + 0.5*np.sqrt(3)*x)) + np.cos(b*(-0.5*y + 0.5*np.sqrt(3)*x)))/3
        (x, y) = coord[n_per_layer + i,0:2]
        disp_init[n_per_layer + i,2] = -0.5 - (np.cos(b*y) + np.cos(b*(0.5*y + 0.5*np.sqrt(3)*x)) + np.cos(b*(-0.5*y + 0.5*np.sqrt(3)*x)))/3
        
        # shear mode of tBG
#        disp_init[i,0] = 1 
#        disp_init[n_per_layer + i,0] = -1 
    disp_init = disp_init/np.linalg.norm(disp_init)
    return disp_init

#### names of temparary directories and files ####
dir_relaxed = "./relaxed/"
dir_force = "./force-eval/"
dir_force2 = "./force-eval2/" # for debug
# atomic coordinates perturbed by phonon vector, in siesta format and Angstrom units (if changed, also need to change the input file fil_siesta_force)
fil_coord_force = "coord-tmp.fdf"

#######################################
######## main body of the code ########
#######################################

#### some golbal variables ####
force_eval_count = 0
time_siesta = 0.0
# conversion between Siesta force constant unit (eV/Ang^2) and phonon frequency units (meV and cm^-1), assuming mass is in a.m.u.
fc_to_meV = 64.656
fc_to_cm = 521.5

# find the new index of each atom when rotated by R
def find_symm_pairs(coord, R):
    n_atoms = len(coord)
    pairs = np.zeros(n_atoms)
    for i in range(n_atoms):
        xyz_transformed_frac = coord[i]@R@np.linalg.inv(cell_para)
        for d in range(3):
            if xyz_transformed_frac[d] > 1 - 0.0001:
                xyz_transformed_frac[d] = xyz_transformed_frac[d] - 1
            elif xyz_transformed_frac[d] < -0.0001:
                xyz_transformed_frac[d] = xyz_transformed_frac[d] + 1
        xyz_transformed = xyz_transformed_frac@cell_para
        for j in range(n_atoms):
            if np.linalg.norm(xyz_transformed - coord[j]) < 0.01:
                pairs[i] = j
                break
        else:
            print("missing partner for atom", i, xyz_transformed_frac, xyz_transformed) 
    return pairs

# symmetrize vector u
def u_symmetrize(u):
    u_transformed = np.zeros((6,len(u)))
    u_transformed[0] = u
    for s in range(5):
        for i in range(n_atoms):
            ip = symm_map_list[s,i]
            u_transformed[s+1, ip*3:ip*3+3] = u[i*3:i*3+3]@symm_rot_list[s]
    return np.sum(u_transformed, axis=0)/6

# symmetrize force constant matrix fc
def fc_symmetrize(fc):
    fc_transformed = np.zeros((6,fc.shape[0],fc.shape[1]))
    fc_transformed[0] = fc
    for s in range(5):
        R = symm_rot_list[s]
        for i in range(n_atoms):
            for j in range(n_atoms):
                ip = symm_map_list[s,i]; jp = symm_map_list[s,j]
                fc_transformed[s+1, ip*3:ip*3+3, jp*3:jp*3+3] = R.T@fc[i*3:i*3+3, j*3:j*3+3]@R
    fc_symmtrized = np.sum(fc_transformed, axis=0)/6
#    print("Asymmetry under rotations:", np.linalg.norm(fc_symmtrized - fc))
    return fc_symmtrized

# compute the forces on atoms when an atomic displacement [displ, array shape (n_atoms, 3)] is given, by runnig Siesta
def displ_to_force(displ, dir_force):
    global force_eval_count, time_siesta # keeping track of the total number of force evaluations and siesta execution time
    
    # write displaced atomic coordinates to file
    coord_displ = coord_relaxed + displ
    coord_displ_siesta = np.concatenate((coord_displ, np.array([atomic_types]).T), axis = 1)
    np.savetxt(dir_force + fil_coord_force, coord_displ_siesta, fmt = ['%12.6f', '%12.6f', '%12.6f', '%3i'], comments = '', 
               header = "\nAtomicCoordinatesFormat  NotScaledCartesianAng \n%block AtomicCoordinatesAndAtomicSpecies", 
              footer = "%endblock AtomicCoordinatesAndAtomicSpecies")

    # compute force with updated atomic coordinates (DM file in relaxed atomic coordinates is used as starting DM to speed up convergence)
    siesta_start_time = time.time()
    subprocess.run("cp " + dir_relaxed + "*.DM " + dir_force, shell = True) 
    subprocess.run("mpirun siesta < " + fil_siesta_force + " > force.out", shell = True, cwd = dir_force) 
    siesta_end_time = time.time()
    time_siesta += siesta_end_time - siesta_start_time
    
    force_eval_count += 1
    
    # read the computed force from file
    force_siesta = np.loadtxt(dir_force + SystemLabel + ".FA", skiprows = 1)
    return force_siesta[:,1:4]

# The function dyn_map takes a phonon vector u and computes Du (the vector one gets when the dynamical matrix D is acted on u)
# Input vectors u and output vector Du are 1-D arrays of length 3*n_atoms
def dyn_map(u, if_symmetrize=False):    
    # get the atomic displacement of desired size
    if displ_size_definition == "norm":
        displ_u_ratio = displ_size/np.linalg.norm(u)
    elif displ_size_definition == "max":
        displ_u_ratio = displ_size/np.max(np.abs(u))
    else:
        print("displ_size_definition not defined.")
    
    # displ, force are 2-D arrays of shape (n_atoms, 3)
    displ = displ_u_ratio * np.reshape(u,(n_atoms,3))
    
#    np.savetxt("u-current.dat", displ, fmt = '%12.6f', header = "displ_u_ratio = " + f"{displ_u_ratio:.4f}")
    
    # average over two opposite displacements 
    force1 = displ_to_force(displ, dir_force)
    force2 = displ_to_force(-displ, dir_force2)
#    force_asym = np.linalg.norm(force1 + force2)/(np.linalg.norm(force1) + np.linalg.norm(force2))
#    print("############ Force assymetry = ", f"{force_asym:.4f}", flush=True)
    # note it's force2 - force1 not the other way round due to the minus sign in the definition of force
    Du = np.ravel((force2 - force1)/2/displ_u_ratio)  
    if len(Du) != len(u):
        print("Output and input vector dimension mismatch.")
    if if_symmetrize:
        Du = u_symmetrize(Du)
    return Du

# build a model force constant matrix in sparse format, from a total of 14 parameters up to next-next-nearest neighbor
neighbor_list = np.array([]); 
def DM_build(coord, cell_para, fc_para):
    global neighbor_list
    n_per_layer = int(n_atoms/2)
    
    # distance of NN, NNN and two types of NNNN
    a = 2.46; b = a/np.sqrt(3)
    r_list = np.array([b, a, 2*b, np.sqrt(7)*b])
    
    if len(neighbor_list) == 0:
        for i in range(n_per_layer):
            for j in range(i):
                rij = coord[i] - coord[j]
                rij_frac = rij@np.linalg.inv(cell_para)
                for d in range(3):
                    if rij_frac[d] > 0.5:
                        rij_frac[d] = rij_frac[d] - 1
                    elif rij_frac[d] < -0.5:
                        rij_frac[d] = rij_frac[d] + 1
                rij = rij_frac@cell_para
                if np.linalg.norm(rij) < r_list[-1] + 0.1:
                    neighbor_match = np.abs(np.linalg.norm(rij) - r_list)
                    pair_type = np.argmin(neighbor_match)
                    if neighbor_match[pair_type] > 0.1:
                        print("Pair does not fall in prescribed neighbor types:", np.array([[i,j,np.linalg.norm(rij),rij[0],rij[1],rij[2]]]))
                    if len(neighbor_list) == 0:
                        neighbor_list = np.array([[i,j,pair_type,rij[0],rij[1],rij[2]]])
                    else:
                        neighbor_list = np.concatenate((neighbor_list, np.array([[i,j,pair_type,rij[0],rij[1],rij[2]]])))
                        
                # the other layer
                ip = i + n_per_layer; jp = j + n_per_layer
                rij = coord[ip] - coord[jp]
                rij_frac = rij@np.linalg.inv(cell_para)
                for d in range(3):
                    if rij_frac[d] > 0.5:
                        rij_frac[d] = rij_frac[d] - 1
                    elif rij_frac[d] < -0.5:
                        rij_frac[d] = rij_frac[d] + 1
                rij = rij_frac@cell_para
                if np.linalg.norm(rij) < r_list[-1] + 0.1:
                    neighbor_match = np.abs(np.linalg.norm(rij) - r_list)
                    pair_type = np.argmin(neighbor_match)
                    if neighbor_match[pair_type] > 0.1:
                        print("Pair does not fall in prescribed neighbor types:", np.array([[ip,jp,np.linalg.norm(rij),rij[0],rij[1],rij[2]]]))
                    if len(neighbor_list) == 0:
                        neighbor_list = np.array([[ip,jp,pair_type,rij[0],rij[1],rij[2]]])
                    else:
                        neighbor_list = np.concatenate((neighbor_list, np.array([[ip,jp,pair_type,rij[0],rij[1],rij[2]]])))
        print("Number of neighbor pairs in the model dynamical matrix:", len(neighbor_list), ", per atom:", len(neighbor_list)/n_atoms)
    
    # from input array of parameters to matrix form that can be transformed by rotation
    fc_xy = np.zeros((len(r_list),2,2)); fc_z = np.zeros(len(r_list))
    for i in range(len(fc_para)):
        i_pair = int(i/3)
        if i%3 == 0:
            fc_xy[i_pair][0,0] = fc_para[i]
        elif i%3 == 1:
            fc_xy[i_pair][1,1] = fc_para[i]
        else:
            fc_z[i_pair] = fc_para[i]
    
    # pre-calculated matrix elements from 21.8 degree tBLG
    # All assume the pair is along x direction, with atom i on the left and atom j on the right
    # C.psml
    #    fc_xy = np.array([[[-31.08, 0], [0, -10.27]], 
    #                      [[-4.77, -0.35], [0.35, 2.73]], 
    #                      [[0.95, 0], [0, -2.05]], 
    #                      [[0.44, -0.65], [-0.65, -0.27]]])
    #    fc_z = np.array([-5.42, 0.39, -0.39, 0.51])
    # C.psf
    #fc_xy = np.array([[[-28.45, 0], [0, -10.79]], 
    #                  [[-4.76, -0.42], [0.42, 2.76]], 
    #                  [[0.90, 0], [0, -2.07]], 
    #                  [[0.44, -0.64], [-0.64, -0.26]]])
    #fc_z = np.array([-5.93, 0.42, -0.39, 0.49])
    fc_diag_xy = -np.trace(3*fc_xy[0] + 6*fc_xy[1] + 3*fc_xy[2] + 6*fc_xy[3])/2
    fc_diag_z = -(3*fc_z[0] + 6*fc_z[1] + 3*fc_z[2] + 6*fc_z[3])
    
    row = np.array([], dtype=int); 
    col = np.array([], dtype=int); 
    matele = np.array([], dtype=float);
    
    for pair in neighbor_list:
        i = int(pair[0]); j = int(pair[1]); pair_type = int(pair[2]); rij = pair[3:6]
        rhat = rij/np.linalg.norm(rij)
        R = np.array([[rhat[0], rhat[1]], [-rhat[1], rhat[0]]])
        fc_ij_xy = R.T@fc_xy[pair_type]@R
        fc_ij_z = fc_z[pair_type]

        row_ij = [i*3, i*3+1, i*3, i*3+1, i*3+2]
        col_ij = [j*3, j*3+1, j*3+1, j*3, j*3+2]
        matele_ij = [fc_ij_xy[0,0], fc_ij_xy[1,1], fc_ij_xy[0,1], fc_ij_xy[1,0], fc_ij_z]
        row = np.append(row, row_ij)
        col = np.append(col, col_ij)
        matele = np.append(matele, matele_ij)
        row = np.append(row, col_ij)
        col = np.append(col, row_ij)
        matele = np.append(matele, matele_ij)
    
    for i in range(n_atoms):
        row_ij = [i*3, i*3+1, i*3+2]
        col_ij = [i*3, i*3+1, i*3+2]
        matele_ij = [fc_diag_xy, fc_diag_xy, fc_diag_z]
        row = np.append(row, row_ij)
        col = np.append(col, col_ij)
        matele = np.append(matele, matele_ij)
    fc_out = sp.csr_matrix((matele, (row, col)), shape=(n_atoms*3, n_atoms*3))
    return fc_out

# optimize force constant parameters in the model
def model_DM_optimize(u, Du):
    N_fc_para = 12; fc_components = [] # each independent component of the f.c.
    for i in range(N_fc_para):
        fc_para = np.zeros(N_fc_para)
        fc_para[i] = 1
        fc_components.append(DM_build(coord_relaxed, cell_para, fc_para))

    coeff = np.zeros((N_fc_para, n_atoms*3))
    for i in range(N_fc_para):
        coeff[i] = u@fc_components[i]
    model_para = Du@np.linalg.pinv(coeff)
    return model_para

##########################################
######## code starts running here ########
##########################################
program_start_time = time.time()
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(program_start_time))
print("Program starts at:", formatted_time)
print("") 
print("Main parameters:") 
print("displ_size (Ang):", displ_size, ", displ_size_definition:", displ_size_definition, ", symmetrize_force:", symmetrize_force) 
print("Davidson_conv_thr:", Davidson_conv_thr, ", Davidson_max_step:", Davidson_max_step, 
      ", Davidson_recompute_step:", Davidson_recompute_step, flush=True) 
print("") 

if not(skip_relaxed):
    subprocess.run(["mkdir", "-p", dir_relaxed])
    subprocess.run(["cp", fil_siesta_relaxed, dir_relaxed])
    subprocess.run(["cp", fil_coord_relaxed, dir_relaxed])
    subprocess.run("cp " + fil_pseudo + " " + dir_relaxed, shell = True)
    subprocess.run("mpirun siesta < " + fil_siesta_relaxed + " > scf.out", shell = True, cwd = dir_relaxed) 
    print("scf calculation for relaxed configuration done.", flush=True)
    
# prepare the directory for force evaluation
subprocess.run(["mkdir", "-p", dir_force])
subprocess.run(["cp", fil_siesta_force, dir_force])
subprocess.run("cp " + fil_pseudo + " " + dir_force, shell = True)

# prepare the directory for force evaluation
subprocess.run(["mkdir", "-p", dir_force2])
subprocess.run(["cp", fil_siesta_force, dir_force2])
subprocess.run("cp " + fil_pseudo + " " + dir_force2, shell = True)

# read the relaxed atomic coordinates and types
coord_relaxed = np.loadtxt(fil_coord_relaxed, skiprows = 1, comments = '%')
n_atoms = len(coord_relaxed)
atomic_types = coord_relaxed[:,3]
coord_relaxed = coord_relaxed[:,0:3]

# read the cell parameters
with open(fil_siesta_force,'r') as f:
    for line in f:
        if '%block LatticeParameters' in line:
            lat = np.fromstring(next(f), sep=' ')
            (alpha, beta, gamma) = lat[3:6]*np.pi/180
            cell_para = np.array([[lat[0], 0, 0], [lat[1]*np.cos(gamma), lat[1]*np.sin(gamma), 0], 
                                  [lat[2]*np.cos(beta), lat[2]*(np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma), 
                                   lat[2]*np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) - 
                                                  np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2)/np.sin(gamma)]])

# define the translational modes, later we want to remove these components
transl_x = np.tile(np.array([1,0,0]), n_atoms); transl_x = transl_x/np.linalg.norm(transl_x)
transl_y = np.tile(np.array([0,1,0]), n_atoms); transl_y = transl_x/np.linalg.norm(transl_y)
transl_z = np.tile(np.array([0,0,1]), n_atoms); transl_z = transl_x/np.linalg.norm(transl_z)
transl_modes = np.array([transl_x, transl_y, transl_z]).T

# define symmetry rotations (symm_rot_list) and find how atoms are mapped by them (symm_map_list)
if symmetrize_force:
    theta = np.pi*2/3
    R1 = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]);
    R2 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]);
    symm_rot_list = np.array([R1, R1@R1, R2, R2@R1, R2@R1@R1])
    symm_map_list = np.zeros((5,n_atoms), dtype=int)
    for i in range(5):
        symm_map_list[i] = find_symm_pairs(coord_relaxed, symm_rot_list[i])

# initialization of the iterative process
if restart == False:
    # get the initial guess of phonon eigenvector
    if vector_initial_source == "function":
        u_0 = vector_initial_func(coord_relaxed)
    elif vector_initial_source == "file": 
        u_0 = np.loadtxt(fil_vector_initial)
    else:
        print("vector_initial_source not defined.")
    u_0 = np.ravel(u_0) # keep u_0 as a 1-D array of length 3*n_atoms
    for d in range(3): u_0 = u_0 - (transl_modes[:,d]@u_0)*transl_modes[:,d] # remove any translational component
    if symmetrize_force:
        u_0p = u_symmetrize(u_0)
        if np.linalg.norm(u_0 - u_0p) > 10**-5:
            print("Initial phonon vector is not symmetric. Norm of asymmetric part:", f"{np.linalg.norm(u_0 - u_0p):.8f}")
            print("Symmetrized vector will be used.", flush=True)
            u_0 = u_0p
    u_0 = u_0/np.linalg.norm(u_0)
    np.savetxt("u-initial.dat", np.reshape(u_0,(n_atoms,3)), fmt = '%10.6f')
    print("Initial phonon vector prepared/read.", flush=True)
    
    if model_DM_source == "random":
        # calculate force on a randomly-displaced configuration to get a model force-constant matrix
        print("Constructing approximate dynamical matrix from a random configuration.", flush=True)
        rng = np.random.default_rng()
        u = rng.normal(0, 1, n_atoms*3)
        u = u/np.linalg.norm(u)
        Du = dyn_map(u)
        model_DM_para = model_DM_optimize(u, Du)
        print("Optimized model DM parameters:", model_DM_para, flush=True)
    else:
        print("Using model DM parameters:", model_DM_para, flush=True)
    DM_model = DM_build(coord_relaxed, cell_para, model_DM_para).todense()
    if symmetrize_force: DM_model = fc_symmetrize(DM_model)
    
    # Initial step of Davidson
    u = u_0; Du = dyn_map(u, symmetrize_force)
    for d in range(3): Du = Du - (transl_modes[:,d]@Du)*transl_modes[:,d]
    rq = u@Du; r = Du - rq*u
    theta = np.arccos(rq/np.linalg.norm(Du))
    rq_0 = rq; u_best = u; rq_best = rq; theta_best = theta; # s_best = np.array([1.0])

    subspace_v = np.array([u]).T          # the orthonormal basis
    subspace_v_raw = np.array([u]).T      # all vectors used as input in force calculations, can be linearly dependent of each other
    subspace_Dv_raw = np.array([Du]).T    # all outputs from above
    v_pinv = np.linalg.pinv(subspace_v_raw)
    M = subspace_Dv_raw@v_pinv@subspace_v - rq_best*subspace_v
    s = np.array([1.0])
    
    i_dav = 0 # step counter
    print("")
    print("#### Initial state ---- RQ =", f"{rq:.5f}", ", theta(u,Du) =", f"{theta:.5f}", flush=True)
    
else: # restart == True
    print("Restarting an interrupted run.")
    [model_DM_para, i_dav, force_eval_count, rq_best, subspace_v, subspace_v_raw, subspace_Dv_raw] = np.load(fil_saves, allow_pickle=True)
#    [i_dav, force_eval_count, rq_best, subspace_v, subspace_v_raw, subspace_Dv_raw] = np.load(fil_saves, allow_pickle=True)
    print("Using model DM parameters:", model_DM_para, flush=True)
    DM_model = DM_build(coord_relaxed, cell_para, model_DM_para).todense()
    if symmetrize_force: DM_model = fc_symmetrize(DM_model)
    
    v_pinv = np.linalg.pinv(subspace_v_raw)
    M = subspace_Dv_raw@v_pinv@subspace_v - rq_best*subspace_v
    eig = np.linalg.eigh(M.T@M)
    s = eig[1][:,0]
    if s[0] < 0: s = -s
    u = subspace_v@s; Du = subspace_Dv_raw@v_pinv@u
    rq = u@Du; r = Du - rq*u
    theta = np.arccos(rq/np.linalg.norm(Du))
    
    u_best = u; rq_best = rq; theta_best = theta;
    
    print("")
    print("#### Restarting state : RQ =", f"{rq:.5f}", ", theta(u,Du) =", f"{theta:.5f}", 
          ", number of independent/total trial vectors:", subspace_v.shape[1], subspace_v_raw.shape[1], flush=True)    

# Main loop starts
while theta_best > Davidson_conv_thr: 
    # Jacobi-Davidson: get the next vector t by solving (I - [u].T@[u])(D - rq*I)(I - [u].T@[u])t = r. 
    # Here the approximate dynamical matrix ("preconditioner") DM_model is used in place of D. 
    # the closer DM_model is to the real D, the faster is the convergence
    # use np.linalg.lstsq instead of np.linalg.solve because the matrix can get close to being singular
#    D_diag_guess = np.tile(Davidson_diag_factor, n_atoms)
#    t = r/D_diag_guess
#    t = spsolve(rq_best*sp.eye(3*n_atoms) - DM_model, r)
    M1 = DM_model - rq*np.eye(3*n_atoms)
    proj_u = np.eye(3*n_atoms) - np.outer(u,u)
    t = np.linalg.lstsq(proj_u@M1@proj_u, r, rcond = None)[0] 
    # M2 should be more accurate than M1? it uses known results inside the existing subspace and only DM_model outside of existing subspace
#    M2 = (M - np.outer(r,s))@subspace_v.T + M1@(np.eye(3*n_atoms) - subspace_v@subspace_v.T)
#    t = np.linalg.lstsq(proj_u@M2, -r, rcond = None)[0]
    t = t/np.linalg.norm(t);
    
    # orthorgonalize the new vector to existing subspace
    t = t - subspace_v@subspace_v.T@t
    if np.linalg.norm(t) < 10**-6:
        print("New vector is not linearly independent. Exiting loop.")
        break
        
    # add the new vectors to the subspace
    v = t/np.linalg.norm(t); Dv = dyn_map(v, symmetrize_force);
    for d in range(3): Dv = Dv - (transl_modes[:,d]@Dv)*transl_modes[:,d]
    subspace_v = np.concatenate((subspace_v, np.array([v]).T), axis = 1)
    subspace_v_raw = np.concatenate((subspace_v_raw, np.array([v]).T), axis = 1)
    subspace_Dv_raw = np.concatenate((subspace_Dv_raw, np.array([Dv]).T), axis = 1)
    
    # find the vector s (and u = subspace_v@s) that minimizes the residual r = D.u - rq_best*u
    # In case subspace_v_raw is not linearly independent, 
    # D is approximated by subspace_Dv_raw@v_pinv, where v_pinv is the Moore-Penrose pseudoinverse of subspace_v_raw
    # this works for the case when subspace_v_raw is linearly independent as well
    v_pinv = np.linalg.pinv(subspace_v_raw)
    M = subspace_Dv_raw@v_pinv@subspace_v - rq_best*subspace_v
    s0 = np.zeros(subspace_v.shape[1]); s0[0] = 1
    H = M.T@M  #- rq_best**2*theta_best**2*np.outer(s0,s0)
    eig = np.linalg.eigh(H) 
    s = eig[1][:,0] # we want the smallest eigenpair, which is the first one if we use np.linalg.eigh
    if s[0] < 0: s = -s
#    print("#### s =", s)
    u = subspace_v@s; Du = subspace_Dv_raw@v_pinv@u
    rq = u@Du; r = Du - rq*u
    theta = np.arccos(rq/np.linalg.norm(Du))
    
    i_dav += 1      
    # track progress
    print("#### Davidson step", f"{i_dav:2d}", ", RQ =", f"{rq:.5f}", ", theta(u,Du) =", f"{theta:.5f}", 
          ", number of independent/total trial vectors:", subspace_v.shape[1], subspace_v_raw.shape[1], flush=True)
            
    # In case of nonharmonticity: recompute another Du directly from u every few steps
    if i_dav%Davidson_recompute_step == 0:
        Du1 = dyn_map(u, symmetrize_force)
        for d in range(3): Du1 = Du1 - (transl_modes[:,d]@Du1)*transl_modes[:,d]
        subspace_v_raw = np.concatenate((subspace_v_raw, np.array([u]).T), axis = 1)
        subspace_Dv_raw = np.concatenate((subspace_Dv_raw, np.array([Du1]).T), axis = 1)
        v_pinv = np.linalg.pinv(subspace_v_raw)
        
        # repeat the process with new subspace
        M = subspace_Dv_raw@v_pinv@subspace_v - rq_best*subspace_v
        eig = np.linalg.eigh(M.T@M)
        s = eig[1][:,0]
        if s[0] < 0: s = -s
        u = subspace_v@s;         
        Du = subspace_Dv_raw@v_pinv@u
        rq = u@Du; r = Du - rq*u
        theta = np.arccos(rq/np.linalg.norm(Du))
        
        print(">>>>>> Recompute step , RQ =", f"{rq:.5f}", ", theta(u,Du) =", f"{theta:.5f}", 
              ", number of independent/total trial vectors:", subspace_v.shape[1], subspace_v_raw.shape[1])   
        print(">>>>>> Maximum weight on a single force calculation:", f"{np.max(np.abs(v_pinv@u)):.5f}", flush=True)
        
    # Update the best vector if we make forward progress
    if theta < theta_best:
        u_best = u
        rq_best = rq
        theta_best = theta
#        s_best = s
        np.savetxt("u-best.dat", np.reshape(u_best,(n_atoms,3)), fmt = '%10.6f')
    
    if i_dav%3 == 0:
        # Save progress, in case we wish to continue an interrupted run
        np.save(fil_saves, [model_DM_para, i_dav, force_eval_count, rq_best, subspace_v, subspace_v_raw, subspace_Dv_raw])
        
    if i_dav >= Davidson_max_step:
        print("#### Davidson max step reached. Stopping.", flush=True)
        break

print("Final result: RQ =", f"{rq_best:.5f}", ", theta(Du,u) =", f"{theta_best:.5f}", ", theta(u,u_0) =", f"{np.arccos(u@subspace_v[:,0]):.5f}", flush=True)
print("")

amass = 12
if rq_best >= 0:
    omega_cm = fc_to_cm*np.sqrt(rq_best/amass)
    omega_meV = fc_to_meV*np.sqrt(rq_best/amass)
else:
    omega_cm = -fc_to_cm*np.sqrt(-rq_best/amass)
    omega_meV = -fc_to_meV*np.sqrt(-rq_best/amass)
print("Phonon energy = ", f"{omega_cm:.3f}", " cm^-1")
print("Phonon energy = ", f"{omega_meV:.3f}", " meV")
print("")

# Save converged phonon eigenvector
np.savetxt(fil_eigv_out, np.reshape(u_best,(n_atoms,3)), fmt = '%12.6f')
print("Phonon eigenvector written to ", fil_eigv_out)

print("Total number of force evaluations:", force_eval_count)

program_end_time = time.time()
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(program_end_time))
print("")
print("Program ends at:", formatted_time)
print("Total elapsed time (sec):", f"{program_end_time - program_start_time:.3f}")
print("Siesta execution time (sec):", f"{time_siesta:.3f}")