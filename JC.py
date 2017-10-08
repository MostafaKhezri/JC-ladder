import numpy as np
import math

"""Diagonalize the Jaynes-Cummings ladder of energies"""

def get_order(omega_r, qubit_energy_list):
    """Get the order of eigenenergies that diagonalizer produces
    
    Use the bare energy of the system as reference.
    omega_r: resonator frequency, qubit_energy_list: list of qubit energies
    """
    tmax = len(qubit_energy_list) #Maximum number of transmon levels
    order = np.zeros(tmax)
    diag_bare = np.array([-i*omega_r + qubit_energy_list[i]  for i in range(tmax)]) #Bare energies of the system
    eigensolver_order = np.linalg.eigvalsh(np.diag(diag_bare)) #Eigenenergies in the order produced by diagonalizer

    for i in range(tmax):
        index, = np.where(eigensolver_order==diag_bare[i]) #Find where the diagonalizer puts the energies
        order[i] = index
    
    #Diagonalizer puts the ith energy level in order[i]th location
    return order.astype(int)

def diagonalize_RWA_strip(tot_excit, omega_r, qubit_energy_list, g_list, order):
    """Diagonalize a single RWA strip
    
    tot_excit: total number of excitations in the strip, omega_r: resonator frequency
    qubit_energy_list: list of qubit energies, g_list: list of qubit to resonator couplings. g_list[i] is qubit nearest neighbor between i+1 and i.
    order: the order that the eigenstates should be sorted
    """
    tmax = len(qubit_energy_list) #Maximum number of transmon levels, which also sets the size of the RWA strip
    diagonal_elements = np.array([(tot_excit-i)*omega_r + qubit_energy_list[i]  for i in range(tmax)]) #Diagonal elements of the RWA strip Hamiltonian
    offdiagonal_elements = np.array([ g_list[i]*math.sqrt((tot_excit-i)*(tot_excit-i>0)) for i in range(tmax-1)]) #Off-diagonal elements of the RWA strip Hamiltonian
    strip_H = np.diag(diagonal_elements) + np.diag(offdiagonal_elements, 1) + np.diag(offdiagonal_elements, -1) #Construct the total Hamiltonian of the RWA strip
    
    eigenvalues, eigenvectors = np.linalg.eigh(strip_H) #Get eigensystem
    eigenvalues = np.array([eigenvalues[i] for i in order]) #Sort eigenvalues according to the order
    eigenvectors = np.array([eigenvectors[:,i] for i in order]) #Sort eigenstates according to the order
    
    #eigenvalues[q] corresponds to eigenenergy of the RWA strip with resonator state 'tot_excit-q' and qubit state 'q'
    return [eigenvalues, eigenvectors]

def diagonalize_ladder(nmax, omega_r, qubit_energy_list, g_list):
    """Diagonalize the JC ladder
    
    nmax: maximum number of photons in the resonator, omega_r: resonator frequency
    qubit_energy_list: list of qubit energies, g_list: qubit charge matrix elements (qubit nearest neighbor couplings)
    """
    tmax = len(qubit_energy_list) #Maximum number of transmon levels
    order = get_order(omega_r, qubit_energy_list) #The order that should be used to sort the eigenenergies
    
    eigen_energies = np.zeros((nmax, tmax))
    eigen_vectors = np.zeros((nmax, tmax, tmax))
    for i in range(nmax): #Get the egeinenergies of the whole ladder
        eigen_energies[i], eigen_vectors[i] = diagonalize_RWA_strip(i, omega_r, qubit_energy_list, g_list, order)
    
    #First indice of eigen_energies signifies the total excitation number of the RWA strip and NOT the resonator state
    #The conversion is: eigenenergy of the system with resonator state 'n' and qubit state 'q' <--> eigen_energies[n+q, q]
    return eigen_energies, eigen_vectors

def get_fan_diagram(eigen_energies, omega_r):
    """Produce the 'fan diagram' of the JC ladder
    
    eigen_energies: eigenenergies produced by the diagonalize_ladder(params), omega_r: resonator frequency
    """
    nmax, tmax = len(eigen_energies), len(eigen_energies[0]) #Maximum number of photons, maximum number of transmon levels
    fan = [np.array([eigen_energies[n,q] - n*omega_r for n in range(q, nmax)]) for q in range(tmax)]
    
    #fan[q][n] is the 'q'th level fan diagram energy at 'n' photons
    return fan

def get_qubit_transitions(eigen_energies):
    """Produce the qubit transition frequencies in the JC ladder
    
    eigen_energies: eigenenergies produced by the diagonalize_ladder(params)
    """
    nmax, tmax = len(eigen_energies), len(eigen_energies[0]) #Maximum number of photons, maximum number of transmon levels
    transitions = [np.array([eigen_energies[n+q+1,q+1] - eigen_energies[n+q,q] for n in range(q, nmax-q-1)]) for q in range(tmax-1)]
    
    #transitions[q][n] is the qubit transition frequency between qubit levels 'q+1' and 'q' (omega_{q+1,q}) at 'n' photons
    return transitions

def get_res_response(eigen_energies):
    """Produce the readout resonator frequencies at each qubit state
    
    eigen_energies: eigenenergies produced by the diagonalize_ladder(params)
    """
    nmax, tmax = len(eigen_energies), len(eigen_energies[0]) #Maximum number of photons, maximum number of transmon levels
    response = [np.array([eigen_energies[n+q+1,q] - eigen_energies[n+q,q] for n in range(q, nmax-q-1)]) for q in range(tmax)]
    
    #response[q][n] is the resonator frequency at 'n' photons when the qubit is in state 'q'
    return response

def transmon_perturbative(tmax, omega_q, eta, g):
    """Produce energies and cuplings of the transmon perturbatively
    
    Uses an anharmonic approximation of the transmon to get the energies
    tmax: maximum number of transmon levels, omega_q: qubit frequency, eta: transmon anharmonicity (defined positive)
    g: coupling normalization. Value of the coupling between levels 0 and 1.
    """
    qubit_energy_list = [omega_q*q - q*(q-1)/2*eta - q*(q-1)*(q-2)/4*eta**2/omega_q for q in range(tmax)]
    g_list = [g*math.sqrt(q+1)*(1-q/2*eta/omega_q) for q in range(tmax-1)]
    
    #qubit_energy_list[q] is transmon energy for level 'q'
    #g_list[q] is transmon charge matrix element between levels 'q+1' and 'q'
    return [qubit_energy_list, g_list]