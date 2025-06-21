"""
Author: Tommaso Simionato
Date: 2025-06-12
Description:
This script performs the analysis of the extinction cross section of gold nanoparticles,
including the calculation of the size-corrected dielectric function, the optimization of parameters
for fitting experimental absorbance data, and the visualization of results. using the Mie theory or Gans theory.


This code is part of a data analysis for the course "Introduction Nanophysics" at the University of Padua held by Professor G. Mattei.
Write to tom.simionato@gmail.com for any question of corrections regarding this code.



University of Padua, Department of Physics and Astronomy,
"""




import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pdb as pdb
from scipy.optimize import minimize


"""
Varie prove con range di fit:
mask = (wavelengths >= 400) & (wavelengths <= 580) non male 
    mask = (wavelengths >= 350) & (wavelengths <= 580) mi prende abbastanza bene la coda a sx


"""


#coloriiii variii, guarda come fare decrescente
def wavelength_to_rgb(wavelength):

    gamma = 0.8
    intensity_max = 1.0

    if 380 <= wavelength <= 440:
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 < wavelength <= 490:
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif 490 < wavelength <= 510:
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif 510 < wavelength <= 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 < wavelength <= 645:
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    elif 645 < wavelength <= 780:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = g = b = 0.0

    if 350 <= wavelength < 380:
        factor = (wavelength - 350) / (380 - 350)
    elif 780 < wavelength <= 800:
        factor = (800 - wavelength) / (800 - 780)
    else:
        factor = 1.0

    def adjust(color):
        if color == 0.0:
            return 0.0
        else:
            return round(intensity_max * (color * factor) ** gamma, 3)

    return (adjust(r), adjust(g), adjust(b))

c = 3e8 #velocità luce m/s
data = np.loadtxt('G1_np-1.txt')
wavelengths_exp = data[:, 0]  # nm da 800 a 350
absorbance_exp = data[:, 1]

#pdb.set_trace()

"""
data = np.loadtxt('G1_np-1.txt')
wavelengths_exp = data[:, 0]  # nm da 800 a 350
absorbance_vuoto = data[:, 1]
"""

try:
    jc_data = np.loadtxt('epsilonAu2.txt')  #da 350 a 800    1 JC   2
    wl_jc = jc_data[:, 0]
    epsilon1_jc = jc_data[:, 1]
    epsilon2_jc = jc_data[:, 2]
except:
    print("File not found")
   

plt.figure(figsize=(12, 5))

epsilon1_jc = epsilon1_jc[::-1] #giro così da 800 a 350
epsilon2_jc = epsilon2_jc[::-1]
wl_jc = wl_jc[::-1]
epsilon_jc=epsilon1_jc+1j*epsilon2_jc

#pdb.set_trace()


"""
#absorbance
plt.subplot(121)
colors = [wavelength_to_rgb(wl) for wl in wavelengths_exp]
plt.scatter(wavelengths_exp, absorbance_exp, color=colors, s=10)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title('Data - Absorbance Au NPs')
plt.grid(True)
plt.xlim(350, 800)

#dieletricAu
plt.subplot(122)
plt.plot(wl_jc, epsilon1_jc, label='$\epsilon_1$')
plt.plot(wl_jc, epsilon2_jc, label='$\epsilon_2$')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Dielectric function')
plt.title('Johnson-Christy Au dielectric function')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""


#sigma no size-dep con R fisso
def miedipnocorr(wavelength, radius, eps1,eps2, epsilon_medium):
    wavelength_m = wavelength * 1e-9     
    radius_m = radius * 1e-9             
    omega = 2 * np.pi * constants.c / wavelength_m  
    volume = (4/3) * np.pi * radius_m**3
    prefattore = 9 * (omega / constants.c) * epsilon_medium**(1.5) *volume #formula di dipolo
    denominator = (eps1 + 2*epsilon_medium)**2 + eps2**2

#pdb.set_trace()

    sigma_ext = prefattore * (eps2 / denominator)  #m^2
    return sigma_ext * 1e18 #nm^2


#parametri per calcolo di L1 e L2

el=0.65     #eletticita si può variare
L1=((1-el**2)/el**2)*(1/(2*el)*np.log((1+el)/(1-el))-1)
L2=(1-L1)/2



def miedipnocorrgans(wavelength, radius, eps1,eps2, epsilon_medium):    #ricordarsi di switchare tra le due
   wavelength_m = wavelength * 1e-9     
   r1 = radius * 1e-9 
   r2 = r1*np.sqrt(1-el**2)
   omega = 2 * np.pi * constants.c / wavelength_m  
   volume = (4/3) * np.pi * r1*r2**2
   prefattore = 1/3 * (omega / constants.c) * epsilon_medium**(1.5) *volume #formula di dipolo
   s1= eps2/(L1**2*((eps1+epsilon_medium*(1-L1)/L1)**2+eps2**2))
   s2= eps2/(L2**2*((eps1+epsilon_medium*(1-L2)/L2)**2+eps2**2))

   sigma_ext = prefattore * (2*s2+s1)  #m^2
   return sigma_ext * 1e18 #nm^2


#paraemtri
#wl_range = np.linspace(350, 800, 450)
radius_fixed = 5  # nm
n_medium = 1.33

sigmanocorr = np.zeros_like(wavelengths_exp)
absorbanceteorica  =  np.zeros_like(wavelengths_exp)
rho0 = 4.75e-10 #stimata a priori sui fogli dal pico massimo part/nm^3
z = 1e7 #larghetta boccetta
cost = np.log10(np.e)
for i, wl in enumerate(wavelengths_exp):
    epsilon1_jc1 = epsilon1_jc[i]
    epsilon2_jc2 = epsilon2_jc[i]
    sigmanocorr[i] = miedipnocorrgans(wl, radius_fixed, epsilon1_jc1,epsilon2_jc2, n_medium**2)
    absorbanceteorica[i] = cost*z*rho0*miedipnocorrgans(wl, radius_fixed, epsilon1_jc1,epsilon2_jc2, n_medium**2)

plt.figure(figsize=(10, 6))
plt.plot(wavelengths_exp, absorbanceteorica, 'b-', linewidth=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('$\sigma_{ext}$ (nm$^2$)')
plt.title(f'Extinction cross section (R = {radius_fixed} nm)')
plt.grid(True)
plt.show()

def residuals(params, wavelengths, absorbance_exp, epsilon1, epsilon2):
    """
    Calcola i residui tra assorbanza teorica e sperimentale
    params = [R, rho, epsilon_m]
    """
    R, rho, epsilon_m = params
    z = 1e7  # spessore cella in nm
    cost = np.log10(np.e)
    
    absorbanceteo1 = np.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        sigma = miedipnocorrgans(wl, R, epsilon1[i], epsilon2[i], epsilon_m)
        absorbanceteo1[i] = cost * z * rho * sigma
    
    # Normalizzazione dei residui rispetto al massimo dell'assorbanza sperimentale
    residuals = (absorbanceteo1 - absorbance_exp) / np.max(absorbance_exp)
    return np.sum(residuals**2)

#def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def epsilon_sizedep(wavelength, R, epsilon_bulk, omega_p=1.37e16, gamma_inf=1.07e14): #rad/s
    
    if R < 2: #problemi R
        R = 2.0
    wavelength_m = wavelength * 1e-9
    R_m = R * 1e-9
    omega = 2 * np.pi * constants.c / wavelength_m
    #gammaR
    v_F = 1.4e6  #m/s
    A = 0.7  # provarne variareee cambia forme particelle
    gamma_R = gamma_inf + A * v_F / R_m  # damping 
   
    omega2_ginf2 = omega**2 + gamma_inf**2
    omega2_gR2 = omega**2 + gamma_R**2
    
    if omega == 0:
        return epsilon_bulk
    
    term1real = omega_p**2 * (1/omega2_ginf2 - 1/omega2_gR2)
    term1imag = -omega_p**2 * (gamma_inf/(omega*omega2_ginf2) - gamma_R/(omega*omega2_gR2))





    epsilon_size = epsilon_bulk + term1real + 1j*term1imag
    return epsilon_size

def calculatoreabsorbance(wavelengths, R, rho, epsilon_medium, epsilon1, epsilon2):
    absorbance = np.zeros_like(wavelengths)
    z = 1e7
    cost = np.log10(np.e)
    
    for i, wl in enumerate(wavelengths):
        epsilonbulk = epsilon1[i] + 1j*epsilon2[i]
        epsilonsize = epsilon_sizedep(wl, R, epsilonbulk) #diendente da sze
        eps1size = np.real(epsilonsize)
        eps2size = np.imag(epsilonsize)
        
        sigma = miedipnocorrgans(wl, R, eps1size, eps2size, epsilon_medium)
        absorbance[i] = cost * z * rho * sigma
    
    return absorbance

def chirange(wavelengths, absorbance_exp, absorbance_theo): #chi quadro li calcolo solo nella finestra del prof
    mask = (wavelengths >= 410) & (wavelengths <= 565)  #bisogna muovere questi, se abbasso ho R maggiori e meglio fit a sx
    chi2 = np.sum((absorbance_theo[mask] - absorbance_exp[mask])**2)
    return chi2



def chi2Rrho(R_range, rho_range, epsilon_m_fixed, wavelengths, absorbance_exp, epsilon1, epsilon2): #miniizzo mappa di residui per (R,rho)
    chi2matrix = np.zeros((len(R_range), len(rho_range)))
    
    for i, R in enumerate(R_range):
        for j, rho in enumerate(rho_range):
            absorbance_theo = calculatoreabsorbance(wavelengths, R, rho, epsilon_m_fixed, epsilon1, epsilon2)
            chi2matrix[i,j] = chirange(wavelengths, absorbance_exp, absorbance_theo)
    
    i_min, j_min = np.unravel_index(np.argmin(chi2matrix), chi2matrix.shape)
    R_best = R_range[i_min]
    rho_best = rho_range[j_min]
    
    return R_best, rho_best, chi2matrix[i_min,j_min]

def chi2rhoepsilon(rho_range, eps_range, R_fixed, wavelengths, absorbance_exp, epsilon1, epsilon2):  #stessa roba per (rho,epsilon)
    chi2matrix = np.zeros((len(rho_range), len(eps_range)))
    
    for i, rho in enumerate(rho_range):
        for j, eps in enumerate(eps_range):
            absorbance_theo = calculatoreabsorbance(wavelengths, R_fixed, rho, eps, epsilon1, epsilon2)
            chi2matrix[i,j] = chirange(wavelengths, absorbance_exp, absorbance_theo)
    
    i_min, j_min = np.unravel_index(np.argmin(chi2matrix), chi2matrix.shape)    #dove ho minimo dei residui guarda se puoi fare meglio
    rho_best = rho_range[i_min]
    eps_best = eps_range[j_min]
    
    return rho_best, eps_best, chi2matrix[i_min,j_min]

def ottimizzazione(chi2_threshold=1e-10, max_iterations=100, min_improvement=1e-8):
    R_range = np.linspace(2, 10, 500 )      #non troppi passi per favore
    rho_range = np.logspace(-12, -8, 400)  
    eps_range = np.linspace(1.6, 2.4, 19) 
    
    Rinziale = [5.0]  #comincio partendo da più valori iniziali di R per vedere se cconverge allas tessa cosa chiedi a matte
    best_params = None
    best_chi2 = np.inf
    
    for R_start in Rinziale:
        print(f"\nOptimization with initial R = {R_start} nm")
        R_current = R_start
        rho_current = 1e-8 #stimato da calcolo
        eps_current = n_medium**2
        chi2_history = []
        param_history = []
        
        iteration = 0
        stop_counter = 0  #contatore per vedere quando non porta a nessun miglioramenteo
        
        while iteration < max_iterations:
            R_best, rho_best, chi2_Rrho = chi2Rrho(R_range, rho_range, eps_current, 
                                                             wavelengths_exp, absorbance_exp, 
                                                             epsilon1_jc, epsilon2_jc)
            R_current = R_best  #aggirono
            rho_current = rho_best

            rho_best, eps_best, chi2_rhoeps = chi2rhoepsilon(rho_range, eps_range, R_current,
                                                                   wavelengths_exp, absorbance_exp,
                                                                   epsilon1_jc, epsilon2_jc)
            rho_current = rho_best  #aggiorno
            eps_current = eps_best

            absorbance_current = calculatoreabsorbance(wavelengths_exp, R_current, rho_current, 
                                                    eps_current, epsilon1_jc, epsilon2_jc)
            current_chi2 = chirange(wavelengths_exp, absorbance_exp, absorbance_current)    #chi quadro nella finestra

            chi2_history.append(current_chi2)
            param_history.append((R_current, rho_current, eps_current))

            if iteration % 10 == 0: #checcko ogni 20 iterations
                print(f"Iteration {iteration}, χ² = {current_chi2:.2e}")
                print(f"R = {R_current:.2f} nm, ρ = {rho_current:.2e} part/nm³, ε_m = {eps_current:.3f}")
            if current_chi2 < chi2_threshold:
                print(f"\nConvergence reached! χ² = {current_chi2:.2e}")
                break
                
            if iteration > 0:       #guardo quando è troppo lento
                improvement = (chi2_history[-2] - current_chi2) / chi2_history[-2]
                if improvement < min_improvement:
                    stop_counter += 1
                else:
                    stop_counter = 0
                if stop_counter > 5:  #lo fermoooo (prima avevo messo a dieci)
                    print(f"\nOptimization stagnating for R_start = {R_start}")
                    break
            
            range_factor = max(0.3, 0.98**iteration) #stringo la ricerca
            
            R_min = max(2.0, R_current * (1-range_factor))
            R_max = min(30.0, R_current * (1+range_factor))
            R_range = np.linspace(R_min, R_max, 29)
            rho_range = np.logspace(np.log10(rho_current) - range_factor, 
                                  np.log10(rho_current) + range_factor, 41)
            eps_range = np.linspace(eps_current * (1-range_factor), 
                                  eps_current * (1+range_factor), 31)
            iteration += 1
        
       
        best_idx = np.argmin(chi2_history) #dato R guardo param migliori
        if chi2_history[best_idx] < best_chi2:
            best_chi2 = chi2_history[best_idx]
            best_params = param_history[best_idx]
            best_history = chi2_history
   
    R_best, rho_best, eps_best = best_params #uso i migliori
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(best_history, 'b.-')
    plt.xlabel('Iteration')
    plt.ylabel('χ²')
    plt.title('χ² trend during optimization')
    plt.grid(True)
    plt.show()
 
    miglioreabs = calculatoreabsorbance(wavelengths_exp, R_best, rho_best, #assorbanza finale
                                         eps_best, epsilon1_jc, epsilon2_jc)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    colors = [wavelength_to_rgb(wl) for wl in wavelengths_exp]
    plt.scatter(wavelengths_exp, absorbance_exp, color=colors, s=10, label='Experimental')
    plt.plot(wavelengths_exp, miglioreabs, 'r-', linewidth=2, 
            label=f'Best fit (χ²={best_chi2:.2e})')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.title('Full range comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    mask = (wavelengths_exp >= 410) & (wavelengths_exp <= 560)
    plt.scatter(wavelengths_exp[mask], absorbance_exp[mask], color='b', s=10, 
               label='Experimental (fit window)')
    plt.plot(wavelengths_exp[mask], miglioreabs[mask], 'r-', linewidth=2, 
             label='Best fit')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.title('Fit window') #mostro fit su zona consigliata dal prof
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("\nFinal parameters:")
    print(f"R = {R_best:.2f} nm")
    print(f"ρ = {rho_best:.2e} part/nm³")
    print(f"ε_m = {eps_best:.3f}")
    print(f"χ² final = {best_chi2:.2e}")
    
    return R_best, rho_best, eps_best, best_chi2

def plotepsiloncorretta(wl_range, radii=[3, 5, 10]):

    plt.figure(figsize=(12, 8)) #plotto la epsilon correttaaaa
    eps1_bulk = epsilon1_jc
    eps2_bulk = epsilon2_jc
    plt.plot(wl_range, eps1_bulk, 'k-', linewidth=2, label='$\epsilon_1$ bulk')
    plt.plot(wl_range, eps2_bulk, 'k--', linewidth=2, label='$\epsilon_2$ bulk')
    
    bright_colors = ['red', 'green', 'purple', 'orange']
    colors = bright_colors[:len(radii)]
    
    for R, color in zip(radii, colors):
        eps1_size = np.zeros_like(wl_range)
        eps2_size = np.zeros_like(wl_range)
        
        for i, wl in enumerate(wl_range):#CAPSICI come emttere label affianco alle linee
            epsilon_bulk = epsilon1_jc[i] + 1j * epsilon2_jc[i]
            epsilon_size = epsilon_sizedep(wl, R, epsilon_bulk)
            eps1_size[i] = np.real(epsilon_size)
            eps2_size[i] = np.imag(epsilon_size)
        
        plt.plot(wl_range, eps1_size, '-', color=color, linewidth=1.5, label=fr'$\epsilon_1$, R={R}nm')
        plt.plot(wl_range, eps2_size, '--', color=color, linewidth=1.5, label=fr'$\epsilon_2$, R={R}nm')

    plt.xlabel('Wavelength (nm)', fontsize=16)
    plt.ylabel('Dielectric function', fontsize=16)
    plt.title('Size-dependent dielectric function for different radii', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower left', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plotepsiloncorretta(wavelengths_exp, radii=[3, 5, 10])

#ottimiazzzzzaione del chi quadro
print("Otimization Size Dependent ongoing.")
R_final, rho_final, eps_final, final_chi2 = ottimizzazione()
