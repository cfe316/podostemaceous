import numpy as np

from podostemaceous.constants import u
from podostemaceous.molecule import VHSMolecule
from podostemaceous.discretization_error_estimates import *

mLi = 6.94 * u

# simulation parameters
t_k = 923
dt = 1.4281e-7
dx = 0.0002118
print(f"Simulation timestep is {dt} s and cell size is {dx} m")

LiVHS = VHSMolecule('LiVHS', mLi, 5.418e-10, 0.948, 850)

n = 1.66934e+21 
mfp = LiVHS.mean_free_path(n, t_k)
mft = LiVHS.mean_free_time(n, t_k)
print(f"At {t_k} K, saturated lithium density is {n:.4} m^{-3},\n mean free path is {mfp:.4} m and mean free time is {mft:.4} s")
dt_n = dt / mft
print(f"Normalized timestep is {dt_n:.4}")
dx_n = dx / mfp
print(f"Normalized cell size is {dx_n:.4}")
err_visc = naive_viscosity_error_dt_dx(dt_n, dx_n)
print(f"Viscosity error factor is ~ {err_visc:.5}")
err_cond = naive_conductivity_error_dt_dx(dt_n, dx_n)
print(f"Thermal conductivity error factor is ~{err_cond:.4}")
n_c = 1000
print(f"\nParticles per cell: {n_c}")
finite_nc_error_enhancement = (K_wall_discretization_error(dt_n, dx_n, n_c) - 1) / (K_wall_discretization_error(dt_n, dx_n, np.inf) - 1)
print(f"Estimated error enhancement factor for finite numbers of particles per cell: ~{finite_nc_error_enhancement:.4}")
err_visc_finite = 1 + (err_visc - 1) * finite_nc_error_enhancement
print(f"Resulting viscosity error: {err_visc_finite:.5}")
