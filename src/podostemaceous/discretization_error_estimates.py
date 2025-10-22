import numpy as np
from numpy import pi, sqrt

def K_wall_discretization_error(dt, dx, n_c):
    """
    Best fit to error ratio of thermal conductivity for finite dt, dx, n_c.

    Parameters:
        dt: normalized timestep
        dx: normalized cell size
        n_c: particles per cell
    
    Returns:
        K_wall(dt, dx, n_c) / K(0, 0, infty)

    Rader, D. J., M. A. Gallis, J. R. Torczynski, and W. Wagner.
    “Direct Simulation Monte Carlo Convergence Behavior of
    the Hard-Sphere-Gas Thermal Conductivity for Fourier Heat Flow.”
    Physics of Fluids 18, no. 7 (July 2006): 077102.
    https://doi.org/10.1063/1.2213640.

    Equation (23)
    """
    ratio = (1.0001 + 0.0287 * dt ** 2 + 0.0405 * dx ** 2
                   - 0.0009 * dx ** 4 - 0.016 * dt ** 2 * dx ** 2
                   + 0.0081 * dt ** 4 * dx **2
                   + (1 / n_c) * (-0.083 + 1.16 * dx
                                         - 0.220 * dx ** 2
                                         + 1.56 * dt ** 2 
                                         - 2.55 * dt ** 2 * dx
                                         + 1.14 * dt ** 2 * dx ** 2
                                         - 0.92 * dt ** 3
                                         + 1.91 * dt ** 3 * dx 
                                         - 0.94 * dt ** 3 * dx ** 2
                                         )
                   + (1 / n_c ** 2) * 0.95 * dt ** 2
                   )
    return ratio

def viscosity_discretization_error_conservative(dt, dx, n_c):
    """
    Conservative estimate: 2.5 * error in K(dt, dx, n_c)

    The exact expressions for viscosity error due to dt and dx 
    have coefficients that are 2.25 (dt) and 2.5 (dx) times larger than 
    the coefficients in the thermal conductivity errors.
    Thus, I conservatively estimate that the whole error 
    (including with the n_c term) is 2.5 times larger.

    Parameters:
        dt: normalized timestep
        dx: normalized cell size
        n_c: particles per cell

    Returns:
        viscosity(dt, dx, n_c) / viscosity(0, 0, infty)

 
    See the function K_wall_discretization_error.
    """
    k_err_ratio = K_wall_discretization_error(dt, dx, n_c)
    visc_err_ratio = 1 + (k_err_ratio - 1) * 2.5
    return visc_err_ratio

def viscosity_discretization_error_less_conservative(dt, dx, n_c):
    """
    Less conservative estimate: error ratio = K(dt * \sqrt{2.25}, dx * \sqrt{2.5}, n_c)

    The exact expressions for viscosity error due to dt and dx 
    have coefficients that are 2.25 (dt) and 2.5 (dx) times larger than 
    the coefficients in the thermal conductivity errors.
    If we assume that the n_c term has unchanged scaling, then we can reuse 
    the K_wall error ratio.

    Parameters:
        dt: normalized timestep
        dx: normalized cell size
        n_c: particles per cell

    Returns:
        viscosity(dt, dx, n_c) / viscosity(0, 0, infty)

    See the function K_wall_discretization_error.
    """
    visc_err_ratio = K_wall_discretization_error(sqrt(2.25) * dt, sqrt(2.5) * dx, n_c)
    return visc_err_ratio

def viscosity_error_timestep(dt):
    """
    Error ratio of viscosity for finite timestep. 
    ...for Hard-Sphere molecules. Calculated using a Green-Kubo analysis.

    Parameters:
        dt: normalized timestep

    Hadjiconstantinou, Nicolas G.
    "Analysis of Discretization in the Direct Simulation Monte Carlo."
    Physics of Fluids 12, no. 10 (2000): 2634.
    https://doi.org/10.1063/1.1289393.

    Equation (19)
    """
    ratio = 1 + 32 / (150 * pi) * dt ** 2
    return ratio

def conductivity_error_timestep(dt):
    """
    Error ratio of thermal conductivity for finite timestep. 
    ...for Hard-Sphere molecules. Calculated using a Green-Kubo analysis.

    Parameters:
        dt: normalized timestep

    Hadjiconstantinou, Nicolas G.
    "Analysis of Discretization in the Direct Simulation Monte Carlo."
    Physics of Fluids 12, no. 10 (2000): 2634.
    https://doi.org/10.1063/1.1289393.

    Equation (20)
    """
    ratio = 1 + 64 / (675 * pi) * dt ** 2
    return ratio

def diffusion_error_timestep(dt):
    """
    Error ratio of self-diffusion for finite timestep. 
    ...for Hard-Sphere molecules. Calculated using a Green-Kubo analysis.

    Parameters:
        dt: normalized timestep

    Hadjiconstantinou, Nicolas G.
    "Analysis of Discretization in the Direct Simulation Monte Carlo."
    Physics of Fluids 12, no. 10 (2000): 2634.
    https://doi.org/10.1063/1.1289393.

    Equation (22)
    """
    ratio = 1 + 4 / (27 * pi) * dt ** 2
    return ratio

def viscosity_error_cell_size(dx):
    """
    Error ratio of viscosity for finite cell size.
    ...for Hard-Sphere molecules. Calculated using a Green-Kubo analysis.

    Parameters:
        dx: normalized cell size

    Alexander, Francis J., Alejandro L. Garcia, and Berni J. Alder.
    "Cell Size Dependence of Transport Coefficients in Stochastic Particle Algorithms."
    Physics of Fluids 10, no. 6 (June 1998): 1540–42. 
    https://doi.org/10.1063/1.869674.

    Equation (8) (as corrected in erratum.)
    """
    ratio = 1 + 16/(45 * pi) * dx ** 2
    return ratio

def conductivity_error_cell_size(dx):
    """
    Error ratio of thermal conductivity for finite cell size.
    ...for Hard-Sphere molecules. Calculated using a Green-Kubo analysis.

    Parameters:
        dx: normalized cell size

    Alexander, Francis J., Alejandro L. Garcia, and Berni J. Alder.
    "Cell Size Dependence of Transport Coefficients in Stochastic Particle Algorithms."
    Physics of Fluids 10, no. 6 (June 1998): 1540–42. 
    https://doi.org/10.1063/1.869674.

    Equation (9)
    """
    ratio = 1 + 32/(225 * pi) * dx ** 2
    return ratio

def naive_viscosity_error_dt_dx(dt, dx):
    """
    Combined viscosity error ratio from finite dx and dt.

    Parameters:
        dt: normalized timestep
        dx: normalized cell size

    Returns:
        eta / eta_{dt->0, dx->0, n_c->\inf}
    """
    return (1 + viscosity_error_timestep(dt) - 1
              + viscosity_error_cell_size(dx) - 1)

def naive_conductivity_error_dt_dx(dt, dx):
    """
    Combined thermal conductivity error ratio from finite dx and dt.

    Parameters:
        dt: normalized timestep
        dx: normalized cell size

    Returns:
        kappa / kappa_{dt->0, dx->0, n_c->\inf}
    """
    return (1 + conductivity_error_timestep(dt) - 1
              + conductivity_error_cell_size(dx) - 1)

if __name__ == "__main__":
    print(K_wall_discretization_error((1/8),(1/3),np.inf))
    print(K_wall_discretization_error((1/8),(1/3),100))
    print(naive_conductivity_error_dt_dx((1/8),(1/3)))

