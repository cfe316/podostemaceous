import numpy as np
from numpy import sqrt, pi
from constants import kB

class DSMCMolecule():
    
    def __init__(self, name, mass, diameter):
        self.name = name
        self.mass = mass
        
    def most_probable_speed(self, t_k):
        return sqrt(2 * kB * t_k / self.mass)

    def mean_speed(self, t_k):
        return sqrt(8 * kB * t_k / (pi * self.mass))

    def rms_speed(self, t_k):
        return sqrt(3 * kB * t_k / self.mass)


class VSSMolecule(DSMCMolecule):

    def __init__(self, name, mass, diameter, omega=1/2, t_ref=273, alpha=1):
        super().__init__(name, mass, diameter)
        self.d_ref = diameter
        self.omega = omega
        self.t_ref = t_ref
        self.alpha = alpha

    def schmidt_number(self):
        """
        Solved from Bird 2013, Ch 3, Equation 21.
        """
        a = self.alpha
        w = self.omega
        return 5 * (2 + a) / (3 * a * (7 - 2 * w))

    def viscosity(self, T):
        """
        Viscosity of a VSS gas as a function of temperature.

        Bird 2013, Chapter 3, Eq 19, solved for mu.
        """
        m = self.mass
        w = self.omega
        Tr = self.t_ref
        dr = self.d_ref
        a = self.alpha
        
        mu_ref_numerator = (5 * (1 + a) * (2 + a) * sqrt(kB * m * Tr / pi))
        mu_ref_denominator = 4 * a * dr **2 * (7 - 2 * w) * (5 - 2 * w)
        mu_ref = mu_ref_numerator / mu_ref_denominator
        
        mu = mu_ref * (T / Tr) ** w
        return mu


class VHSMolecule(VSSMolecule):
    """Special case of a VSS molecule with alpha=1"""

    def __init__(self, name, mass, diameter, omega=1/2, t_ref=273):
        super().__init__(name, mass, diameter, omega, t_ref, 1)

    def mean_free_path(self, n, T):
        """
        Mean free path of a VHS molecule.

        Bird 2013, Ch 2, Eq 34.
        """
        lambda_0 = 1 / (sqrt(2) * pi * self.d_ref **2 
                        * n * (self.t_ref / T) ** (self.omega - 0.5))
        return lambda_0

    def mean_free_time(self, n, t_k):
        return self.mean_free_path(n, t_k) / self.most_probable_speed(t_k)  

    def conductivity(self, T):
        """
        Thermal conductivity of a VHS gas a function of temperature.

        Parameter:
            T: Temperature in Kelvin

        Bird 2013, Chapter 2, Equation 44.
        """
        m = self.mass
        mu = self.viscosity(T)
        K = (15/4) * kB * mu / m
        return K

if __name__=="__main__":
    from podostemaceous.constants import u
    mol = VHSMolecule("LI", 6 * u, 1e-3)
    print(mol.mean_free_time(1000, 500))
