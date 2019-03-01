import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from podostemaceous.constants import u
from podostemaceous.molecule import VHSMolecule
from podostemaceous.fit_dsmc_vhs import VHS_model_from_viscosity

# borrowed from my lithium data package
def eta1_Bouledroua_Table_I(TK):
    """
    Viscosity of lithium monomers as a function of temperature.

    Parameters:
        TK, temperature in Kelvin. 200 < TK < 2000.
    Returns: 
        viscosity in kg / (m s)

    Bouledroua, M, A Dalgarno, and R Côté.
    “Viscosity and Thermal Conductivity of Li, Na, and K Gases.”
    Physica Scripta 71, no. 5 (January 1, 2005): 519–22. 
    https://doi.org/10.1238/Physica.Regular.071a00519.

    Data from Tables I and IV. Eta in micropoise.
    """
    data_T = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    data_eta1 = [23, 49, 75, 100, 123, 144, 164, 184, 202, 221]
    
    interp = interp1d(data_T, data_eta1, kind='cubic', bounds_error=False)
    eta1 = 1e-7 * interp(TK)
    return eta1

mLi = 6.94 * u

visc_func_to_fit = eta1_Bouledroua_Table_I
mass = mLi
T_min, T_max = 700, 1000
vhs_b = VHS_model_from_viscosity(visc_func_to_fit, mass, T_min, T_max)

TK = np.linspace(700,1000, 100)
eta_vhs = vhs_b.viscosity(TK)

print("VHS Model is:")
print(vhs_b)

eta_bouledroua_normalized = visc_func_to_fit(TK) / eta_vhs

plt.style.use('seaborn-colorblind')
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1,1,1)

ax.plot(TK, eta_bouledroua_normalized, label='Bouledroua 2005 (fit model)')

ax.set_xlabel('T / K')
ax.set_ylabel('$\eta / \eta_\mathrm{VHS}$')
plt.title('Viscosity of lithium vapor normalized to VHS model')
plt.annotate("$D_\mathrm{ref}$: " + str(vhs_b.d_ref), xy=(0.3,0.4), xycoords='axes fraction')
plt.annotate("$\omega$: " + str(vhs_b.omega), xy=(0.3,0.32), xycoords='axes fraction')
plt.annotate("$T_\mathrm{ref}$: " + str(vhs_b.t_ref), xy=(0.3,0.24), xycoords='axes fraction')

ax.set_xlim([700,1000])

ax.legend(loc=4)
plt.show()
