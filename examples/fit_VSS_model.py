import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from podostemaceous.constants import u
from podostemaceous.molecule import VSSMolecule
from podostemaceous.fit_dsmc_vss import VSS_model_from_eta_and_D11

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

def D11_Fialho_1993_Table(TK):
    """
    Self-diffusion coefficient at 0.10 MPa for monoatomic lithium.

    Fialho, Paulo S., J.M.N.A. Fareleira, M.L.V. Ramires,
    and C.A. Nieto de Castro. "Thermophysical Properties
    of Alkali Metal Vapours, Part I.A." 
    Berichte Der Bunsen-Gesellschaft Fur Physikalische Chemie
    97, no. 11 (1993): 1487–92.

    Data from Table 3.
    """
    D_self_Fialho_1993_table = np.array([
        [700, 0.8885],
        [800, 1.1491],
        [900, 1.4393],
        [1000, 1.7589],
        [1100, 2.1077],
        [1200, 2.4859]])
    data = D_self_Fialho_1993_table
    interp = interp1d(data[:,0], data[:,1], kind='cubic', bounds_error=False)
    return 1e-4 * interp(TK)

mLi = 6.94 * u

visc_func = eta1_Bouledroua_Table_I
diff_func = D11_Fialho_1993_Table
mass = mLi
T_min, T_max = 700, 1000
vss = VSS_model_from_eta_and_D11(visc_func, diff_func, mass, T_min, T_max)

TK = np.linspace(700,1000, 100)
eta_vss = vss.viscosity(TK)

print("VSS Model is:")
print(vss)

eta_bouledroua_normalized = visc_func(TK) / eta_vss

plt.style.use('seaborn-colorblind')
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1,1,1)

ax.plot(TK, eta_bouledroua_normalized, label='Bouledroua 2005 (fit model)')

ax.set_xlabel('T / K')
ax.set_ylabel('$\eta / \eta_\mathrm{VSS}$')
plt.title('Viscosity of lithium vapor normalized to VSS model')
plt.annotate("$D_\mathrm{ref}$: " + str(vss.d_ref), xy=(0.3,0.4), xycoords='axes fraction')
plt.annotate("$\omega$: " + str(vss.omega), xy=(0.3,0.32), xycoords='axes fraction')
plt.annotate("$T_\mathrm{ref}$: " + str(vss.t_ref), xy=(0.3,0.24), xycoords='axes fraction')
plt.annotate(r"$\alpha$: " + str(vss.alpha), xy=(0.3,0.16), xycoords='axes fraction')

ax.set_xlim([700,1000])

ax.legend(loc=4)
plt.show()
