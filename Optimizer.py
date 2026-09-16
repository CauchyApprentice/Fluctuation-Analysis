from scipy.optimize import curve_fit
from Settings import parameter, Setting
import numpy as np
import matplotlib.pyplot as plt

class Optimizer:
    def nld_function(self, x):
        t0 = 0.48473
        e0 = -1.31817
        u = x-e0
        a = 14.58
        cutoff = 0.1461 * np.sqrt(a*u) * parameter[Setting.g_nAMass] ** (2/3)
        j = 1
        rf = (2*j+1) / (2*cutoff) * np.exp(-(j +1/2)**2 / (2*cutoff))
        pi = 1/2
        return 1/t0 * np.exp(u/t0) * rf * pi

    def nld_model(self, x, a, b, c):
        return a*np.exp(b*x) + c


    def get_fit(self, fa_collection):
        fa = fa_collection["fa"]
        energy_array = fa["fin_energy_int"]
        recon_nld_array = fa["fa_dens"]
        for k in range(len(energy)):
            pass
        p0 = [1.0, 2.0, 0.0]
        params, covar = curve_fit(self.nld_model, energy, recon_nld, p0)
        return [self.nld_model(x, *params) for x in energy]

    def fa_get_similarity():
        pass

optimizier = Optimizer()