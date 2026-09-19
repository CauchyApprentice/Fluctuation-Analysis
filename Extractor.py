import numpy as np
from enum import IntEnum, auto
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot
from Settings import Setting, Settings, parameter
from SimTool import Run
from scipy.stats import norm

class Extractor:
    def __init__(self):
        pass

    def apply_resolution(self, energy: list[float], data: list[float]) -> list:
        h = np.mean(np.diff(energy)) #bin width
        bin_center = lambda i: energy[i] + h/2
        smeared = [0.0 for _ in data] #copying data; initializing smeared list
        sigma = lambda i: parameter[Setting.exp_resolution]
        for j in range(len(data)):
            weights = (
                norm.cdf(energy[1:], loc=bin_center(j), scale=sigma(j))
                -
                norm.cdf(energy[:-1], loc=bin_center(j), scale=sigma(j))
            )
            smeared += data[:-1][j] * weights
        return smeared
                

    def pop_EJ(self, run: Run, *, spin_spacing = 1):
        energy_bin = parameter[Setting.fluct_bin]
        tree = run.root_tree
        JI_int = tree["JI_int"]
        ExI = tree["ExI"]
        nPar = tree["nPar"]
        JI_int = JI_int.copy()
        for k in range(len(JI_int)):
            if nPar[k] == 0:
                JI_int[k] = -JI_int[k]
        spin_min = JI_int.min()
        spin_max = JI_int.max()
        spin_axis = np.arange(spin_min, spin_max + 2*spin_spacing, spin_spacing) #2*spin_spacing works even though 1* should work no?
        energy_min = 0
        energy_max = ExI.max() #get the global simulation E_max... whatever that is
        energy_axis = np.linspace(energy_min, energy_max, energy_bin + 1)
        return np.histogram2d(JI_int, ExI, bins=[spin_axis, energy_axis])

    def get_fluct_data_spin(self, run: Run, *, plot: bool = True) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        counts, spin_edges, energy_edges = self.pop_EJ(run)
        if plot:
            plt.figure()
            im = plt.pcolormesh(spin_edges, energy_edges, counts.T, shading="auto", norm=LogNorm())
            plt.colorbar(im, label="Number of events")
            plt.title("EJ counts")
            plt.xlabel("J[hbar]")
            plt.ylabel("Ex[MeV]")
        print(energy_edges.shape)
        energy = 1/2 * (energy_edges[1:] + energy_edges[:-1])
        print(energy.shape)
        fluct_data_dict = {}
        spins = spin_edges[:-1]
        for k in range(len(spins)):
            spin = spins[k]
            fluct_data_dict[int(spin)] = counts[k]
        return (energy, fluct_data_dict)

    def spectrum_smeared(self, run: Run, *, plot: bool = True) -> tuple[list, list]:
        energy, fluct_data_dict = self.get_fluct_data_spin(run, plot=plot)
        data = fluct_data_dict[-1]
        return energy, self.apply_resolution(energy, data)
        

extract = Extractor()