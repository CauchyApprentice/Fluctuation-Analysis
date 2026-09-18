import numpy as np
from enum import IntEnum, auto
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot
from Settings import Setting, Settings, parameter
from SimulationTool import Run

class Extractor:
    def __init__(self):
        pass

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
        energy_axis = run.level_data[0]
        bin_width = np.mean(np.diff(energy_axis))
        energy_axis = np.concatenate([
            [energy_axis[0] - bin_width / 2],
            energy_axis + bin_width / 2
        ])
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
        energy = 1/2 * (energy_edges[1:] + energy_edges[:-1])
        fluct_data_dict = {}
        spins = spin_edges[:-1]
        for k in range(len(spins)):
            spin = spins[k]
            fluct_data_dict[int(spin)] = counts[k]
        return (energy, fluct_data_dict)
        

extract = Extractor()