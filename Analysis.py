import numpy as np
from enum import IntEnum, auto
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from Settings import Settings, Setting, parameter
from Func import func
from dataclasses import dataclass
from typing import Any
from functools import singledispatchmethod
from SimTool import Run
from Extractor import extract

class FaStep(IntEnum):
    fluct = auto()
    nld = auto()
    smoothing = auto()
    stationary = auto()
    autocorr = auto()
    comparison = auto()

@dataclass
class FluctuationAnalysisResult:
    fluct_energy: np.ndarray
    fluct_data: np.ndarray
    fine: np.ndarray
    rough: np.ndarray
    stationary: np.ndarray
    energy_range: np.ndarray
    nld: np.ndarray
    settings: dict[Setting, Any]

class FluctuationAnalysisPlot:
    def __init__(self):
        pass

    def plot_fluct_data(self, energy: list, fluct_data: list, *, show_full: bool = False) -> None:
        plt.plot(energy, fluct_data)
        plt.xlabel("E in MeV")
        plt.ylabel("Total Absorption Spectrum")
        plt.title("Input data")
        if not show_full:
            plt.ylim(0,np.percentile(fluct_data, [0,99.8])[1])

    def plot_nld(self, energy, nld, save_path, file_name = "myNLD.png", series = True):
        if not series:
            plt.figure()
        plt.plot(energy, nld)
        plt.xlabel("E in MeV")
        plt.ylabel("#levels per MeV")
        plt.title("Nuclear level density")
        plt.yscale("log")
        if not series:
            plt.savefig(save_path / Settings.folder_NLD_name / file_name, dpi=300)
            plt.close()

    def plot_smooth(self, energy, fluct_data, fine, rough, save_path, file_name = "mySmooth.png", series = True):
        if not series:
            plt.figure()
        plt.plot(energy, fluct_data)
        plt.plot(energy, fine)
        plt.plot(energy, rough)
        plt.xlabel("E in MeV")
        plt.ylabel("Coincidence")
        plt.title("Rough/fine smoothing")
        plt.yscale("log")
        if not series:
            plt.savefig(save_path / Settings.folder_smoothing_name / file_name, dpi=300)
            plt.close()

    def plot_stationary(self, energy, d_full, save_path, file_name, series = True):
        if not series:
            plt.figure()
        plt.xlabel("E in MeV")
        plt.ylabel("'relative fluctuation'")
        plt.title("Stationary spectrum")
        plt.plot(energy, d_full)
        if not series:
            plt.savefig(save_path / Settings.folder_stationary_name / file_name, dpi=300)
            plt.close()

    def plot_autocorrelation(self, eps_start, eps_end, eps_step, energy, full_data, save_path, file_name, series = True, *, interval_low = 5, interval_high = 6):
        epsilons = np.arange(eps_start,eps_end,eps_step)
        if not series:
            plt.figure()
        plt.plot(epsilons, [self.autocorr(eps, energy, full_data, interval_low, interval_high) for eps in epsilons])
        plt.xlabel("epsilon")
        plt.ylabel("Autocorr(epsilon)")
        plt.title("Autocorrelation function")
        plt.xlim(0,0.5) 
        if not series:
            plt.savefig(save_path / Settings.folder_autocorr_name / file_name, dpi=300)
            plt.close()

    def plot_comparison(self, E_int, fa_dens, nld_energy, nld, save_path, file_name, series = True, *, c_val = 1):
        (save_path/"Comparison").mkdir(exist_ok=True)
        if not series:
            plt.figure()
        plt.plot(nld_energy, nld)
        plt.scatter(E_int, fa_dens, color="purple")
        plt.yscale("log")
        plt.title("Original NLD and extracted NLD")
        plt.xlabel("E in MeV")
        plt.ylabel("#levels per MeV")
        if not series:
            plt.savefig(save_path / Settings.folder_comparison_name / file_name, dpi=300)
            plt.close()

    def plot_comparison2(self, E_int, fa_dens, nld_energy, nld, save_path, file_name, series = True, *, c_val = 1):
        (save_path/"Comparison").mkdir(exist_ok=True)
        if not series:
            plt.figure()
        plt.plot(nld_energy, [func.rho(e) for e in nld_energy])
        plt.scatter(E_int, fa_dens, color="purple")
        plt.yscale("log")
        plt.title("Original NLD and extracted NLD")
        plt.xlabel("E in MeV")
        plt.ylabel("#levels per MeV")
        if not series:
            plt.savefig(save_path / Settings.folder_comparison_name / file_name, dpi=300)
            plt.close()

    def plot_comparison3(self):
        pass

    def helper_seriesplot(self, fa: FluctuationAnalysisResult, fa_step: FaStep, *, save_path: Path = None, file_name: str = None) -> None:
        if save_path == None:
            save_path = Settings.std_path
        if file_name == None:
            file_name = "unnamed_fluct_plot"
        fluct_energy = fa.fluct_energy
        fluct_data = fa.fluct_data
        fine = fa.fine
        rough = fa.rough
        stationary = fa.stationary
        energy_range = fa.energy_range
        extract_nld = fa.nld
        match fa_step:
            case FaStep.fluct:
                self.plot_fluct_data(fluct_energy, fluct_data)
            # case FaStep.nld:
            #     self.plot_nld(nld_energy, nld, save_path, file_name)
            case FaStep.smoothing:
                #self.plot_smooth(energy[start:finish], fluct_data[start:finish], fine[start:finish], rough[start:finish], save_path, file_name)
                self.plot_smooth(fluct_energy, fluct_data, fine, rough, save_path, file_name)
            case FaStep.stationary:
                self.plot_stationary(fluct_energy, stationary, save_path, file_name)
            case FaStep.autocorr:
                pass
            case FaStep.comparison:
                self.plot_comparison2(energy_range, extract_nld, energy_range, extract_nld, save_path, file_name)


    def create(self, fa_collection: list[FluctuationAnalysisResult], *, save_path: Path = None, file_name: str = "test", print_nld = True, print_smooth = True, print_stationary = True, print_autocorrelation = True, print_comparison = True) -> None:
        fa_collection_array = fa_collection
        if save_path == None:
            save_path = Settings.std_path
        (save_path/Settings.folder_fluct_name).mkdir(exist_ok=True, parents=True)
        (save_path/Settings.folder_NLD_name).mkdir(exist_ok=True, parents=True)
        (save_path/Settings.folder_smoothing_name).mkdir(exist_ok=True, parents=True)
        (save_path/Settings.folder_stationary_name).mkdir(exist_ok=True, parents=True)
        (save_path/Settings.folder_autocorr_name).mkdir(exist_ok=True, parents=True)
        (save_path/Settings.folder_comparison_name).mkdir(exist_ok=True, parents=True)
        for step in FaStep:
            if step == FaStep.autocorr: #TEMPORARY
                continue
            plt.figure()
            for fa in fa_collection_array:
                self.helper_seriesplot(fa, step, save_path=save_path, file_name=file_name)
            plt.savefig(save_path / FluctuationAnalysis.fa_step_to_folder_name[step] / (file_name+".png"), dpi = 300)
            plt.close()

class FluctuationAnalysis:
    fa_step_to_folder_name: dict[FaStep, str] = {
        FaStep.fluct : Settings.folder_fluct_name,
        FaStep.nld : Settings.folder_NLD_name,
        FaStep.smoothing : Settings.folder_smoothing_name,
        FaStep.stationary : Settings.folder_stationary_name,
        FaStep.autocorr : Settings.folder_autocorr_name,
        FaStep.comparison : Settings.folder_comparison_name
    }

    def __init__(self):
        self.plot = FluctuationAnalysisPlot()

    def get_smooth(self, fluct_data, deltaE, sigma_fine, sigma_rough):
        np_data = np.array(fluct_data)
        fine = gaussian_filter(np_data, sigma=sigma_fine/deltaE)
        rough = gaussian_filter(np_data, sigma=sigma_rough/deltaE)
        return fine, rough

    def get_interval2(self, energy, myData, lower, upper):
        mask = (energy >= lower) & (energy <= upper)
        return myData[mask]
    
    def autocorr(self, x, energy, full_data, lower, upper, *, print_cut = True):
        h1 = self.get_interval(energy, full_data, lower, upper)
        h2 = self.get_interval(energy, full_data, lower+x, upper+x)
        c1 = 0
        c2 = 0
        while h1.size > h2.size:
            c1 += 1
            h1 = h1[:-1]
        while h1.size < h2.size:
            c2 += 1
            h2 = h2[:-1]
        if print_cut:
            if c1 > 1:
                print("Autocorrelation 1: cut", c1, " out of ",len(h1))
            if c2 > 1:
                print("Autocorrelation 1: cut", c2, " out of ",len(h2))
        return np.mean(h1*h2)/(np.mean(h1)*np.mean(h2))

    def autocorr_zero(self, energy, stationary, Emin, Emax):
        d = self.get_interval2(energy, stationary, Emin, Emax)
        return np.mean(d**2) / (np.mean(d))**2

    def get_avg_level_spacing(self, energy, stationary, Emin, Emax, sigma):
        alpha = parameter[Setting.alpha_parameter]
        c_min_1 = self.autocorr_zero(energy, stationary, Emin, Emax) - 1
        return c_min_1 * 2 * sigma * np.sqrt(np.pi) / alpha

    def get_level_density(self, energy, stationary, Emin, Emax, sigma):
        return 1 / self.get_avg_level_spacing(energy, stationary, Emin, Emax, sigma)

    def get_energy_width(self, energy):
        return np.mean(np.diff(energy))

    @singledispatchmethod
    def fluctuation_analysis(self, energy: list, fluct_data: list) -> FluctuationAnalysisResult:
        E_start = parameter[Setting.analysis_E_start]
        E_end = parameter[Setting.analysis_E_end]
        sliding_window_shift = parameter[Setting.sliding_window_E_shift]
        E_step = parameter[Setting.analysis_E_step]

        #calculate smoothing energies sigma
        deltaE = parameter[Setting.exp_resolution]
        bin_width = self.get_energy_width(energy)
        #print("Energy bin width: ", bin_width)
        sigma_fine = deltaE/2
        sigma_rough = 3*sigma_fine
        fine, rough = self.get_smooth(fluct_data, deltaE, sigma_fine, sigma_rough)
        stationary = fine/rough

        energy_range = np.arange(E_start + E_step/2, E_end - E_step/2, sliding_window_shift)
        nld_list = []
        for E_val in energy_range:
            nld_list.append(
                self.get_level_density(energy, stationary, E_val - E_step/2, E_val + E_step/2, sigma_fine)
            )
        return FluctuationAnalysisResult(energy,fluct_data,fine,rough,stationary,energy_range,nld_list,parameter)

    @fluctuation_analysis.register
    def _(self, run: Run):
        energy, fluct_data_dict = extract.get_fluct_data_spin(run,plot=False)
        fluct_data = fluct_data_dict[-1]
        return self.fluctuation_analysis(energy, fluct_data)

    def iterate(self, energy, fluct_data, nld_energy, nld, param, param_range):
        param0 = parameter[param]
        result = []
        for k in range(len(param_range)):
            print(Setting(param).name+": "+str(param_range[k]))
            parameter[param] = param_range[k]
            result.append(self.fluctuation_analysis(energy, fluct_data, nld_energy, nld))
        parameter[param] = param0
        return result

    def plot_iteration(self, fa_collection):
        for k in range(len(fa_collection)):
            self.plot_fluctuation_analysis(fa_collection[k],file_name=str(k))




fluc = FluctuationAnalysis()