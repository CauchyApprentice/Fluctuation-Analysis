import numpy as np
from enum import IntEnum, auto
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

class FluctuationAnalysis:
    def __init__(self, settings_data):
        self.Setting = settings_data["Setting"]
        self.settings = settings_data["settings"]
        self.setting_definer = settings_data["setting_definer"]
        self.value_setting = settings_data["value_setting"]

        self.rainier_sample_folder = settings_data["rainier_sample_folder"]
        self.this_dir = settings_data["this_dir"]
        self.std_path = settings_data["std_path"]


    def plot_nld(self, energy, nld, save_path, file_name = "myNLD.png"):
        plt.figure()
        plt.plot(energy, nld)
        plt.xlabel("E in MeV")
        plt.ylabel("Levels in Energy Bin")
        plt.title("NLD für alle spins")
        plt.yscale("log")
        plt.savefig(save_path / "NLD_input" / file_name, dpi=300)
        plt.close()

    def plot_smooth(self, energy, nld, fine, rough, save_path, file_name = "mySmooth.png"):
        plt.figure()
        plt.plot(energy, nld)
        plt.plot(energy, rough)
        plt.plot(energy, fine)
        plt.xlabel("E in MeV")
        plt.ylabel("Levels in Energy Bin")
        plt.title("Rough/fine smoothing")
        plt.yscale("log")
        plt.savefig(save_path / "Smoothing" / file_name, dpi=300)
        plt.close()

    def plot_stationary(self, energy, d_full, save_path, file_name):
        plt.figure()
        plt.plot(energy, d_full)
        plt.savefig(save_path / "Stationary" / file_name, dpi=300)
        plt.close()

    def plot_autocorrelation(self, eps_start, eps_end, eps_step, energy, full_data, save_path, file_name, *, interval_low = 5, interval_high = 6):
        epsilons = np.arange(eps_start,eps_end,eps_step)
        plt.figure()
        plt.title("Autocorrelation function for 5MeV-6MeV")
        plt.plot(epsilons, [self.autocorr(eps, energy, full_data, interval_low, interval_high) for eps in epsilons])
        plt.xlim(0,0.5) 
        plt.savefig(save_path / "Autocorrelation" / file_name, dpi=300)
        plt.close()

    def plot_comparison(self, E_int, fa_dens, energy, nld, save_path, file_name, *, new_figure = True, save_fig = True, c_val = 1):
        (save_path/"Comparison").mkdir(exist_ok=True)
        if new_figure:
            plt.figure()
        plt.plot(energy, nld)
        plt.scatter(E_int, fa_dens, color=plt.cm.viridis(c_val))
        plt.yscale("log")
        plt.title("Fluctuation Analysis compared to original level scheme")
        plt.xlabel("E in MeV")
        plt.ylabel("Levels per MeV")
        if save_fig:
            plt.savefig(save_path / "Comparison" / file_name, dpi=300)
        if new_figure:
            plt.close()

    def plot_fluctuation_analysis(self, energy, result, *, save_path = None, file_name = "test", print_nld = True, print_smooth = True, print_stationary = True, print_autocorrelation = True, print_comparison = True):
        if save_path == None:
            save_path = self.std_path
        save_path.mkdir(exist_ok=True)
        (save_path/"NLD_input").mkdir(exist_ok=True)
        (save_path/"Smoothing").mkdir(exist_ok=True)
        (save_path/"Stationary").mkdir(exist_ok=True)
        (save_path/"Autocorrelation").mkdir(exist_ok=True)
        (save_path/"Comparison").mkdir(exist_ok=True)
        nld = result["nld"]
        fine = result["fine"]
        rough = result["rough"]
        stationary = result["stationary"]
        E_int = result["fin_energy_int"]
        fa_dens = result["fa_dens"]
        if print_nld:
            self.plot_nld(energy, nld, save_path, file_name)
        if print_smooth:
            self.plot_smooth(energy, nld, fine, rough, save_path, file_name)
        if print_stationary:
            self.plot_stationary(energy, stationary, save_path, file_name)
        if print_autocorrelation:
            self.plot_autocorrelation(0, 0.5, 0.001, energy, stationary, save_path, file_name)
        if print_comparison:
            self.plot_comparison(E_int, fa_dens, energy, nld, save_path, file_name)

    def plot_fluctuation_analysis_all_spins(self, result, save_path, *, print_nld = True, print_smooth = True, print_stationary = True, print_autocorrelation = True, print_comparison = True):
        energy = result["energy"]
        for key in result.keys():
            if key != "energy":
                self.plot_fluctuation_analysis(energy, result[key], save_path, file_name = key, print_nld=print_nld, print_smooth=print_smooth, print_stationary=print_stationary, print_autocorrelation=print_autocorrelation, print_comparison=print_comparison)

    def get_nld(self, nld_data):
        return nld_data

    def get_smooth(self, nld):
        pd_data = pd.Series(nld)
        fine = pd_data.rolling(window=self.settings[self.Setting.fine_binning], center=True).mean()
        rough = pd_data.rolling(window=self.settings[self.Setting.rough_binning], center=True).mean()
        return fine, rough

    def get_interval(self, energy, myData, lower, upper):
        intervalled_data = []
        for k in range(myData.size):
            if energy[k] >= lower and energy[k] <= upper:
                intervalled_data.append(myData[k])
        intervalled_data = np.array(intervalled_data)
        return intervalled_data
    
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

    def calc_lvl_dens(self, E, E_step, energy, full_data, sigma = None, alpha = None):
        if sigma == None:
            sigma = self.settings[self.Setting.sigma_parameter]
        if alpha == None:
            alpha = self.settings[self.Setting.alpha_parameter]
        return 1/ ((self.autocorr(0, energy, full_data, E, E + E_step)-1)*2*sigma*np.sqrt(np.pi)/alpha)

    def get_fa_density(self, E_start, E_end, E_step, energy, full_data):
        E_int = np.arange(E_start, E_end, E_step)
        n = len(E_int)
        result = np.zeros((2,n))
        for k in range(n):
            result[0][k] = E_int[k]
            result[1][k] = self.calc_lvl_dens(E_int[k], E_step, energy, full_data)
        return result

    def get_level_data(self, run):
        cut = run.find("More levels exist at higher spins")
        #print(run[cut:])
        cut2 = run[cut:].find("E(MeV)")
        first = run[cut+cut2+10:]
        cut3 = first.find("Total Number of Levels")
        second = first[:cut3]
        return second

    def get_level_arrays(self, text):
        energy = []
        spin_array = []
        for k in range(len(text)):
            ch = text[k]
            if ch != '.':
                continue
            a = 1
            while text[k-(a+1)].isdigit():
                a = a+1
            num = float(text[k-a:k+4])
            energy.append(num)
            l = 20
            b = 8
            spin_array_one_energy = [[] for x in range(20)]
            while l > 0:
                if text[k+b] == " ":
                    b = b+1
                c = 1
                #print("b:",b)
                while text[k+b+c] != "|":
                    c = c+1
                #print("c:",c)
                spin_val = text[k+b:k+b+c]
                #print(spin_val)
                spin_val = spin_val.replace("\n", "")
                spin_array_one_energy[l-1] = int(spin_val)
                #print("spin"+str(20-l)+":"+spin_val)
                #print("Spin"+str(19-l)+":"+text[k+b:k+b+c+2])
                b = b+c+1
                l = l-1
            #print(spin_array_one_energy)
            spin_array.append(spin_array_one_energy)
        energy = np.array(energy)
        spin_array = np.array(spin_array)
        spin_array = spin_array.T
        return (energy,spin_array)

    def replace_val(self, text, definer, new_val):
        pos = text.find(definer)
        start_val = pos + len(definer)
        end_val = start_val
        while text[end_val] != ";":
            end_val = end_val + 1
        val = text[start_val:end_val]
        if val != str(new_val):
            print("(changed) "+text[pos:start_val] + str(new_val))
            return text[:start_val] + str(new_val) + text[end_val:]
        else:
            print(text[pos:end_val])
            return text

    def apply_settings(self):
        with open(self.rainier_sample_folder / "settings.h", "r") as f:
            text = f.read()
        with open(self.rainier_sample_folder / "settings.h", "w") as f:
            for key in self.settings.keys():
                if key in self.value_setting:
                    text = self.replace_val(text, self.setting_definer[key], self.settings[key])
            f.write(text)

    def fluctuation_analysis(self, energy, nld_data):
        nld = self.get_nld(nld_data)
        fine, rough = self.get_smooth(nld)
        d_full = fine/rough
        E_int, fa_dens = self.get_fa_density(self.settings[self.Setting.E_start], self.settings[self.Setting.E_end], self.settings[self.Setting.E_step], energy, d_full)
        return {
            "nld" : nld,
            "fine" : fine,
            "rough" : rough,
            "stationary" : d_full,
            "fin_energy_int" : E_int,
            "fa_dens" : fa_dens
        }

    def run_simulation(self):
        self.apply_settings()
        return subprocess.run(["cmd", "/c", "root", r"C:\RAINIER\RAINIER.C"], capture_output=True, text=True, cwd=r"C:\RAINIER\sample_folder").stdout

    def fluctuation_analysis_all_spins(self, run):
        energy, spin_array = self.get_level_arrays(self.get_level_data(run))
        result = {"energy" : energy}
        for s in range(spin_array.shape[0]):
            result[str(s)] = self.fluctuation_analysis(energy, spin_array[s])
        return result

    def sliding_window_spin(self, run, E_shift, s):
        folder_path = self.std_path / "sliding_window"
        folder_path.mkdir(exist_ok=True)
        E_start_0 = self.settings[self.Setting.E_start]
        E_end_0 = self.settings[self.Setting.E_end]
        N = int(self.settings[self.Setting.E_step]/E_shift)
        plt.figure()
        for k in range(N):
            self.settings[self.Setting.E_start] = E_start_0 + k * E_shift
            self.settings[self.Setting.E_end] = E_end_0 + k * E_shift
            fa = self.fluctuation_analysis(run)
            E_int = fa[str(s)]["fin_energy_int"]
            fa_dens = fa[str(s)]["fa_dens"]
            energy = fa["energy"]
            nld = fa[str(s)]["nld"]
            self.plot_comparison(E_int, fa_dens, energy, nld, folder_path, "sliding"+str(s), new_figure = False, save_fig = False, c_val = k/N)
        plt.savefig(folder_path / ("sliding"+str(s)), dpi = 300)
        plt.close()
        self.settings[self.Setting.E_start] = E_start_0
        self.settings[self.Setting.E_end] = E_end_0
