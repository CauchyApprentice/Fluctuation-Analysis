from Settings import Setting, Settings
import subprocess
from enum import IntEnum, auto
from pathlib import Path
import numpy as np
import time
import uproot

class SimTool:
    def __init__(self):
        self.run_path = Settings.std_path / "runs"
        self.settings_file_name = "settings"

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

    def simple_run(self, *, print_setting = True):
        Settings.apply_settings(print_setting=print_setting)
        return subprocess.run(["cmd", "/c", "root", r"C:\RAINIER\RAINIER.C"], capture_output=True, text=True, cwd=r"C:\RAINIER\sample_folder").stdout

    def run_simulation(self, *, save_path = None, file_name = None, print_setting = True):
        if file_name == None:
            file_name = "unnamed_run"
        if save_path == None:
            save_path = Settings.std_path
        else:
            save_path.mkdir(exist_ok = True, parents=True)
        current_run_folder = save_path / file_name
        current_run_folder.mkdir(exist_ok=True)
        run_text = self.simple_run(print_setting=print_setting)
        with open(current_run_folder / (file_name+".txt"), "w") as file:
            file.write(run_text)
        with open(current_run_folder / (self.settings_file_name+".txt"), "w") as file:
            for key in Settings.settings:
                file.write(Setting(key).name+" : "+str(Settings.settings[key])+"\n")
        run_path = Settings.rainier_sample_folder / "Run0001.root"
        run_path.rename(current_run_folder / (file_name+".root"))

    def sim_iterate(self, param, param_range, *, save_path = None):
        if save_path == None:
            save_path = Settings.std_path
        param0 = Settings.settings[param]
        for k in range(len(param_range)):
            print(Setting(param).name+": "+str(param_range[k]))
            Settings.settings[param] = param_range[k]
            start_time = time.perf_counter()
            self.run_simulation(save_path = save_path, file_name = str(Setting(param).name)+"_"+str(param_range[k]), print_setting=False)
            time_passed = time.perf_counter() - start_time
            print("time taken: "+str(round(time_passed)))
        Settings.settings[param] = param0

    def read_run(self, run_folder_path):
        collection = {
            "settings" : {},
            "level_data" : ([],[[]]),
            "root_tree" : "..."
                      }
        for file in run_folder_path.iterdir():
            if file.name[-3:] == "txt":
                with open(file) as f:
                    if file.name == self.settings_file_name+".txt":                
                        settings_text = f.readlines()
                        for entry in settings_text:
                            colon = entry.find(":")
                            param = entry[:colon-1]
                            param_value = entry[colon+2:-1]
                            collection["settings"][Setting[param]] = float(param_value)
                    else:
                        run_text = f.read()
                        collection["level_data"] = self.get_level_arrays(self.get_level_data(run_text))
            else:
                with uproot.open(file) as f:
                    collection["root_tree"] = f["tree;1"]
        return collection

    def read_folder(self, folder_path):
        runs = []
        for file in folder_path.iterdir():
            runs.append(self.read_run(file))
        return runs


#ok für nächstes mal: kombination aus simulations- und fluctuationsparametern

sim = SimTool()

#event_range = [0, 100, 200]
#runs_folder = Settings.std_path / "runs"
#sim.sim_iterate(Setting.g_nEvent, event_range, save_path=runs_folder)
#collection = sim.read_folder(runs_folder)