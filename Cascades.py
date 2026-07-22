import numpy as np
from enum import IntEnum, auto
import pandas as pd
import matplotlib.pyplot as plt
import uproot

class Cascades:
    def __init__(self, settings_data):
        self.Setting = settings_data["Setting"]
        self.settings = settings_data["settings"]
        self.setting_definer = settings_data["setting_definer"]
        self.value_setting = settings_data["value_setting"]

        self.rainier_sample_folder = settings_data["rainier_sample_folder"]
        self.this_dir = settings_data["this_dir"]
        self.std_path = settings_data["std_path"]

    def get_cascades(self, save_path = None, file_name = "Run0001.root"):
        if save_path == None:
            save_path = self.rainier_sample_folder
        with uproot.open(save_path / file_name) as file:
            cascades = file["tree;1"]["Egs"].array(library="np")
            return cascades

    def from_cascades_get_binned_events(self, *, bin = None, E_start = 0, E_end = 7):
        if bin == None:
            bin = self.settings[self.Setting.g_nConEBin]
        bin_counter = np.zeros(bin)
        E_step = (E_end - E_start) / bin
        energy = np.zeros(bin)
        for k in range(bin):
            energy[k] = E_step * k
        cascades = self.get_cascades()
        for casc in cascades:
            ex = sum(casc)
            index = int(np.floor(ex/E_step))
            bin_counter[index] += 1
        return energy, bin_counter