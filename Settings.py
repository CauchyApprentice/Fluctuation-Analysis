from enum import IntEnum, auto
from pathlib import Path

class SettingsClass:
    class Setting(IntEnum):
        g_nReal = auto()
        g_nZ = auto()
        g_nAMass = auto()
        g_nConEBin = auto()
        g_nEvent = auto()
        alpha_parameter = auto()
        sigma_parameter = auto()
        fine_binning = auto()
        rough_binning = auto()
        E_step = auto()
        E_start = auto()
        E_end = auto()

    def __init__(self):
        self.settings = {
            self.Setting.g_nReal : 1,
            self.Setting.g_nZ : 32,
            self.Setting.g_nAMass : 76,
            self.Setting.g_nConEBin : 600,
            self.Setting.g_nEvent : "3e5",
            self.Setting.alpha_parameter : 0.45,
            self.Setting.sigma_parameter : 5,
            self.Setting.fine_binning : 4,
            self.Setting.rough_binning : 8,
            self.Setting.E_step : 0.5,
            self.Setting.E_start : 0,
            self.Setting.E_end : 7
        }

        self.setting_definer = {
            self.Setting.g_nReal : "const int g_nReal = ",
            self.Setting.g_nZ : "const int g_nZ = ",
            self.Setting.g_nAMass : "const int g_nAMass = ",
            self.Setting.g_nConEBin : "const int g_nConEBin = ",
            self.Setting.g_nEvent : "const int g_nEvent = "
        }

        self.value_setting = {
                self.Setting.g_nReal,
                self.Setting.g_nZ,
                self.Setting.g_nAMass,
                self.Setting.g_nConEBin,
                self.Setting.g_nEvent
        }

        self.rainier_sample_folder = Path(r"C:\RAINIER\sample_folder")
        self.this_dir = self.rainier_sample_folder
        self.std_path = self.this_dir / "fluctuation_analysis"

Settings = SettingsClass()