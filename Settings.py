from enum import IntEnum, auto
from pathlib import Path

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

settings = {
    Setting.g_nReal : 1,
    Setting.g_nZ : 32,
    Setting.g_nAMass : 76,
    Setting.g_nConEBin : 600,
    Setting.g_nEvent : "3e5",
    Setting.alpha_parameter : 0.45,
    Setting.sigma_parameter : 5,
    Setting.fine_binning : 4,
    Setting.rough_binning : 8,
    Setting.E_step : 0.5,
    Setting.E_start : 0,
    Setting.E_end : 7
}

setting_definer = {
    Setting.g_nReal : "const int g_nReal = ",
    Setting.g_nZ : "const int g_nZ = ",
    Setting.g_nAMass : "const int g_nAMass = ",
    Setting.g_nConEBin : "const int g_nConEBin = ",
    Setting.g_nEvent : "const int g_nEvent = "
}

value_setting = {Setting.g_nReal,
                 Setting.g_nZ,
                 Setting.g_nAMass,
                 Setting.g_nConEBin,
                 Setting.g_nEvent}

rainier_sample_folder = Path(r"C:\RAINIER\sample_folder")
this_dir = rainier_sample_folder
std_path = this_dir / "fluctuation_analysis"