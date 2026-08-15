from dataclasses import dataclass
import numpy as np


@dataclass
class PanelChannels:
    G: np.ndarray                       # Channel for base station to panel
    b: np.ndarray                       # Channel for panel to target
    f_by_user: dict                     # Channel for panel to user

@dataclass
class UserChannels:
    hdk: np.ndarray                     # Channel for base station to user

@dataclass 
class TargetChannels:
    rtt: np.ndarry
