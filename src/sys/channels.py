from dataclasses import dataclass
import numpy as np


@dataclass
class PanelChannels:
    G: np.ndarray                       # Channel for base station to panel
    f_by_user: dict                     # Channel for panel to user
    b_by_target: dict                   # Channel for panel to target

@dataclass
class UserChannels:
    hdk: np.ndarray                     # Channel for base station to user

@dataclass
class TargetChannels:
    target_id: int
    b_i: np.ndarray
    T: np.ndarray = None
    rcs: float = None  
    J: np.ndarray = None
    SNR: np.ndarray = None