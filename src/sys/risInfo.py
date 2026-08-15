from dataclasses import dataclass
import numpy as np
from src.utils.config import settings

@dataclass
class PanelState:
    active: bool
    a: int = 0                          # selection indicator 
    phases: np.ndarray = None           # radians depending on number of elements
    gains: np.ndarray = None            # If active, gains are generated
    @property
    def theta(self) -> np.ndarray:
        return np.diag(np.exp(1j * self.phases))

    @property
    def phi(self) -> np.ndarray:
        if self.active:
            A = np.diag(self.gains.astype(complex))
            return A @ self.theta
        return self.theta
    
from dataclasses import dataclass
import numpy as np
from src.utils.config import settings

@dataclass
class PanelState:
    active: bool
    a: int = 0                          # selection indicator 
    phases: np.ndarray = None           # radians depending on number of elements
    gains: np.ndarray = None            # If active, gains are generated
    noise: int = 0

    @property
    def theta(self) -> np.ndarray:
        return np.diag(np.exp(1j * self.phases))

    @property
    def phi(self) -> np.ndarray:
        if self.active:
            A = np.diag(self.gains.astype(complex))
            return A @ self.theta
        return self.theta
    
 