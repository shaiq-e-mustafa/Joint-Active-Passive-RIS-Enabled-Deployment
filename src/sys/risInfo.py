from dataclasses import dataclass
import numpy as np


@dataclass
class PanelState:
    active: bool
    a: int = 0                          # selection indicator
    phases: np.ndarray = None           # radians, one per element
    gains: np.ndarray = None            # amplification per element, if active
    noise: int = 0
    noise_2: int = 0

    @property
    def phi_vec(self) -> np.ndarray:
        """Diagonal of Phi_i as a length-L vector (gain * unit-modulus phase). Use this instead of
        phi: the dense L x L diagonal matrix costs O(L^2) memory and O(L^3) in Phi Phi^H."""
        ph = np.exp(1j * self.phases)
        if self.active:
            return self.gains.astype(complex) * ph
        return ph

    @property
    def theta(self) -> np.ndarray:
        return np.diag(np.exp(1j * self.phases))

    @property
    def phi(self) -> np.ndarray:
        return np.diag(self.phi_vec)
