from src.utils.channel_utils import to_linear
from src.utils.config import settings
import numpy as np


def _steering_vector(elements: int, angle_rad: float) -> np.ndarray:
  # Unit-MODULUS per element (not unit-NORM overall): each element reflects/
  # receives with gain 1, so ||a||^2 = elements, matching standard phased-
  # array/RIS array-gain physics (and Liaquat et al.'s explicit L-dependent
  # RISeffect terms). The previous 1/sqrt(elements) prefactor made ||a||=1
  # regardless of L, silently capping the LoS array gain at a constant --
  # see docs/phase_a_log.html Entry 11.
  q = np.arange(elements)
  return np.exp(1j * np.pi * q * np.sin(angle_rad))


def get_path_loss_linear(d: float, eta:float):
  # float(), not int(): int(-35.96) silently truncated beta_0 to -35 dB (Entry 12)
  return to_linear(float(settings.config.channel_model.beta_0_dB)) * ((d / float(settings.config.channel_model.d_0)) ** -eta)

def _get_H_bar(rx_elements: int, tx_elements: int, angle_rx_rad: float, angle_tx_rad: float = None) -> np.ndarray:
  a_rx = _steering_vector(rx_elements, angle_rx_rad)
  if tx_elements == 1:
    return a_rx.reshape(-1,1)
  
  a_tx = _steering_vector(tx_elements, angle_tx_rad)
  return np.outer(a_rx, a_tx.conj())

def _get_H_tilde(rx_elements: int, tx_elements: int, rng: np.random.Generator) -> np.ndarray:
  real = rng.standard_normal((rx_elements, tx_elements))
  imag = rng.standard_normal((rx_elements, tx_elements))
  return (real + 1j * imag) / np.sqrt(2.0)

def get_hybird_channel_model(rx_elements:int, tx_elements:int, rician_factor: int, beta:float, angle_rx_rad: float, angle_tx_rad: float = None, rng: np.random.Generator = None) -> np.ndarray:
  h_bar = _get_H_bar(rx_elements=rx_elements, tx_elements=tx_elements, angle_rx_rad=angle_rx_rad, angle_tx_rad=angle_tx_rad)
  h_tilde = _get_H_tilde(rx_elements=rx_elements, tx_elements=tx_elements, rng=rng)

  temp_term = (beta * rician_factor) / (rician_factor + 1)

  scale_h_bar_to_path_loss_power = np.sqrt(temp_term) * h_bar

  scale_h_tilde_to_path_loss_pwoer = (np.sqrt(beta / (rician_factor + 1))) * h_tilde

  return scale_h_bar_to_path_loss_power + scale_h_tilde_to_path_loss_pwoer


# ---------------------------------------------------------------------------
# Exact-distance (spherical-wavefront) channel -- see docs/phase_a_log.html Entry 12.
# The planar steering vector above assumes d >> 2D^2/lambda (Fraunhofer distance); panels with
# L > 32 at 1.5 GHz and 20-100 m ranges violate that. Here every LoS entry uses the true
# element-to-element distance for BOTH its phase exp(-jk d) and its path-loss amplitude.
# ---------------------------------------------------------------------------

def wavefront_is_exact() -> bool:
  return getattr(settings.config.channel_model, "wavefront", "exact") == "exact"


def wavelength() -> float:
  cm = settings.config.channel_model
  return float(cm.c) / float(cm.carrier_frequency)


def element_positions(center, n: int) -> np.ndarray:
  """ULA along y (the axis the planar model's sin(global bearing) implies), centered on `center`,
  half-wavelength spacing. n == 1 -> a single point (target / user)."""
  center = np.asarray(center, dtype=float)
  if n == 1:
    return center.reshape(1, 2)
  off = (np.arange(n) - (n - 1) / 2) * (wavelength() / 2)
  return np.stack([np.full(n, center[0]), center[1] + off], axis=1)


def get_hybrid_channel_exact(rx_center, n_rx: int, tx_center, n_tx: int,
                             kappa: float, eta: float, rng: np.random.Generator) -> np.ndarray:
  """(n_rx x n_tx) hybrid Rician channel from exact element-to-element distances.
  LoS entry = sqrt(beta(d)) sqrt(K/(K+1)) exp(-j 2 pi d / lambda); NLoS = sqrt(beta(d)/(K+1)) CN(0,1)."""
  rx = element_positions(rx_center, n_rx)
  tx = element_positions(tx_center, n_tx)
  d = np.linalg.norm(rx[:, None, :] - tx[None, :, :], axis=2)
  amp = np.sqrt(get_path_loss_linear(d, eta))
  los = np.sqrt(kappa / (kappa + 1)) * np.exp(-1j * 2 * np.pi * d / wavelength())
  nlos = np.sqrt(1.0 / (kappa + 1)) * _get_H_tilde(n_rx, n_tx, rng)
  return amp * (los + nlos)
