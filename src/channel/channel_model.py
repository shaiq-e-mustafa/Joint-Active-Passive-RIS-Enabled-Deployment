from src.utils.channel_utils import to_db, to_linear
from src.utils.config import settings
import numpy as np


def _steering_vector(elements: int, angle_rad: float) -> np.ndarray:
  q = np.arange(elements)
  return (1.0 / np.sqrt(elements)) * np.exp(1j * np.pi * q * np.sin(angle_rad))


def get_path_loss_linear(d: float, eta:float):
  return to_linear(int(settings.config.channel_model.beta_0_dB)) * ((d / int(settings.config.channel_model.d_0)) ** -eta)

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
