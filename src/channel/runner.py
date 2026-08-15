from src.channel.channel_model import get_hybird_channel_model, get_path_loss_linear
from src.utils.channel_utils import to_linear
import numpy as np

#test file to test with H

def run_channel_model(rx:int, tx:int, angle_rx: float, angle_tx:float, rician_factor: float, d : float, eta : float, rng: np.random.Generator) -> np.ndarray:
  beta = get_path_loss_linear(d = d, eta = eta)
  rician_factor = to_linear(rician_factor)
  return get_hybird_channel_model(
    rx_elements=rx,
    tx_elements=tx,
    rician_factor=rician_factor,
    beta=beta,
    angle_rx_rad=angle_rx,
    angle_tx_rad=angle_tx,
    rng=rng
  )