# src/waveform/symbols.py
import numpy as np

def generate_qpsk_symbols(K: int, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    """
    Returns s of shape (K, n_symbols), unit power per symbol: E[|s_k|^2] = 1,
    E[s_k s_j*] = delta_kj across users (independent bit streams) — matches
    the unit-power symbol assumption under Eq. (1).
    """
    bits = rng.integers(0, 2, size=(K, n_symbols, 2))
    real = 1 - 2 * bits[:, :, 0]
    imag = 1 - 2 * bits[:, :, 1]
    s = (real + 1j * imag) / np.sqrt(2.0)
    return s