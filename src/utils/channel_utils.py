import numpy as np

def to_linear(num_db: float) -> float:
  return 10 ** (num_db/ 10)

def to_db(num_linear: float) -> float:
  return 10 * np.log10(num_linear)