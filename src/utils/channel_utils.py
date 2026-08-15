import numpy as np
from scipy.special import erfc

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def to_linear(db):
    return 10 ** (db / 10.0)

def sample_polar(radius_range, angle_range, rng):
    r = rng.uniform(*radius_range)
    theta = rng.uniform(*angle_range)
    return np.array([r * np.cos(theta), r * np.sin(theta)])

def bearing(from_pos, to_pos):
    from_pos = np.asarray(from_pos, dtype=float)
    to_pos = np.asarray(to_pos, dtype=float)

    delta = to_pos - from_pos

    return float(np.arctan2(delta[1], delta[0]))

def distance(from_pos, to_pos):
    return np.linalg.norm(to_pos - from_pos)

def to_db(num_linear: float) -> float:
  return 10 * np.log10(num_linear)
