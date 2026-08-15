from src.utils.channel_utils import to_linear, distance
import numpy as np 
 
def occlusion_probability(d, d0=60.0, scale=20.0, p_max=0.6):
    """Logistic blockage probability: rises with distance, capped at p_max."""
    return p_max / (1 + np.exp(-(d - d0) / scale))

def is_blocked(d, rng, **kwargs):
    return rng.random() < occlusion_probability(d, **kwargs)

def get_link_params(pos_a, pos_b, rng,
                     kappa_los_db: float = 5.0,
                     eta_los: float = 2.5,
                     eta_nlos: float = 3.5) -> tuple[float, float, bool]:

    d = distance(pos_a, pos_b)
    blocked = is_blocked(d, rng)
    if blocked:
        return 0.0, eta_nlos, blocked
    return to_linear(kappa_los_db), eta_los, blocked