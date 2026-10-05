from src.utils.channel_utils import to_linear, distance
import numpy as np 
 
def occlusion_probability(d, d0=60.0, scale=20.0, p_max=0.6):
    """Logistic blockage probability: rises with distance, capped at p_max."""
    return p_max / (1 + np.exp(-(d - d0) / scale))

def is_blocked(d, rng, **kwargs):
    return rng.random() < occlusion_probability(d, **kwargs)


def los_probability_line_boolean(d, lambda_b, mean_L):
    """P(LoS) under a Boolean model of randomly oriented line-segment obstacles:
    centers ~ Poisson Point Process (density lambda_b per m^2), lengths ~ mean_L.
    Standard closed form (Bai & Heath-style blockage models used in 2024-2025 RIS
    literature): P_LoS(d) = exp(-2 * lambda_b * mean_L * d / pi).
    """
    return np.exp(-2.0 * lambda_b * mean_L * d / np.pi)


def calibrate_lambda_b(target_p_block, d_ref, mean_L):
    """Solve for the obstacle density lambda_b that makes the Line Boolean
    model produce `target_p_block` at reference distance `d_ref` (e.g. the
    mean BS-panel distance). Inverts los_probability_line_boolean().
    """
    p_los_target = 1.0 - target_p_block
    return -np.pi * np.log(p_los_target) / (2.0 * mean_L * d_ref)


def is_blocked_line_boolean(d, rng, lambda_b, mean_L):
    """Bernoulli draw for a single link under the Line Boolean model — same
    call pattern as is_blocked(), but using the calibrated closed form above
    instead of the ad hoc logistic occlusion_probability().
    """
    p_block = 1.0 - los_probability_line_boolean(d, lambda_b, mean_L)
    return rng.random() < p_block

def get_link_params(pos_a, pos_b, rng,
                     kappa_los_db: float = 5.0,
                     eta_los: float = 2.5,
                     eta_nlos: float = 3.5) -> tuple[float, float, bool]:

    d = distance(pos_a, pos_b)
    blocked = is_blocked(d, rng)
    if blocked:
        return 0.0, eta_nlos, blocked
    return to_linear(kappa_los_db), eta_los, blocked