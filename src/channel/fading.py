from src.channel.channel_model import get_hybird_channel_model


def get_Gi(beta_g, kappa_g, g_rx_elements, g_tx_elements, angle_rx, angle_tx, rng):
    """BS<->panel, Eq.(13). g_rx_elements=L (panel), g_tx_elements=M (BS)."""
    return get_hybird_channel_model(g_rx_elements, g_tx_elements, kappa_g, beta_g,
                                     angle_rx, angle_tx, rng)
def get_fi(beta_f, kappa_f, f_rx_elements, angle_rx, angle_tx, rng):
    """panel<->user, Eq.(14). f_rx_elements=L (panel); user side is always 1."""
    return get_hybird_channel_model(f_rx_elements, 1, kappa_f, beta_f,
                                     angle_rx, angle_tx, rng)
 
 
def get_bi(beta_b, kappa_b, b_rx_elements, angle_rx, angle_tx, rng):
    """panel<->target, Eq.(15). b_rx_elements=L (panel); target side is always 1."""
    return get_hybird_channel_model(b_rx_elements, 1, kappa_b, beta_b,
                                     angle_rx, angle_tx, rng)
 
 
def get_hdk(beta_hdk, rx_elements, angle_rx, rng):
    """Direct BS<->user, Eq.(16). rx_elements=M (BS); kappa=0 exactly (pure NLoS)."""
    return get_hybird_channel_model(rx_elements, 1, 0.0, beta_hdk,
                                     angle_rx, 0.0, rng)

def get_rtt(rcs, G_i, ris, b_i):
    """BS to target, Eq. (8), """
    return 

