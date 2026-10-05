"""Required single-pulse SNR (dB) for the detection convention in configs/default.yaml."""
import numpy as np
from src.utils.config import settings


def required_snr_db(pd=None, pfa=None, fluctuation=None):
    d = getattr(settings.config.channel_model, "detection", None)
    pd = float(pd if pd is not None else getattr(d, "pd", 0.9))
    pfa = float(pfa if pfa is not None else getattr(d, "pfa", 1e-6))
    fluctuation = fluctuation or getattr(d, "target_fluctuation", "swerling1")
    if fluctuation == "swerling1":                      # Pd = Pfa^(1/(1+SNR)), exact for a Rayleigh (CN) target
        return float(10 * np.log10(np.log(pfa) / np.log(pd) - 1))
    A, B = np.log(0.62 / pfa), np.log(pd / (1 - pd))    # Albersheim, non-fluctuating, one pulse
    return float((6.2 + 4.54 / np.sqrt(1.44)) * np.log10(A + 0.12 * A * B + 1.7 * B))
