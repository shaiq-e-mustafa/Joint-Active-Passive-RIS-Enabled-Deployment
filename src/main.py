import numpy as np
from src.utils.config import settings
from src.utils.channel_utils import to_linear, to_db
from src.sys.factory import build_system
from src.waveform.symbols import generate_qpsk_symbols
from dataclasses import dataclass, asdict
from src.sim.validation import validate_sinr_montecarlo



if __name__ == "__main__":
    rng = np.random.default_rng(0)
    settings.load_config()

    N_PANELS = 50
    N_SYMBOLS = 100000
    K_USERS = 8
    L = 32          # elements per panel
    M = 64          # BS antennas

    active_mask = np.zeros(N_PANELS, dtype=bool)
    active_mask[: N_PANELS // 4] = True
    rng.shuffle(active_mask)

    system = build_system(
        n_panels=N_PANELS,
        k_users=K_USERS,
        active_mask=active_mask,
        L=L,
        M=M,
        p_total_linear=to_linear(settings.config.channel_model.P_max),   
        rng=rng
    )

    system.build_hbar()
    system.build_w_mrt()
    

    s = generate_qpsk_symbols(K_USERS, n_symbols=N_SYMBOLS, rng=rng)
    x = system.transmit_waveform(s)
    y, desired, interfernce = system.get_received_signal(s=s, rng=rng)
    system.build_sinr()
    if settings.config.validate == True:
      validate_sinr_montecarlo(system, y, desired)
    