import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

import numpy as np
from src.utils.config import settings
from src.utils.channel_utils import to_linear, to_db
from src.sys.factory import build_system
from src.waveform.symbols import generate_qpsk_symbols
from src.sim.validation import validate_sinr_montecarlo, plot



if __name__ == "__main__":
    rng = np.random.default_rng(0)
    settings.load_config()

    N_PANELS = 50
    N_SYMBOLS = 100000
    N_TARGETS = 2  # Multi-target scenario

    # Configure active RIS panels: ~25% active, ~75% passive
    active_mask = np.zeros(N_PANELS, dtype=bool)
    active_mask[: N_PANELS // 4] = True
    rng.shuffle(active_mask)

    print("="*80)
    print("MULTI-RIS ISAC SYSTEM DEPLOYMENT")
    print("="*80)
    print(f"Configuration:")
    print(f"  Total panels: {N_PANELS}")
    print(f"  Active panels: {np.sum(active_mask)}")
    print(f"  Communication users: {settings.config.channel_model.K_USERS}")
    print(f"  Radar targets: {N_TARGETS}")
    print()

    # Build system with multiple targets
    system = build_system(
        n_panels=N_PANELS,
        k_users=settings.config.channel_model.K_USERS,
        k_targets=N_TARGETS,  # Multi-target support
        active_mask=active_mask,
        L=settings.config.channel_model.L,
        M=settings.config.channel_model.M,
        p_total_linear=to_linear(settings.config.channel_model.P_max),
        rng=rng,
    )

    # Build communication channels
    system.build_hbar()
    system.build_w_mrt()

    # Communication link simulation
    print("Communication Link Analysis:")
    print("-" * 80)
    s = generate_qpsk_symbols(
        settings.config.channel_model.K_USERS,
        n_symbols=N_SYMBOLS,
        rng=rng,
    )
    x = system.transmit_waveform(s)
    y, desired, interference = system.get_received_signal(s=s, rng=rng)
    system.build_sinr()
    system.get_achievable_rate()
    system.get_ber_per_user()
    system.get_total_ber()

    # Plot communication constellation
    plot(y, desired, 0, s)

    # Validate communication SINR
    if True:
        validate_sinr_montecarlo(system, y, desired)

    print(f"\nCommunication metrics:")
    for k, user in enumerate(system.users):
        print(f"  User {k}: SINR = {to_db(user.sinr):>8.2f} dB, "
              f"Rate = {user.air:>6.2f} bps/Hz, BER = {user.ber:.2e}")
    print(f"  Average BER: {system.ber_comm:.2e}")
    print()

    # Radar link simulation
    print("Radar Link Analysis:")
    print("-" * 80)

    # Build radar sensing operators (corrected formulation)
    system.build_T()    # Deterministic round-trip operator T_i
    system.build_J()    # Interference-plus-noise covariance J

    # Compute SNR/SINR for all targets
    # Single-target SNR: ignores mutual interference
    avg_snr_single, snr_dict_single = system.compute_radar_snr(
        multi_target_interference=False
    )

    # Multi-target SINR: includes mutual-target interference (Clarifications.tex, §Q3)
    avg_snr_multi, snr_dict_multi = system.compute_radar_snr(
        multi_target_interference=True
    )

    print(f"\nRadar SNR (ignoring mutual interference):")
    for target_id, snr_linear in snr_dict_single.items():
        print(f"  Target {target_id}: SNR = {to_db(snr_linear):>8.2f} dB")
    print(f"  Average SNR: {to_db(avg_snr_single):>8.2f} dB")

    if N_TARGETS > 1:
        print(f"\nRadar SINR (with mutual-target interference):")
        for target_id, sinr_linear in snr_dict_multi.items():
            snr_single = snr_dict_single[target_id]
            interference_penalty = to_db(snr_single / sinr_linear)
            print(f"  Target {target_id}: SINR = {to_db(sinr_linear):>8.2f} dB "
                  f"(penalty = {interference_penalty:>5.2f} dB)")
        print(f"  Average SINR: {to_db(avg_snr_multi):>8.2f} dB")

    # ISAC trade-off metrics
    print()
    print("ISAC Trade-off Metrics:")
    print("-" * 80)
    alpha_list = [0.25, 0.5, 0.75]
    for alpha in alpha_list:
        radar_term = alpha * to_db(avg_snr_multi) / 30  # Normalize to 30 dB
        comm_term = (1 - alpha) * (1 - system.ber_comm)
        isac_utility = radar_term + comm_term
        print(f"  α={alpha:.2f}: Radar={radar_term:.3f}, Comm={comm_term:.3f}, "
              f"Utility={isac_utility:.3f}")

    print()
    print("="*80)
    print("Deployment complete. Check RADAR_SNR_IMPLEMENTATION.md for details.")
    print("="*80)  
