"""
Comprehensive radar SNR validation and simulation for multi-target multi-RIS ISAC.

Validates the corrected radar sensing equations from Clarifications.tex:
  - Deterministic round-trip operator T_i with Φ on both hops
  - Interference-plus-noise covariance J with proper σ_t² and σ_v² accounting
  - Multi-target SINR with mutual-target interference
"""

import numpy as np
import matplotlib.pyplot as plt
from src.utils.channel_utils import to_db, to_linear, to_linear_dbm
from src.sys.factory import build_system
from src.waveform.symbols import generate_qpsk_symbols
from src.utils.config import settings
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RadarSNRResults:
    """Container for radar SNR validation results."""
    snr_closed_form: Dict[int, float]  # target_id -> SNR (linear)
    snr_empirical: Dict[int, float]    # target_id -> SNR from echo energy ratio
    avg_snr_cf: float                  # average closed-form SNR
    avg_snr_emp: float                 # average empirical SNR
    mismatch_db: Dict[int, float]      # target_id -> diff in dB


def validate_radar_snr_single_symbol(system, x: np.ndarray) -> RadarSNRResults:
    """
    Validate closed-form radar SNR against echo energy for a single transmit snapshot.

    For each target, compute:
      - Closed-form: SNR_cf = σ_t² Tr(T R_x T† J⁻¹)
      - Empirical: SNR_emp = E[|signal|²] / E[|noise|²] from echo structure

    Args:
        system: ISACSystem with pre-built T and J
        x: M×1 transmit waveform (single symbol)

    Returns:
        RadarSNRResults with validation metrics
    """
    M = x.shape[0]
    Rx = system.get_Rx()  # Sum of user beamformer outer products

    snr_cf = {}
    snr_emp = {}

    for target in system.targets:
        sigma_t_sq = target.rcs

        # Closed-form: SNR = σ_t² Tr(T R_x T† J⁻¹)
        J_inv = np.linalg.inv(target.J)
        snr_cf[target.target_id] = (
            sigma_t_sq * np.trace(target.T @ Rx @ target.T.conj().T @ J_inv).real
        )

        # Empirical: build echo decomposition
        # y_r = α_t T x + [active noise] + [thermal noise]
        sigma_v_sq = to_linear(
            int(settings.config.channel_model.active_ris_noise) - 30
        )

        # Signal power: E[|α_t T x|²] = σ_t² ||T x||²
        signal_power = sigma_t_sq * np.linalg.norm(target.T @ x) ** 2

        # Noise power: from J, integrated with x
        # For simplicity, use Frobenius-norm heuristic over frequency
        noise_power = np.trace(target.J @ (Rx if Rx.shape == target.J.shape else np.eye(M)))

        snr_emp[target.target_id] = signal_power / (noise_power + 1e-10)

    avg_snr_cf = np.mean(list(snr_cf.values()))
    avg_snr_emp = np.mean(list(snr_emp.values()))

    mismatch_db = {
        tid: to_db(snr_cf[tid]) - to_db(snr_emp.get(tid, 1.0))
        for tid in snr_cf.keys()
    }

    return RadarSNRResults(
        snr_closed_form=snr_cf,
        snr_empirical=snr_emp,
        avg_snr_cf=avg_snr_cf,
        avg_snr_emp=avg_snr_emp,
        mismatch_db=mismatch_db,
    )


def validate_radar_snr_montecarlo(
    system, n_symbols: int = 10000, rng=None
) -> RadarSNRResults:
    """
    Validate radar SNR via Monte Carlo averaging over many transmit symbols.

    Simulates target echo for n_symbols transmit instances, averaging the observed
    SNR ratio across realizations.

    Args:
        system: ISACSystem (pre-built T, J, beamformers)
        n_symbols: Number of transmit symbols to simulate
        rng: numpy random generator

    Returns:
        RadarSNRResults: Closed-form vs. empirical SNR comparison
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    K = len(system.users)
    M = settings.config.channel_model.M
    Rx = system.get_Rx()

    snr_cf = {}
    snr_emp = {}

    for target in system.targets:
        sigma_t_sq = target.rcs

        # Closed-form SNR
        J_inv = np.linalg.inv(target.J)
        snr_cf[target.target_id] = (
            sigma_t_sq * np.trace(target.T @ Rx @ target.T.conj().T @ J_inv).real
        )

        # Empirical: average over symbols
        signal_energy_sum = 0.0
        noise_energy_sum = 0.0

        for sym_idx in range(n_symbols):
            # Generate symbols and transmit
            s = generate_qpsk_symbols(K, n_symbols=1, rng=rng)  # K x 1
            x = system.transmit_waveform(s)  # M x 1

            # Sample RCS (Swerling-I: constant over CPI, decorrelate scan-to-scan)
            alpha_t = (
                (rng.standard_normal() + 1j * rng.standard_normal())
                * np.sqrt(sigma_t_sq / 2)
            )

            # Target echo signal
            y_signal = alpha_t * (target.T @ x)  # M x 1

            # Noise: use J as noise power spectral density
            noise_vec = (
                (rng.standard_normal(M) + 1j * rng.standard_normal(M))
                * np.sqrt(1.0 / 2)
            )
            y_noise = (
                np.linalg.cholesky(target.J) @ noise_vec
            )  # Colored noise from J

            # Full echo
            y_echo = y_signal + y_noise  # M x 1

            # Extract SNR from energy ratio
            signal_energy_sum += np.linalg.norm(y_signal) ** 2
            noise_energy_sum += np.linalg.norm(y_noise) ** 2

        snr_emp[target.target_id] = signal_energy_sum / (noise_energy_sum + 1e-10)

    avg_snr_cf = np.mean(list(snr_cf.values()))
    avg_snr_emp = np.mean(list(snr_emp.values()))

    mismatch_db = {
        tid: to_db(snr_cf[tid]) - to_db(snr_emp.get(tid, 1.0))
        for tid in snr_cf.keys()
    }

    return RadarSNRResults(
        snr_closed_form=snr_cf,
        snr_empirical=snr_emp,
        avg_snr_cf=avg_snr_cf,
        avg_snr_emp=avg_snr_emp,
        mismatch_db=mismatch_db,
    )


def print_snr_validation(results: RadarSNRResults, title: str = "Radar SNR Validation"):
    """Print formatted SNR validation report."""
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)
    print(
        f"{'Target':>8} {'Closed-Form (dB)':>20} {'Empirical (dB)':>20} {'Mismatch (dB)':>15}"
    )
    print("-" * 70)

    for tid in sorted(results.snr_closed_form.keys()):
        cf_db = to_db(results.snr_closed_form[tid])
        emp_db = to_db(results.snr_empirical.get(tid, 1.0))
        mismatch = results.mismatch_db[tid]

        flag = "" if abs(mismatch) < 1.0 else "  <-- CHECK"
        print(f"{tid:>8} {cf_db:>20.2f} {emp_db:>20.2f} {mismatch:>15.2f}{flag}")

    print("-" * 70)
    print(
        f"{'AVERAGE':>8} {to_db(results.avg_snr_cf):>20.2f} "
        f"{to_db(results.avg_snr_emp):>20.2f} "
        f"{to_db(results.avg_snr_cf / (results.avg_snr_emp + 1e-10)):>15.2f}"
    )
    print("=" * 70 + "\n")


def simulate_snr_vs_power(
    n_panels: int = 50,
    k_users: int = 4,
    k_targets: int = 1,
    active_ratio: float = 0.25,
    p_max_dbm_list: List[float] = None,
    rng=None,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Simulate radar SNR as a function of transmit power.

    Sweeps P_max and for each value, builds a fresh system and computes SNR.

    Args:
        n_panels: Total RIS panels
        k_users: Number of communication users
        k_targets: Number of radar targets
        active_ratio: Fraction of panels that are active
        p_max_dbm_list: Transmit power budget in dBm
        rng: Random generator

    Returns:
        (p_max_dbm_list, snr_dict)
          where snr_dict['multi_ris'] is the multi-RIS SNR in dB
                snr_dict['single_ris'] is single-RIS baseline SNR in dB
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    if p_max_dbm_list is None:
        p_max_dbm_list = np.linspace(10, 40, 7)

    results = {
        "multi_ris": [],
        "single_ris": [],
        "dual_ris": [],
        "no_ris": [],
    }

    settings.load_config()

    for p_max_dbm in p_max_dbm_list:
        p_max_linear = to_linear_dbm(p_max_dbm)

        # Active panel configuration
        active_mask = np.zeros(n_panels, dtype=bool)
        n_active = max(1, int(n_panels * active_ratio))
        active_mask[:n_active] = True
        rng.shuffle(active_mask)

        # Multi-RIS system (all panels)
        system_multi = build_system(
            n_panels=n_panels,
            k_users=k_users,
            k_targets=k_targets,
            active_mask=active_mask,
            L=settings.config.channel_model.L,
            M=settings.config.channel_model.M,
            p_total_linear=p_max_linear,
            rng=rng,
        )
        system_multi.build_hbar()
        system_multi.build_w_mrt()
        system_multi.build_T()
        system_multi.build_J()
        avg_snr_multi, _ = system_multi.compute_radar_snr()
        results["multi_ris"].append(to_db(avg_snr_multi))

        # Single-RIS baseline: use best single panel
        single_active = np.zeros(n_panels, dtype=bool)
        single_active[0] = True
        single_active_true = np.zeros(1, dtype=bool)
        single_active_true[0] = True if active_mask[0] else False

        system_single = build_system(
            n_panels=1,
            k_users=k_users,
            k_targets=k_targets,
            active_mask=single_active_true,
            L=settings.config.channel_model.L,
            M=settings.config.channel_model.M,
            p_total_linear=p_max_linear,
            rng=rng,
        )
        system_single.build_hbar()
        system_single.build_w_mrt()
        system_single.build_T()
        system_single.build_J()
        avg_snr_single, _ = system_single.compute_radar_snr()
        results["single_ris"].append(to_db(avg_snr_single))

        # Dual-RIS baseline: use best two panels
        dual_active = np.zeros(2, dtype=bool)
        for i in range(min(2, n_panels)):
            dual_active[i] = active_mask[i]

        system_dual = build_system(
            n_panels=2,
            k_users=k_users,
            k_targets=k_targets,
            active_mask=dual_active,
            L=settings.config.channel_model.L,
            M=settings.config.channel_model.M,
            p_total_linear=p_max_linear,
            rng=rng,
        )
        system_dual.build_hbar()
        system_dual.build_w_mrt()
        system_dual.build_T()
        system_dual.build_J()
        avg_snr_dual, _ = system_dual.compute_radar_snr()
        results["dual_ris"].append(to_db(avg_snr_dual))

        # No-RIS baseline: direct BS-target link only
        system_no_ris = build_system(
            n_panels=1,
            k_users=k_users,
            k_targets=k_targets,
            active_mask=np.zeros(1, dtype=bool),  # No panels active
            L=settings.config.channel_model.L,
            M=settings.config.channel_model.M,
            p_total_linear=p_max_linear,
            rng=rng,
        )
        system_no_ris.build_hbar()
        system_no_ris.build_w_mrt()
        system_no_ris.build_T()
        system_no_ris.build_J()
        avg_snr_no_ris, _ = system_no_ris.compute_radar_snr()
        results["no_ris"].append(to_db(avg_snr_no_ris))

    return list(p_max_dbm_list), results


def plot_snr_vs_power(p_max_dbm_list, snr_results, save_path: str = None):
    """
    Plot radar SNR vs. transmit power for different RIS configurations.

    Args:
        p_max_dbm_list: Transmit power budget in dBm
        snr_results: Dict from simulate_snr_vs_power()
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        p_max_dbm_list,
        snr_results["multi_ris"],
        "b-o",
        linewidth=2,
        label="Multi-RIS (Proposed)",
        markersize=8,
    )
    ax.plot(
        p_max_dbm_list,
        snr_results["dual_ris"],
        "g-s",
        linewidth=2,
        label="Dual-RIS",
        markersize=7,
    )
    ax.plot(
        p_max_dbm_list,
        snr_results["single_ris"],
        "r-^",
        linewidth=2,
        label="Single-RIS",
        markersize=7,
    )
    ax.plot(
        p_max_dbm_list,
        snr_results["no_ris"],
        "k--x",
        linewidth=2,
        label="No-RIS",
        markersize=7,
    )

    ax.set_xlabel("Transmit Power (dBm)", fontsize=12)
    ax.set_ylabel("Radar SNR (dB)", fontsize=12)
    ax.set_title("Radar SNR vs. Transmit Power", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def simulate_active_passive_tradeoff(
    n_panels: int = 50,
    k_users: int = 4,
    k_targets: int = 1,
    p_max_dbm: float = 30,
    rng=None,
) -> Tuple[List[int], List[float]]:
    """
    Simulate the effect of active/passive mix on system utility.

    Sweeps the number of active panels and records the radar SNR and communication
    performance.

    Args:
        n_panels: Total panels
        k_users: Communication users
        k_targets: Targets
        p_max_dbm: Transmit power
        rng: Random generator

    Returns:
        (n_active_list, snr_list)
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    settings.load_config()
    p_max_linear = to_linear_dbm(p_max_dbm)

    n_active_list = list(range(0, n_panels + 1, max(1, n_panels // 10)))
    snr_list = []

    for n_active in n_active_list:
        active_mask = np.zeros(n_panels, dtype=bool)
        if n_active > 0:
            active_mask[:n_active] = True
            rng.shuffle(active_mask)

        system = build_system(
            n_panels=n_panels,
            k_users=k_users,
            k_targets=k_targets,
            active_mask=active_mask,
            L=settings.config.channel_model.L,
            M=settings.config.channel_model.M,
            p_total_linear=p_max_linear,
            rng=rng,
        )
        system.build_hbar()
        system.build_w_mrt()
        system.build_T()
        system.build_J()
        avg_snr, _ = system.compute_radar_snr()
        snr_list.append(to_db(avg_snr))

    return n_active_list, snr_list


def plot_active_passive_tradeoff(n_active_list, snr_list, save_path: str = None):
    """Plot the effect of active panel count on SNR."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        n_active_list,
        snr_list,
        "b-o",
        linewidth=2,
        markersize=8,
    )

    ax.set_xlabel("Number of Active Panels", fontsize=12)
    ax.set_ylabel("Radar SNR (dB)", fontsize=12)
    ax.set_title("Active/Passive Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
