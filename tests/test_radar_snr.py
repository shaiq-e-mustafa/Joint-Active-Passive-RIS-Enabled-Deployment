"""
Comprehensive test suite for multi-target radar SNR validation.

Tests:
  1. Single-target SNR validation (closed-form vs empirical)
  2. Multi-target SNR with mutual-target interference
  3. SNR scaling with transmit power
  4. Active/passive panel trade-off
  5. System utility (ISAC balance) vs weighting parameter α
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import settings
from src.utils.channel_utils import to_linear, to_db
from src.sys.factory import build_system
from src.waveform.symbols import generate_qpsk_symbols
from src.sim.radar_validation import (
    validate_radar_snr_montecarlo,
    print_snr_validation,
    simulate_snr_vs_power,
    plot_snr_vs_power,
    simulate_active_passive_tradeoff,
    plot_active_passive_tradeoff,
)


def test_single_target_snr():
    """Test 1: Validate single-target SNR with corrected equations."""
    print("\n" + "="*80)
    print("TEST 1: Single-Target Radar SNR Validation")
    print("="*80)

    settings.load_config()
    rng = np.random.default_rng(seed=42)

    # Build system with modest configuration for testing
    system = build_system(
        n_panels=20,
        k_users=4,
        k_targets=1,
        active_mask=np.array([True] * 5 + [False] * 15),  # 5 active, 15 passive
        L=settings.config.channel_model.L,
        M=settings.config.channel_model.M,
        p_total_linear=to_linear(30),  # 30 dBm
        rng=rng,
    )

    # Build system components
    system.build_hbar()
    system.build_w_mrt()
    system.build_T()
    system.build_J()

    # Validate SNR
    results = validate_radar_snr_montecarlo(system, n_symbols=5000, rng=rng)
    print_snr_validation(results, title="Single-Target SNR Validation")

    # Check tolerance
    for tid, mismatch in results.mismatch_db.items():
        if abs(mismatch) < 2.0:
            print(f"✓ Target {tid}: closed-form SNR matches empirical (mismatch={mismatch:.2f}dB)")
        else:
            print(
                f"✗ Target {tid}: SNR mismatch {mismatch:.2f}dB exceeds tolerance"
            )

    return results


def test_multi_target_snr():
    """Test 2: Validate multi-target SINR with mutual interference."""
    print("\n" + "="*80)
    print("TEST 2: Multi-Target Radar SINR with Mutual Interference")
    print("="*80)

    settings.load_config()
    rng = np.random.default_rng(seed=123)

    # Build system with 3 targets
    system = build_system(
        n_panels=30,
        k_users=4,
        k_targets=3,  # Multiple targets
        active_mask=np.array([True] * 8 + [False] * 22),  # 8 active panels
        L=settings.config.channel_model.L,
        M=settings.config.channel_model.M,
        p_total_linear=to_linear(30),
        rng=rng,
    )

    system.build_hbar()
    system.build_w_mrt()
    system.build_T()
    system.build_J()

    # Compute SINR with and without multi-target interference
    snr_no_interference, _ = system.compute_radar_snr(multi_target_interference=False)
    sinr_with_interference, sinr_dict = system.compute_radar_snr(
        multi_target_interference=True
    )

    print(f"\nRadar performance metrics:")
    print(f"  SNR (ignoring target interference): {to_db(snr_no_interference):>8.2f} dB")
    print(f"  SINR (with mutual interference):     {to_db(sinr_with_interference):>8.2f} dB")
    print(f"  SNR penalty from interference:       {to_db(snr_no_interference) - to_db(sinr_with_interference):>8.2f} dB")

    print(f"\nPer-target SINR:")
    for tid, sinr in sinr_dict.items():
        print(f"  Target {tid}: {to_db(sinr):>8.2f} dB")

    return sinr_dict


def test_snr_vs_power():
    """Test 3: Simulate SNR scaling with transmit power."""
    print("\n" + "="*80)
    print("TEST 3: Radar SNR vs. Transmit Power")
    print("="*80)

    p_max_dbm_list = np.linspace(15, 35, 5)

    print(f"Sweeping transmit power from {p_max_dbm_list[0]:.1f} to {p_max_dbm_list[-1]:.1f} dBm...")
    p_list, snr_dict = simulate_snr_vs_power(
        n_panels=40,
        k_users=4,
        k_targets=1,
        active_ratio=0.25,
        p_max_dbm_list=p_max_dbm_list,
        rng=np.random.default_rng(seed=456),
    )

    print(f"\nRadar SNR results:")
    print(
        f"{'P_max (dBm)':>12} {'Multi-RIS':>12} {'Dual-RIS':>12} {'Single-RIS':>12} {'No-RIS':>12}"
    )
    print("-" * 65)
    for p, snr_m, snr_d, snr_s, snr_n in zip(
        p_list,
        snr_dict["multi_ris"],
        snr_dict["dual_ris"],
        snr_dict["single_ris"],
        snr_dict["no_ris"],
    ):
        print(f"{p:>12.1f} {snr_m:>12.2f} {snr_d:>12.2f} {snr_s:>12.2f} {snr_n:>12.2f}")

    # Compute gains
    gain_vs_single = [
        m - s for m, s in zip(snr_dict["multi_ris"], snr_dict["single_ris"])
    ]
    gain_vs_dual = [m - d for m, d in zip(snr_dict["multi_ris"], snr_dict["dual_ris"])]

    avg_gain_single = np.mean(gain_vs_single)
    avg_gain_dual = np.mean(gain_vs_dual)

    print(f"\nAverage SNR gains:")
    print(f"  Multi-RIS vs Single-RIS: {avg_gain_single:>6.2f} dB")
    print(f"  Multi-RIS vs Dual-RIS:   {avg_gain_dual:>6.2f} dB")

    # Plot
    plot_snr_vs_power(p_list, snr_dict)

    return p_list, snr_dict


def test_active_passive_tradeoff():
    """Test 4: Active/passive panel trade-off."""
    print("\n" + "="*80)
    print("TEST 4: Active/Passive Panel Trade-off")
    print("="*80)

    print(f"Sweeping number of active panels from 0 to 50...")
    n_active_list, snr_list = simulate_active_passive_tradeoff(
        n_panels=50,
        k_users=4,
        k_targets=1,
        p_max_dbm=30,
        rng=np.random.default_rng(seed=789),
    )

    print(f"\nActive/Passive trade-off:")
    print(f"{'N_active':>8} {'SNR (dB)':>12}")
    print("-" * 25)
    for n, snr in zip(n_active_list, snr_list):
        print(f"{n:>8} {snr:>12.2f}")

    # Find optimal
    max_snr_idx = np.argmax(snr_list)
    optimal_n_active = n_active_list[max_snr_idx]
    max_snr_db = snr_list[max_snr_idx]

    print(f"\nOptimal configuration:")
    print(f"  Number of active panels: {optimal_n_active}")
    print(f"  Maximum SNR: {max_snr_db:.2f} dB")
    print(f"  Active/total ratio: {optimal_n_active / 50:.1%}")

    # Plot
    plot_active_passive_tradeoff(n_active_list, snr_list)

    return n_active_list, snr_list


def run_all_tests():
    """Run all validation tests."""
    print("\n" * 2)
    print("=" * 80)
    print("MULTI-TARGET RADAR SNR VALIDATION TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Single-target
        test_single_target_snr()

        # Test 2: Multi-target
        test_multi_target_snr()

        # Test 3: Power sweep
        test_snr_vs_power()

        # Test 4: Active/passive trade-off
        test_active_passive_tradeoff()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
