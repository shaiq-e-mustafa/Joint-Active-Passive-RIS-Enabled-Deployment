"""
Phase A, Week 1: minimum panel count for blockage-robust radar sensing.

Original framing ("minimum N for 20 dB worst-case SNR") assumed an absolute
SNR scale that doesn't match this codebase's actual passive double-reflection
link budget (~-200 dB at 28 GHz, 20-100m range, random/unoptimized RIS
phases -- see docs/phase_a_log.html Entry 2 for the derivation). Reframed as
a RELATIVE target: how many panels does it take so that blockage costs you
at most `target_margin_db` relative to the same geometry's unblocked
ceiling? This answers the actual question ("how much does blockage hurt, and
how many panels buys back robustness") without depending on an absolute
number tied to unrelated system parameters (transmit power, array size).

For each candidate panel count N, generate many random geometries; for each
geometry, measure its unblocked ceiling once, then resample the Line Boolean
blockage pattern many times (same positions/channels, only which panels are
active changes) and keep the worst SNR seen. The per-geometry gap
(ceiling - worst-case) is averaged across geometries.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.config import settings
from src.utils.channel_utils import to_linear_dbm, to_db, distance
from src.sys.factory import build_system
from src.channel.risConfig import calibrate_lambda_b, los_probability_line_boolean
from src.sim.deployment import BS_POS, PANEL_RADIUS


def resample_blockage(system, target_p_block, mean_obstacle_len, rng):
    """Redraw a Line Boolean blockage mask for an already-built system, using
    each panel's stored position, then rebuild everything downstream of panel
    activation (h_bar, MRT beamformers, T, J). Channel realizations (G, b, f)
    are untouched -- only which panels contribute changes.
    """
    d_ref = 0.5 * (PANEL_RADIUS[0] + PANEL_RADIUS[1])
    lambda_b = calibrate_lambda_b(target_p_block, d_ref, mean_obstacle_len)

    for panel in system.panels:
        d = distance(BS_POS, panel.pos)
        p_block = 1.0 - los_probability_line_boolean(d, lambda_b, mean_obstacle_len)
        panel.state.a = 0 if rng.random() < p_block else 1

    system.build_hbar()
    system.build_w_mrt()
    system.build_T()
    system.build_J()


def hybrid_snr_db(system, snr_meas_ratio, rng):
    """Closed-form radar SNR plus additive Gaussian measurement noise scaled
    relative to the true SNR (sigma = SNR_true / snr_meas_ratio). Captures
    CSI-quantization / estimation error without simulating raw symbols.
    """
    multi_target = len(system.targets) > 1
    avg_snr_linear, _ = system.compute_radar_snr(multi_target_interference=multi_target)
    sigma_meas = avg_snr_linear / snr_meas_ratio
    noisy_snr_linear = avg_snr_linear + rng.normal(0.0, sigma_meas)
    # Floor only guards log(<=0) when every panel is blocked (avg_snr_linear == 0)
    # or a pathological noise draw; this system's real SNRs run ~1e-20 (~-200 dB),
    # so the floor must sit far below that, not at an arbitrary "small" constant.
    noisy_snr_linear = max(noisy_snr_linear, np.finfo(float).tiny)
    return to_db(noisy_snr_linear)


class ProgressLogger:
    """Explicit, immediately-flushed progress log -- stdout alone was silently
    buffered when redirected to a file (as background runs always are), which
    made a long sweep indistinguishable from a hung one until it finished.
    Writes to both stdout (flushed per line) and a log file that can be
    tailed live with `tail -f` while the sweep is running.
    """

    def __init__(self, log_path=None):
        self.log_file = open(log_path, "w", encoding="utf-8") if log_path else None
        self.start_time = time.time()

    def write(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] [+{time.time() - self.start_time:7.1f}s] {msg}"
        print(line, flush=True)
        if self.log_file is not None:
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def close(self):
        if self.log_file is not None:
            self.log_file.close()


def evaluate_geometry_robustness(n_panels, k_users, k_targets, L, M, p_total_linear,
                                  target_p_block, mean_obstacle_len, n_blockage_trials,
                                  snr_meas_ratio, rng):
    """One random geometry: build it once (unblocked), measure its ceiling SNR,
    then resample the blockage pattern n_blockage_trials times and measure the
    worst SNR seen. Returns (ceiling_db, worst_case_db, gap_db) for this one
    geometry -- gap_db is how much blockage costs THIS geometry specifically
    (a paired measurement, since both are drawn from the same channels).
    """
    system = build_system(
        n_panels=n_panels, k_users=k_users, k_targets=k_targets,
        active_mask=np.zeros(n_panels, dtype=bool),  # passive-only baseline, see log
        L=L, M=M, p_total_linear=p_total_linear, rng=rng,
        target_p_block=0.0,  # build unblocked; blockage applied per-trial below
    )

    # All panels start active (built with target_p_block=0.0); build the
    # downstream quantities once to measure the unblocked ceiling.
    system.build_hbar()
    system.build_w_mrt()
    system.build_T()
    system.build_J()
    ceiling_db = hybrid_snr_db(system, snr_meas_ratio, rng)

    snrs_db = []
    for _ in range(n_blockage_trials):
        resample_blockage(system, target_p_block, mean_obstacle_len, rng)
        snrs_db.append(hybrid_snr_db(system, snr_meas_ratio, rng))
    worst_case_db = min(snrs_db)

    return ceiling_db, worst_case_db, ceiling_db - worst_case_db


def run_week1_sweep(
    n_values=(20, 30, 40, 50),
    target_margin_db=3.0,
    target_p_block=0.30,
    n_geometry_trials=100,
    n_blockage_trials=100,
    mean_obstacle_len=5.0,
    snr_meas_ratio=10.0,
    k_users=None,
    k_targets=2,
    seed=42,
    out_dir="out/figs",
    log_path="out/week1_sweep.log",
    progress_every=1,
):
    """Run the full Week 1 sweep and save the N-vs-robustness plot.

    Logs progress (current N, geometry trial index, elapsed/ETA) to stdout
    and to `log_path` as it runs -- both flushed per line, so `tail -f
    out/week1_sweep.log` (or Read on it) shows live progress even while the
    process itself is running in the background with its own stdout buffered.

    Returns a dict with per-N results and the minimum N whose average
    ceiling-to-worst-case gap is within target_margin_db.
    """
    settings.load_config()
    k_users = k_users if k_users is not None else settings.config.channel_model.K_USERS
    L = settings.config.channel_model.L
    M = settings.config.channel_model.M
    p_total_linear = to_linear_dbm(settings.config.channel_model.P_max)

    rng = np.random.default_rng(seed)
    log = ProgressLogger(log_path)

    total_trials = len(n_values) * n_geometry_trials
    trials_done = 0

    log.write(f"Starting Week 1 sweep: N={list(n_values)}, "
              f"{n_geometry_trials} geometry trials x {n_blockage_trials} blockage trials each "
              f"({total_trials} geometry-trials total), target_p_block={target_p_block}, "
              f"target_margin_db={target_margin_db}")

    avg_ceiling_db, avg_worst_db, avg_gap_db = [], [], []
    all_gap_db = {}

    for n_panels in n_values:
        log.write(f"N={n_panels}: starting {n_geometry_trials} geometry trials "
                  f"({n_blockage_trials} blockage draws each)")
        ceilings, worsts, gaps = [], [], []

        for trial_idx in range(n_geometry_trials):
            ceiling_db, worst_case_db, gap_db = evaluate_geometry_robustness(
                n_panels, k_users, k_targets, L, M, p_total_linear,
                target_p_block, mean_obstacle_len, n_blockage_trials,
                snr_meas_ratio, rng,
            )
            ceilings.append(ceiling_db)
            worsts.append(worst_case_db)
            gaps.append(gap_db)
            trials_done += 1

            if (trial_idx + 1) % progress_every == 0 or trial_idx + 1 == n_geometry_trials:
                elapsed = time.time() - log.start_time
                rate = trials_done / elapsed if elapsed > 0 else 0
                eta = (total_trials - trials_done) / rate if rate > 0 else float("nan")
                log.write(
                    f"N={n_panels}: geometry trial {trial_idx + 1}/{n_geometry_trials} done "
                    f"(gap so far: mean={np.mean(gaps):5.2f} dB, last={gap_db:5.2f} dB) "
                    f"| overall {trials_done}/{total_trials} | "
                    f"{rate:.3f} trials/s | ETA {eta:6.0f}s"
                )

        all_gap_db[n_panels] = gaps
        avg_ceiling_db.append(float(np.mean(ceilings)))
        avg_worst_db.append(float(np.mean(worsts)))
        avg_gap_db.append(float(np.mean(gaps)))

        log.write(f"N={n_panels:>3} COMPLETE: ceiling={np.mean(ceilings):7.2f} dB, "
                  f"worst-case={np.mean(worsts):7.2f} dB, "
                  f"gap={np.mean(gaps):5.2f} dB (std={np.std(gaps):4.2f})")

    min_n = next((n for n, gap in zip(n_values, avg_gap_db) if gap <= target_margin_db), None)

    fig_path = _plot_sweep(n_values, avg_ceiling_db, avg_worst_db, avg_gap_db, all_gap_db,
                            target_margin_db, target_p_block, out_dir)

    log.write(f"Sweep complete. min_n={min_n}. Figure saved to {fig_path}")
    log.close()

    return {
        "n_values": list(n_values),
        "avg_ceiling_db": avg_ceiling_db,
        "avg_worst_snr_db": avg_worst_db,
        "avg_gap_db": avg_gap_db,
        "all_gap_db": all_gap_db,
        "min_n": min_n,
        "target_margin_db": target_margin_db,
        "target_p_block": target_p_block,
        "fig_path": fig_path,
    }


def _plot_sweep(n_values, avg_ceiling_db, avg_worst_db, avg_gap_db, all_gap_db,
                 target_margin_db, target_p_block, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gap_stds = [np.std(all_gap_db[n]) for n in n_values]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    ax1.plot(n_values, avg_ceiling_db, "-o", color="#10b981", linewidth=2, markersize=7,
              label="Ceiling (0% blocked)")
    ax1.plot(n_values, avg_worst_db, "-o", color="#1e3a8a", linewidth=2, markersize=7,
              label=f"Worst-case ({int(target_p_block * 100)}% blocked)")
    ax1.set_ylabel("Radar SNR (dB)", fontsize=12)
    ax1.set_title(
        f"Week 1: Blockage Robustness vs. Panel Count\n"
        f"({int(target_p_block * 100)}% blockage, Line Boolean model)",
        fontsize=13, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="best")

    ax2.errorbar(n_values, avg_gap_db, yerr=gap_stds, fmt="-o", color="#ef4444",
                 linewidth=2, markersize=7, capsize=4, label="Ceiling - worst-case gap")
    ax2.axhline(target_margin_db, color="#f59e0b", linestyle="--", linewidth=1.5,
                label=f"Target margin = {target_margin_db:.1f} dB")
    ax2.set_xlabel("Number of panels (N)", fontsize=12)
    ax2.set_ylabel("SNR lost to blockage (dB)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="best")

    plt.tight_layout()
    fig_path = str(Path(out_dir) / "week1_panel_count_sweep.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


if __name__ == "__main__":
    results = run_week1_sweep()
    print()
    if results["min_n"] is not None:
        print(f"Decision: minimum N = {results['min_n']} panels keeps blockage cost within "
              f"{results['target_margin_db']} dB of the unblocked ceiling "
              f"at {int(results['target_p_block'] * 100)}% blockage")
    else:
        print(f"No tested N closes the gap to {results['target_margin_db']} dB -- sweep needs larger N values")
    print(f"Figure saved to {results['fig_path']}")
