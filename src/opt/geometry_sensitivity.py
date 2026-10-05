"""
Geometry/frequency sensitivity sweep, triggered by: "what if we used a lower
carrier frequency (e.g. L-band) and/or shorter deployment ranges?"

Entry 4 of docs/phase_a_log.html established that the ~-220 dB SNR scale is a
physically consistent result of beta_0_dB=-61 dB (the exact Friis free-space
loss at 1m for 28 GHz) combined with a passive double-reflection link over
20-100m. This script grids over beta_0_dB (standing in for carrier frequency,
since the channel model does not actually derive path loss from
carrier_frequency -- see note below) and over panel/target deployment radii,
to see how much of the gap to a realistic operating SNR each lever closes.

Note: `carrier_frequency` in configs/default.yaml is NOT wired into
get_path_loss_linear() -- only beta_0_dB drives path loss. So "switching to
L-band" here means overriding beta_0_dB directly, computed from Friis at the
target frequency, not editing carrier_frequency (which would do nothing).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.config import settings
from src.utils.channel_utils import to_linear_dbm, to_db
from src.sys.factory import build_system


def friis_beta_0_db(freq_hz, d0=1.0, c=3e8):
    """Free-space path gain (negative of path loss) at reference distance d0,
    for a carrier at freq_hz. beta_0_dB = -20*log10(4*pi*d0*f/c).
    """
    return -20.0 * np.log10(4.0 * np.pi * d0 * freq_hz / c)


def ceiling_snr_db(n_panels, k_users, k_targets, L, M, p_total_linear,
                    panel_radius, target_radius, active_fraction, rng):
    """Unblocked (every panel's PanelState.a=1, i.e. none knocked out by the
    Line Boolean model -- see Entry 1) closed-form radar SNR for one random
    geometry at the given deployment radii, using whatever beta_0_dB is
    currently set in settings.config.channel_model (caller sets this).

    active_fraction controls a DIFFERENT axis: what fraction of panels are
    active (amplifying, gain up to pmax_dB, injects its own noise) vs passive
    (pure phase reflector, no gain, no extra noise) -- PanelState.active, not
    PanelState.a. A random active_fraction of panels is chosen fresh each
    call via `rng`.
    """
    active_mask = rng.random(n_panels) < active_fraction
    system = build_system(
        n_panels=n_panels, k_users=k_users, k_targets=k_targets,
        active_mask=active_mask,
        L=L, M=M, p_total_linear=p_total_linear, rng=rng,
        target_p_block=0.0,
        panel_radius=panel_radius, target_radius=target_radius,
    )
    system.build_hbar()
    system.build_w_mrt()
    system.build_T()
    system.build_J()
    multi_target = len(system.targets) > 1
    avg_snr_linear, _ = system.compute_radar_snr(multi_target_interference=multi_target)
    return to_db(avg_snr_linear)


def run_sensitivity_sweep(
    freq_configs=None,
    panel_radius_configs=None,
    target_radius_configs=None,
    active_fraction_configs=None,
    n_panels=40,
    k_users=None,
    k_targets=2,
    n_trials=10,
    seed=123,
    out_dir="out/figs",
):
    """Grid sweep over (carrier frequency, panel radius, target radius,
    active-panel fraction), reporting mean ceiling SNR (dB) per combination,
    averaged over n_trials random geometries each. Restores the original
    beta_0_dB when done. Produces one heatmap figure (freq x panel/target
    radius) per active_fraction value, sharing one color scale for
    comparability across figures.
    """
    settings.load_config()
    k_users = k_users if k_users is not None else settings.config.channel_model.K_USERS
    L = settings.config.channel_model.L
    M = settings.config.channel_model.M
    p_total_linear = to_linear_dbm(settings.config.channel_model.P_max)
    original_beta_0_db = settings.config.channel_model.beta_0_dB

    if freq_configs is None:
        freq_configs = [
            ("28 GHz (current)", 28e9),
            ("5.9 GHz (C-band)", 5.9e9),
            ("1.5 GHz (L-band)", 1.5e9),
        ]
    if panel_radius_configs is None:
        panel_radius_configs = [("20-40m (current)", (20.0, 40.0)),
                                 ("10-20m", (10.0, 20.0)),
                                 ("5-10m", (5.0, 10.0))]
    if target_radius_configs is None:
        target_radius_configs = [("60-100m (current)", (60.0, 100.0)),
                                  ("30-50m", (30.0, 50.0)),
                                  ("15-25m", (15.0, 25.0))]
    if active_fraction_configs is None:
        active_fraction_configs = [
            ("0% active (passive-only, current)", 0.0),
            ("25% active", 0.25),
            ("100% active (all)", 1.0),
        ]

    rng = np.random.default_rng(seed)
    results = []

    try:
        for af_label, active_fraction in active_fraction_configs:
            for freq_label, freq_hz in freq_configs:
                beta_0_db = friis_beta_0_db(freq_hz)
                settings.config.channel_model.beta_0_dB = beta_0_db

                for pr_label, panel_radius in panel_radius_configs:
                    for tr_label, target_radius in target_radius_configs:
                        snrs = [
                            ceiling_snr_db(n_panels, k_users, k_targets, L, M, p_total_linear,
                                           panel_radius, target_radius, active_fraction, rng)
                            for _ in range(n_trials)
                        ]
                        mean_snr = float(np.mean(snrs))
                        results.append({
                            "active_fraction_label": af_label, "active_fraction": active_fraction,
                            "freq_label": freq_label, "freq_hz": freq_hz, "beta_0_db": beta_0_db,
                            "panel_radius_label": pr_label, "panel_radius": panel_radius,
                            "target_radius_label": tr_label, "target_radius": target_radius,
                            "mean_snr_db": mean_snr, "std_snr_db": float(np.std(snrs)),
                        })
                        print(f"{af_label:32s} | {freq_label:20s} | panel={pr_label:14s} | "
                              f"target={tr_label:14s} | beta_0_dB={beta_0_db:7.2f} "
                              f"| SNR={mean_snr:8.2f} dB (std={np.std(snrs):.2f})")
    finally:
        settings.config.channel_model.beta_0_dB = original_beta_0_db

    all_snrs = [r["mean_snr_db"] for r in results]
    vmin, vmax = min(all_snrs), max(all_snrs)

    fig_paths = {}
    for af_label, active_fraction in active_fraction_configs:
        subset = [r for r in results if r["active_fraction_label"] == af_label]
        safe_name = af_label.split(" ")[0].replace("%", "pct")
        fig_paths[af_label] = _plot_heatmaps(
            subset, freq_configs, panel_radius_configs, target_radius_configs, out_dir,
            vmin=vmin, vmax=vmax, suptitle_suffix=f" -- {af_label}",
            filename=f"geometry_sensitivity_heatmap_{safe_name}.png",
        )

    return results, fig_paths


def _plot_heatmaps(results, freq_configs, panel_radius_configs, target_radius_configs, out_dir,
                    vmin=None, vmax=None, suptitle_suffix="", filename="geometry_sensitivity_heatmap.png"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_freq = len(freq_configs)
    fig, axes = plt.subplots(1, n_freq, figsize=(5.2 * n_freq, 5.5), squeeze=False,
                              constrained_layout=True)
    axes = axes[0]

    pr_labels = [lbl for lbl, _ in panel_radius_configs]
    tr_labels = [lbl for lbl, _ in target_radius_configs]

    all_snrs = [r["mean_snr_db"] for r in results]
    if vmin is None:
        vmin = min(all_snrs)
    if vmax is None:
        vmax = max(all_snrs)

    im = None
    for idx, (ax, (freq_label, freq_hz)) in enumerate(zip(axes, freq_configs)):
        grid = np.full((len(pr_labels), len(tr_labels)), np.nan)
        for r in results:
            if r["freq_label"] != freq_label:
                continue
            i = pr_labels.index(r["panel_radius_label"])
            j = tr_labels.index(r["target_radius_label"])
            grid[i, j] = r["mean_snr_db"]

        im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(tr_labels)))
        ax.set_xticklabels(tr_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(pr_labels)))
        ax.set_yticklabels(pr_labels if idx == 0 else [], fontsize=8)
        ax.set_xlabel("Target radius", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Panel radius", fontsize=9)
        ax.set_title(f"{freq_label}", fontsize=11, fontweight="bold")

        for i in range(len(pr_labels)):
            for j in range(len(tr_labels)):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        color="black")

    fig.colorbar(im, ax=axes, label="Ceiling radar SNR (dB)", shrink=0.8, pad=0.02)
    fig.suptitle(f"Geometry/Frequency Sensitivity: Ceiling SNR (N=40, unblocked){suptitle_suffix}",
                 fontsize=13, fontweight="bold")

    fig_path = str(Path(out_dir) / filename)
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


if __name__ == "__main__":
    results, fig_paths = run_sensitivity_sweep()
    print()
    for label, path in fig_paths.items():
        print(f"{label}: {path}")
