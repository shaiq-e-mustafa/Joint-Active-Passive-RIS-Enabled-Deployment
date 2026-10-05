"""
Phase A, Weeks 2-3: greedy panel placement.

For each of N panel slots, evaluate many candidate positions and keep the one
maximizing worst-case (blockage-robust) radar SNR, holding all previously
placed panels fixed. Scoped as radar-only (see docs/phase_a_log.html Entry 9 --
communication is structurally insensitive to panel placement at this
project's channel scale regardless of frequency, so there's nothing for
placement to optimize on the comm side yet).

Performance note: the original Phase A plan's pseudocode rebuilds the WHOLE
system (every already-placed panel included) for every candidate evaluated --
at this project's measured per-build cost (Entry 3), that's roughly 40 hours
for the originally-planned scale. This implementation instead builds only the
ONE new candidate panel's channel per evaluation (build_one_panel_channel),
reusing already-placed panels' channels unchanged, which is the dominant
speedup; candidate count and blockage-trial count are also reduced from the
original spec for a first pass (see run_greedy_placement's defaults).
"""

import time
import numpy as np

from src.utils.config import settings
from src.utils.channel_utils import to_linear_dbm, to_db, sample_polar, random_rcs, bearing, distance
from src.sys.factory import build_one_panel_channel, make_direct_channel
from src.sys.system import RISPanel, UserLink, TargetLink, ISACSystem
from src.sys.channels import UserChannels
from src.channel.fading import get_hdk
from src.channel.risConfig import get_link_params, calibrate_lambda_b, is_blocked_line_boolean
from src.channel.channel_model import get_path_loss_linear
from src.sim.deployment import BS_POS, PANEL_RADIUS, TARGET_RADIUS, USER_RADIUS, TARGET_RCS_LIM
from src.opt.week1_panel_count import hybrid_snr_db, ProgressLogger


def build_users_and_targets(k_users, k_targets, M, rng, target_radius=None, user_radius=None):
    """Build fixed users/targets once for a greedy placement run -- their
    channels don't depend on which panels exist, so unlike panels they never
    need rebuilding during the search.
    """
    target_radius = target_radius if target_radius is not None else TARGET_RADIUS
    user_radius = user_radius if user_radius is not None else USER_RADIUS

    user_pos = [sample_polar(user_radius, (0, 2 * np.pi), rng) for _ in range(k_users)]
    target_pos = [sample_polar(target_radius, (0, 2 * np.pi), rng) for _ in range(k_targets)]
    target_rcs = [random_rcs((0.1, TARGET_RCS_LIM), rng) for _ in range(k_targets)]

    users_channels = [make_direct_channel(upos, M, rng) for upos in user_pos]
    users = [UserLink(user_id=k, channels=c, pos=p) for k, (c, p) in enumerate(zip(users_channels, user_pos))]
    targets = [TargetLink(target_id=k, rcs=target_rcs[k], pos=target_pos[k]) for k in range(k_targets)]
    return users, targets, user_pos, target_pos


def generate_candidates(n_candidates, panel_radius=None):
    """Evenly-spaced candidate positions around the panel ring (linear radius
    spacing x uniform angle spacing, matching the original Phase A pseudocode).
    """
    panel_radius = panel_radius if panel_radius is not None else PANEL_RADIUS
    r_min, r_max = panel_radius
    candidates = []
    for i in range(n_candidates):
        r = r_min + (r_max - r_min) * (i / n_candidates)
        theta = (2 * np.pi / n_candidates) * i
        candidates.append(np.array([r * np.cos(theta), r * np.sin(theta)]))
    return candidates


def worst_case_snr_for_panels(panels, users, targets, p_total_linear, target_p_block,
                               mean_obstacle_len, n_blockage_trials, snr_meas_ratio, rng,
                               direct_link_scale=1.0):
    """Worst-case radar SNR (min over n_blockage_trials resampled Line Boolean
    blockage patterns, see Entry 1/2) for a FIXED list of already-built
    RISPanel objects.
    """
    system = ISACSystem(panels=panels, users=users, p_total_linear=p_total_linear,
                         targets=targets, rng=rng, direct_link_scale=direct_link_scale)

    d_ref = 0.5 * (PANEL_RADIUS[0] + PANEL_RADIUS[1])
    lambda_b = calibrate_lambda_b(target_p_block, d_ref, mean_obstacle_len)

    snrs = []
    for _ in range(n_blockage_trials):
        for panel in system.panels:
            d = distance(BS_POS, panel.pos)
            blocked = is_blocked_line_boolean(d, rng, lambda_b, mean_obstacle_len)
            panel.state.a = 0 if blocked else 1
        system.build_hbar()
        system.build_w_mrt()
        system.build_T()
        system.build_J()
        snrs.append(hybrid_snr_db(system, snr_meas_ratio, rng))
    return min(snrs)


def run_greedy_placement(
    N=40,
    k_users=None,
    k_targets=2,
    n_candidates=50,
    n_blockage_trials=10,
    target_p_block=0.30,
    mean_obstacle_len=5.0,
    snr_meas_ratio=10.0,
    seed=42,
    log_path="out/week2_greedy.log",
    direct_link_scale=1.0,
):
    """Greedy placement: iteratively add the panel (from n_candidates ring
    positions) that maximizes worst-case radar SNR, holding previously-placed
    panels fixed. Returns (placed_panels, history) where history is a list of
    (position, worst_case_snr_db) in placement order.
    """
    if n_candidates < N:
        raise ValueError(f"n_candidates ({n_candidates}) must be >= N ({N}) -- each placed panel "
                          f"permanently removes one candidate from the pool, so there must be enough "
                          f"distinct candidate positions to place all N panels without reuse.")

    settings.load_config()
    k_users = k_users if k_users is not None else settings.config.channel_model.K_USERS
    L = settings.config.channel_model.L
    M = settings.config.channel_model.M
    p_total_linear = to_linear_dbm(settings.config.channel_model.P_max)

    rng = np.random.default_rng(seed)
    users, targets, user_pos, target_pos = build_users_and_targets(k_users, k_targets, M, rng)
    candidates = generate_candidates(n_candidates)

    log = ProgressLogger(log_path)
    log.write(f"Starting greedy placement: N={N}, n_candidates={n_candidates}, "
              f"n_blockage_trials={n_blockage_trials}, target_p_block={target_p_block}")

    placed_panels = []
    history = []
    available = list(range(len(candidates)))  # indices not yet used by a placed panel

    for k in range(N):
        best_score = -np.inf
        best_pos = None
        best_panel = None
        best_idx = None

        for idx in available:
            cand_pos = candidates[idx]
            channels, state = build_one_panel_channel(
                cand_pos, target_pos, user_pos, active=False, panel_active=1, L=L, M=M, rng=rng
            )
            candidate_panel = RISPanel(panel_id=k, channels=channels, state=state, pos=cand_pos)
            test_panels = placed_panels + [candidate_panel]

            score = worst_case_snr_for_panels(
                test_panels, users, targets, p_total_linear, target_p_block,
                mean_obstacle_len, n_blockage_trials, snr_meas_ratio, rng,
                direct_link_scale=direct_link_scale
            )
            if score > best_score:
                best_score = score
                best_pos = cand_pos
                best_panel = candidate_panel
                best_idx = idx

        available.remove(best_idx)  # a physical panel can't be placed twice at the same spot
        placed_panels.append(best_panel)
        history.append((best_pos.copy(), best_score))
        log.write(f"Panel {k + 1}/{N}: placed at ({best_pos[0]:.1f}, {best_pos[1]:.1f}), "
                  f"worst-case SNR = {best_score:.2f} dB")

    log.write(f"Greedy placement complete. Final worst-case SNR = {history[-1][1]:.2f} dB")
    log.close()

    return placed_panels, history


if __name__ == "__main__":
    placed_panels, history = run_greedy_placement()
    print(f"\nFinal: N={len(placed_panels)} panels, worst-case SNR = {history[-1][1]:.2f} dB")
