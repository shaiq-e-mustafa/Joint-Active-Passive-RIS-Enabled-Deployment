"""
Phase A, Week 4 diagnostic: how much does closed-form, single-objective
coherent phase alignment help radar SNR or communication SINR, and how much
does each cost the OTHER objective? Answers whether the manifold-optimization
scope (needed only for the joint multi-target/multi-user case) is actually
justified, before spending time implementing it. See docs/phase_a_log.html
Orientation section, "Decision point" box, for the full reasoning.

This measures a CEILING: phases are computed from the true target/user
position, matching how Entry 6/7's ceiling_snr_db() already measures a
best-case (unblocked) scenario. A real deployment would need a search-then-
track loop to acquire that position in the first place -- explicitly out of
scope here, to be revisited in Phase B (online adaptation) if this diagnostic
shows the gain is worth chasing.
"""

import numpy as np

from src.utils.channel_utils import bearing
from src.sim.deployment import BS_POS


def coherent_phases(panel_pos, aim_pos, L, mode):
    """Per-element phase vector (radians) that coherently combines a panel's
    L elements toward `aim_pos`, derived from the exact steering-vector
    conventions in build_T()/build_hbar():

    mode="radar": build_T() applies Phi_i unconjugated (reflected_b = Phi_i @ b_i)
        phases_l = pi * l * (sin(theta_BS) - sin(theta_aim))

    mode="comm": build_hbar() applies Phi_i^H (M_i = G_i^H @ Phi_i^H)
        phases_l = pi * l * (sin(theta_aim) - sin(theta_BS))  =  -radar formula
    """
    theta_bs = bearing(panel_pos, BS_POS)
    theta_aim = bearing(panel_pos, aim_pos)
    l_idx = np.arange(L)
    base = np.pi * l_idx * (np.sin(theta_bs) - np.sin(theta_aim))
    if mode == "radar":
        return base
    elif mode == "comm":
        return -base
    else:
        raise ValueError(f"mode must be 'radar' or 'comm', got {mode!r}")


def set_coherent_phases(system, aim_pos, mode):
    """Set every panel's phases toward aim_pos (a target's or user's .pos),
    then rebuild everything downstream (h_bar, MRT beamformers, T, J).
    """
    for panel in system.panels:
        L = panel.state.phases.shape[0]
        panel.state.phases = coherent_phases(panel.pos, aim_pos, L, mode)

    system.build_hbar()
    system.build_w_mrt()
    system.build_T()
    system.build_J()
