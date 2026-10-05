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
from src.channel.channel_model import wavefront_is_exact, element_positions, wavelength
from src.sys.convention import echo_vec, comm_vec, design_phases


def coherent_phases(panel_pos, aim_pos, L, mode):
    """Per-element phase vector (radians) that coherently combines a panel's
    L elements toward `aim_pos`, derived from the exact steering-vector
    conventions in build_T()/build_hbar(). The sin() formulas below are the LEGACY planar path (wavefront: planar) in the hermitian
    convention; with wavefront: exact the design is design_phases() of src/sys/convention.py (k (d1 + d2) when reciprocal).

    mode="radar": build_T() applies Phi_i unconjugated (reflected_b = Phi_i @ b_i)
        phases_l = pi * l * (sin(theta_BS) - sin(theta_aim))

    mode="comm": build_hbar() applies Phi_i^H (M_i = G_i^H @ Phi_i^H)
        phases_l = pi * l * (sin(theta_aim) - sin(theta_BS))  =  -radar formula
    """
    if wavefront_is_exact():
        # Spherical-wave design (Entry 12): cancel each element's true path-length difference.
        # reciprocal convention: the echo u = G^T Phi b has terms e^{-jk d1_l} e^{j phi_l} e^{-jk d2_l}  ->  phi_l = k (d1_l + d2_l) for radar
        # and comm alike (design_phases). The legacy hermitian convention gave radar k (d2_l - d1_l) and comm the negative.
        el = element_positions(panel_pos, L)
        d1 = np.linalg.norm(el - np.asarray(BS_POS, float), axis=1)
        d2 = np.linalg.norm(el - np.asarray(aim_pos, float), axis=1)
        return design_phases(d1, d2, 2 * np.pi / wavelength(), mode)
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


def coherent_phases_upa(panel, aim_pos2, mode):
    """Spherical-wave phase design for a 3-D UPA panel (same derivation as the exact ULA case, with the
    panel's true element positions, the BS array centre, and the aim point at target/user height)."""
    from src.channel.geometry3d import cfg, to3, bs_array
    bs_el, _ = bs_array()
    bs_c = bs_el.mean(axis=0)
    h = float(cfg("h_target", 1.5)) if mode == "radar" else float(cfg("h_user", 1.5))
    aim = to3(aim_pos2, h)
    d1 = np.linalg.norm(panel.elements - bs_c, axis=1)
    d2 = np.linalg.norm(panel.elements - aim, axis=1)
    return design_phases(d1, d2, 2 * np.pi / wavelength(), mode)


def set_coherent_phases(system, aim_pos, mode):
    """Set every panel's phases toward aim_pos (a target's or user's .pos),
    then rebuild everything downstream (h_bar, MRT beamformers, T, J).
    """
    for panel in system.panels:
        L = panel.state.phases.shape[0]
        if getattr(panel, "elements", None) is not None:
            panel.state.phases = coherent_phases_upa(panel, aim_pos, mode)
        else:
            panel.state.phases = coherent_phases(panel.pos, aim_pos, L, mode)

    system.build_hbar()
    system.build_w_mrt()
    system.build_T()
    system.build_J()


def cross_panel_phase_offsets(system, target_idx=0, sweeps=3, grid=72, matched=False):
    """Per-panel SCALAR phase offsets beta_i (added to all L element phases of panel i, on top of
    the intra-panel formula) that align the panels' M-dim contributions g_i = G_i^H Phi_i b_i so
    they add in the all-pairs operator T = (sum g_i)(sum g_i)^H (needs system.cross_panel_T=True
    to be rewarded by the SNR). Coordinate ascent on (g^H Rx g)(g^H J^-1 g); not globally optimal.
    Returns (panel_indices, betas). Call after set_coherent_phases(); then apply_cross_panel_offsets().
    """
    target = system.targets[target_idx]
    Rx = system.get_Rx()
    Jinv = np.linalg.inv(target.J)
    idx, gs = [], []
    for i, p in enumerate(system.panels):
        if p.state.a == 0:
            continue
        b = p.channels.b_by_target[target.target_id]
        gs.append(echo_vec(p.channels.G, p.state.phi_vec, b).reshape(-1))
        idx.append(i)
    thetas = np.linspace(0, 2 * np.pi, grid, endpoint=False)
    E = np.exp(1j * thetas)[None, :]
    betas = np.zeros(len(gs))
    g_direct = target.direct.reshape(-1) if getattr(target, "direct", None) is not None else 0.0
    tot = g_direct + sum(gs)
    for _ in range(sweeps):
        for k, g in enumerate(gs):
            base = tot - np.exp(1j * betas[k]) * g
            G = base[:, None] + g[:, None] * E
            if matched:     # best single transmit beam: (g^H Rx g) -> P |g|^2
                vals = (np.einsum("ij,ij->j", G.conj(), G).real
                        * np.einsum("ij,ij->j", G.conj(), Jinv @ G).real)
            else:
                vals = (np.einsum("ij,ij->j", G.conj(), Rx @ G).real
                        * np.einsum("ij,ij->j", G.conj(), Jinv @ G).real)
            betas[k] = thetas[int(np.argmax(vals))]
            tot = base + np.exp(1j * betas[k]) * g
    return idx, betas


def cross_panel_comm_offsets(system, user_idx=0, sweeps=3, grid=72):
    """Comm analogue: per-panel scalar offsets so every panel's RIS contribution r_i =
    G_i^H Phi_i^H f_i adds in phase with the others (and with the direct link, if present),
    maximizing ||h_bar||^2 (the MRT signal power). build_hbar() uses Phi^H, so the offset to
    ADD to a panel's phases is -beta_i. Returns (panel_indices, offsets_to_add).
    """
    user = system.users[user_idx]
    thetas = np.linspace(0, 2 * np.pi, grid, endpoint=False)
    E = np.exp(1j * thetas)[None, :]
    idx, rs = [], []
    for i, p in enumerate(system.panels):
        if p.state.a == 0:
            continue
        rs.append(comm_vec(p.channels.G, p.state.phi_vec, p.channels.f_by_user[user.user_id]).reshape(-1))
        idx.append(i)
    base0 = system.direct_link_scale * user.channels.hdk.reshape(-1)
    betas = np.zeros(len(rs))
    tot = base0 + sum(rs)
    for _ in range(sweeps):
        for k, r in enumerate(rs):
            base = tot - np.exp(1j * betas[k]) * r
            vals = np.linalg.norm(base[:, None] + r[:, None] * E, axis=0)
            betas[k] = thetas[int(np.argmax(vals))]
            tot = base + np.exp(1j * betas[k]) * r
    return idx, -betas


def apply_cross_panel_offsets(system, idx, betas):
    for i, b in zip(idx, betas):
        system.panels[i].state.phases = system.panels[i].state.phases + b
    system.build_hbar()
    system.build_w_mrt()
    system.build_T()
    system.build_J()
