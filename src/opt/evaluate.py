"""Evaluate a Design on a fixed set of scene draws (users, targets, blockage, fading), reproducibly.

evaluate_design(design, scenes) -> list of per-scene dicts:
    radar_db    (Q,)  radar SNR per target, extended-target, dB. Split designs: R_x = comm beams + a matched sensing beam (share rho);
                      'joint': shared SDP design at the SINR target (nan if the SINR target is infeasible)
    sinr_db     (Q,)  worst user SINR for the design aimed at target q (split: sensing beam interferes; joint: = the target if feasible)
    rate_sum          sum over users of min(log2(1 + SINR), rate cap), worst target
    power_w           network power drawn in this scene (circuits + amplifier output / efficiency; blocked panels still draw)
    frac_ext          largest intercepted-power fraction over targets (extended target; conservative: all panels assumed aimed at the target)
    n_los             panels with line of sight to the BS
    n_radar           panels the role policy assigned to radar in this scene
    joint_ok          False if the joint SDP was infeasible for some target
Scenes are fixed by seed, so two designs are compared on identical users, targets, blockage and fading (paired).
"""
from dataclasses import dataclass
import numpy as np
from src.utils.channel_utils import to_db, to_linear, to_linear_dbm
from src.channel.geometry3d import cfg
from src.channel.risConfig import calibrate_lambda_b, is_blocked_line_boolean
from src.sim.deployment import BS_POS
from src.sys import scenario as sc
from src.sys import extended_target as ET
from src.sys.convention import echo_vec, comm_vec, illumination
from src.opt import beamforming as bf
from src.opt import joint_beamforming as jb
from src.opt import power_budget as pb
from src.opt import active_gains as ag
from src.opt.phase_alignment import coherent_phases_upa
from src.opt.design import opt_cfg


@dataclass
class Scene:
    users: list
    targets: list
    seed: int


def sample_scenes(n=None, seed=None, k_users=None, q_targets=None):
    n = int(opt_cfg("n_draws", 8)) if n is None else n
    seed = int(opt_cfg("scene_seed", 1234)) if seed is None else seed
    k = int(opt_cfg("n_users", 8)) if k_users is None else k_users
    q = int(opt_cfg("n_targets", 3)) if q_targets is None else q_targets
    out = []
    for d in range(n):
        users, targets = sc.make_users_targets(k, q, np.random.default_rng(seed + d), direct_mode="nlos")
        out.append(Scene(users, targets, seed + d))
    return out


def build_system(design, scene, p_block=0.30, d_ref=100.0, mean_len=5.0):
    """The scene's system for this design: panels, blockage (Line-Boolean unless bs_panel_blockage == none), budgeted active gains, J."""
    rng = np.random.default_rng(10_000 + scene.seed)
    P = to_linear_dbm(float(cfg("P_max", 30.0)))
    system = sc.build_system([np.asarray(p) for p in design.sites], scene.users, scene.targets, design.nx, design.ny,
                             lambda i: bool(design.active[i]), P, rng, direct_link_scale=0.0, cross_panel_T=True)
    if str(cfg("bs_panel_blockage", "line_boolean")) == "none":
        for p in system.panels:
            p.state.a = 1
    else:
        lam_b = calibrate_lambda_b(p_block, d_ref, mean_len)
        for p in system.panels:
            p.state.a = 0 if is_blocked_line_boolean(np.linalg.norm(p.pos - BS_POS), rng, lam_b, mean_len) else 1
    if str(cfg("gain_mode", "random")) == "budget" and design.n_active:
        system.gain_report = ag.assign_budget_gains(system)
    return system


def assign_roles(system, design):
    """Per-scene role of every panel: returns (radar bool array, radar phase sets, comm phase sets, advantage score log(r_i/c_i)).

    Each panel has two ways to be useful in THIS scene (reciprocal design k(d1+d2) in both):
      radar: focus on its best target;   benefit r_i = |echo vector| of that focus, element gains included
      comm : focus on its nearest user;  benefit c_i = |its contribution to that user's channel|, element gains included
    policy 'advantage': among panels with a line of sight to the BS (the others contribute nothing), the round(radar_share * usable)
    panels with the largest 2 log r_i - balance * log c_i serve radar, the rest comm. policy 'fixed_order': the first round(radar_share * N) panels in list order."""
    n = len(system.panels)
    ph_r, ph_c, score = [], [], np.zeros(n)
    for i, p in enumerate(system.panels):
        amp = p.state.gains.astype(float) if (p.state.active and p.state.gains is not None) else np.ones(p.channels.G.shape[0])   # element gains (1 if passive)
        cand = [coherent_phases_upa(p, t.pos, "radar") for t in system.targets]
        nrm = [np.linalg.norm(echo_vec(p.channels.G, amp * np.exp(1j * c), p.channels.b_by_target[t.target_id])) for c, t in zip(cand, system.targets)]
        q = int(np.argmax(nrm))
        k = int(np.argmin([np.linalg.norm(u.pos - p.pos) for u in system.users]))
        pc = coherent_phases_upa(p, system.users[k].pos, "comm")
        c_i = np.linalg.norm(comm_vec(p.channels.G, amp * np.exp(1j * pc), p.channels.f_by_user[k]))
        ph_r.append(cand[q]); ph_c.append(pc)
        # The echo passes the panel twice (radar SNR ~ |u|^4), the comm link once (SINR ~ |h|^2), so in dB-like terms the advantage is
        # 2 log r_i - log c_i (balance = 1; the policy may weight the comm term differently). A panel that cannot help radar at all (r_i = 0) is ranked last, one that cannot help comm (c_i = 0) first.
        score[i] = -np.inf if nrm[q] <= 0 else (np.inf if c_i <= 0 else 2 * np.log(nrm[q]) - design.balance * np.log(c_i))
    radar = np.zeros(n, bool)
    if design.policy == "fixed_order":
        radar[: int(round(design.radar_share * n))] = True
    elif design.policy == "advantage":
        usable = np.array([bool(p.state.a) for p in system.panels])
        idx = np.flatnonzero(usable)
        k_r = int(round(design.radar_share * len(idx)))
        radar[idx[np.argsort(-score[idx], kind="stable")[:k_r]]] = True
    else:
        raise ValueError(f"unknown role policy {design.policy!r}")
    return radar, ph_r, ph_c, score


def _aim(system, design):
    """Set every panel's phases from this scene's role assignment (see assign_roles)."""
    radar, ph_r, ph_c, _ = assign_roles(system, design)
    for i, p in enumerate(system.panels):
        p.state.phases = ph_r[i] if radar[i] else ph_c[i]
    return radar


def evaluate_scene(design, scene, sinr_target_db=None, rate_cap=None):
    sinr_target_db = float(opt_cfg("sinr_target_db", 10)) if sinr_target_db is None else sinr_target_db
    rate_cap = float(opt_cfg("rate_cap_bps_hz", 8)) if rate_cap is None else rate_cap
    system = build_system(design, scene)
    P = system.p_total_linear
    N0 = to_linear(float(cfg("reciever_nosie", -96)) - 30)
    SV = to_linear(float(cfg("active_ris_noise", -96.0)) - 30)
    n_radar = int(_aim(system, design).sum())
    system.build_hbar()
    system.build_J()
    H = np.hstack([u.h_bar for u in system.users])
    noise = np.array([N0 + sum(SV * np.sum(np.abs(p.state.phi_vec) ** 2 * np.abs(p.channels.f_by_user[k][:, 0]) ** 2)
                               for p in system.panels if p.state.active and p.state.a) for k in range(len(system.users))])
    radar, sinr, fracs, rates, joint_ok = [], [], [], [], True
    links = ET.panel_links(system)                       # target-independent: computed once per scene, shared by all targets
    for t in system.targets:
        u = np.zeros((H.shape[0], 1), dtype=complex) if t.direct is None else t.direct.reshape(-1, 1).astype(complex).copy()
        for p in system.panels:
            if p.state.a:
                u = u + echo_vec(p.channels.G, p.state.phi_vec, p.channels.b_by_target[t.target_id])
        ill = illumination(u)[:, 0]
        nrm = np.linalg.norm(ill)
        if nrm == 0:
            radar.append(float("nan")); sinr.append(float("nan")); fracs.append(0.0); rates.append(0.0)
            continue
        ill = ill / nrm
        # matched-beam SNR = rcs (ill_unnorm^H Rx ill_unnorm)(u^H J^-1 u) with ill_unnorm = |u| ill, hence the |u|^2 factor below
        quad = float((u[:, 0].conj() @ np.linalg.solve(t.J, u[:, 0])).real) * float(nrm ** 2)
        v = ET.validity(system, t, seed=scene.seed, links=links)                                     # conservative bound: all panels aimed at t
        fracs.append(v["frac_ext"])
        ext_db = ET.extended_loss_db(system, t, seed=scene.seed, current=True, links=links) if ET.is_extended() else 0.0   # focus as evaluated
        if design.beamformer == "joint" and np.linalg.norm(H) >= 1e-30:
            r = jb.joint_design(H, ill, P, noise, 10 ** (sinr_target_db / 10))
            if not r["status"].startswith("optimal"):
                joint_ok = False
                radar.append(float("nan")); sinr.append(float("nan")); rates.append(0.0)
                continue
            Rx = r["Rx"]
            sn = jb.user_sinr(H, r["W"], r["R0"], noise)
        else:
            Pc, Ps = ((1 - design.rho) * P, design.rho * P) if design.beamformer != "joint" else (P, 0.0)
            R0 = Ps * np.outer(ill, ill.conj())
            if Pc <= 0 or np.linalg.norm(H) < 1e-30:     # all power on sensing, or no usable user channel: no comm signal at all
                V = np.zeros((H.shape[0], H.shape[1]), dtype=complex)
            else:
                V = bf.rzf(H, Pc, noise) if design.beamformer == "rzf" else bf.wmmse(H, Pc, noise, R0=R0)
            Rx = V @ V.conj().T + R0
            sn = bf.sinr(H, V, noise, R0)
        f = float(np.real(ill.conj() @ Rx @ ill))
        radar.append(to_db(max(t.rcs * f * quad, 1e-30)) + ext_db)
        sinr.append(float(10 * np.log10(max(sn.min(), 1e-30))))
        rates.append(float(np.sum(np.minimum(np.log2(1 + sn), rate_cap))))
    return dict(radar_db=np.array(radar), sinr_db=np.array(sinr), rate_sum=float(np.min(rates)) if rates else 0.0,
                power_w=pb.network_power_w(system), frac_ext=float(np.max(fracs)) if fracs else 0.0,
                n_los=int(sum(p.state.a for p in system.panels)), n_radar=n_radar, joint_ok=joint_ok)


def evaluate_design(design, scenes, **kw):
    return [evaluate_scene(design, s, **kw) for s in scenes]
