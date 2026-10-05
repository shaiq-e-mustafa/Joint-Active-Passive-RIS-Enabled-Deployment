"""Experiment library for the 100 m zone report (Entry 12). All SNRs in dB; radar yardstick = Albersheim
(Pd=0.9, Pfa=1e-6, one pulse) = +13.11 dB. Radar SNR is per sample at the config noise level; coherent
integration over n samples adds 10 log10 n (assumes a static target over the CPI)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.utils.channel_utils import to_db, to_linear_dbm
from src.channel.risConfig import calibrate_lambda_b, is_blocked_line_boolean
from src.channel.geometry3d import cfg
from src.sys import scenario as sc
from src.opt import placement3d as pl
from src.opt.phase_alignment import coherent_phases_upa
from src.sim.deployment import BS_POS
from src.sys.convention import echo_vec as _echo_vec, illumination
from src.sys import extended_target as ET

P30 = to_linear_dbm(settings.config.channel_model.P_max)
REQ_DB = 13.11
GRID = 72
THETAS = np.linspace(0, 2 * np.pi, GRID, endpoint=False)
E = np.exp(1j * THETAS)[None, :]


def blockage_lambda(p_block=0.30, d_ref=100.0, mean_len=5.0):
    return calibrate_lambda_b(p_block, d_ref, mean_len), mean_len


def make_scene(N, nx, ny, positions, users, targets, active_frac, rng, p_block=0.30, direct_link_scale=0.0, active_set=None):
    """Panels at `positions[:N]` + Line-Boolean BS-link blockage + a random active subset."""
    lam_b, ml = blockage_lambda(p_block)
    n_act = int(round(active_frac * N))
    act = set(rng.permutation(N)[:n_act].tolist()) if active_set is None else set(active_set)
    system = sc.build_system(positions[:N], users, targets, nx, ny, lambda i: i in act, P30, rng,
                             direct_link_scale=direct_link_scale, cross_panel_T=True)
    if str(cfg("bs_panel_blockage", "line_boolean")) == "none":     # siting guarantees line of sight to the BS
        for p in system.panels:
            p.state.a = 1
    else:
        for p in system.panels:
            p.state.a = 0 if is_blocked_line_boolean(np.linalg.norm(p.pos - BS_POS), rng, lam_b, ml) else 1
    if str(cfg("gain_mode", "random")) == "budget" and (active_frac > 0 or bool(act)):
        from src.opt import active_gains as _ag
        system.gain_report = _ag.assign_budget_gains(system)
    system.build_J()
    return system


def echo_vec(panel, phases, target):
    ph = np.exp(1j * phases)
    phi = panel.state.gains.astype(complex) * ph if panel.state.active else ph
    return _echo_vec(panel.channels.G, phi, panel.channels.b_by_target[target.target_id]).reshape(-1)


def offsets(gs, base, Jinv, sweeps=3):
    """Per-panel scalar phases maximizing |g|^2 g^H J^-1 g  (g = base + sum_i e^{j b_i} g_i), coordinate ascent."""
    betas = np.zeros(len(gs))
    tot = base + (sum(gs) if gs else 0)
    for _ in range(sweeps):
        for k, g in enumerate(gs):
            rest = tot - np.exp(1j * betas[k]) * g
            Gm = rest[:, None] + g[:, None] * E
            vals = np.einsum("ij,ij->j", Gm.conj(), Gm).real * np.einsum("ij,ij->j", Gm.conj(), Jinv @ Gm).real
            betas[k] = THETAS[int(np.argmax(vals))]
            tot = rest + np.exp(1j * betas[k]) * g
    return betas


def snr_from_vec(g, J, rcs, P=P30):
    return rcs * P * (np.linalg.norm(g) ** 2) * (g.conj() @ np.linalg.solve(J, g)).real


def _base(t):
    return t.direct.reshape(-1) if t.direct is not None else np.zeros(t.J.shape[0], complex)


def radar_timemux(system, q, alignment="intra+cross"):
    """Per-target matched-beam SNR with ALL selected panels aligned to target q (time-multiplexed CPIs).
    alignment: 'none' (random phases) | 'intra' (exact per-element phases) | 'intra+cross' (+ per-panel offsets)."""
    t = system.targets[q]
    base = _base(t)
    sel = [p for p in system.panels if p.state.a != 0]
    if alignment == "none":
        gs = [echo_vec(p, p.state.phases, t) for p in sel]
        betas = np.zeros(len(gs))
    else:
        gs = [echo_vec(p, coherent_phases_upa(p, t.pos, "radar"), t) for p in sel]
        betas = offsets(gs, base, np.linalg.inv(t.J)) if alignment == "intra+cross" else np.zeros(len(gs))
    g = base + sum(np.exp(1j * b) * x for b, x in zip(betas, gs)) if gs else base
    snr = to_db(snr_from_vec(g, t.J, t.rcs))
    return snr + ET.extended_loss_db(system, t, seed=q) if ET.is_extended() and sel else snr


def radar_simultaneous(system):
    """All targets served in the SAME CPI: each panel is aligned to the target it serves best, transmit power is
    split P/Q across one matched beam per target, and each target's return sees the others as interference.
    Returns the per-target SINR in dB."""
    Q = len(system.targets)
    sel = [p for p in system.panels if p.state.a != 0]
    aligned = [[echo_vec(p, coherent_phases_upa(p, t.pos, "radar"), t) for t in system.targets] for p in sel]
    assign = [int(np.argmax([np.linalg.norm(v) for v in row])) for row in aligned]
    phases = [coherent_phases_upa(p, system.targets[a].pos, "radar") for p, a in zip(sel, assign)]
    betas = np.zeros(len(sel))
    for q, t in enumerate(system.targets):
        idx = [i for i, a in enumerate(assign) if a == q]
        b = offsets([aligned[i][q] for i in idx], _base(t), np.linalg.inv(t.J))
        for i, bi in zip(idx, b):
            betas[i] = bi
    g_t = []
    for t in system.targets:
        g = _base(t).copy()
        for i, p in enumerate(sel):
            g = g + echo_vec(p, phases[i] + betas[i], t)
        g_t.append(g)
    ill = [illumination(g) for g in g_t]                       # transmit direction that maximizes each target's echo
    Rx = sum((system.p_total_linear / Q) * np.outer(v, v.conj()) / np.linalg.norm(v) ** 2 for v in ill)
    out = []
    for q, t in enumerate(system.targets):
        J = t.J.copy()
        for q2, t2 in enumerate(system.targets):
            if q2 != q:
                a = g_t[q2]
                J = J + t2.rcs * ((ill[q2].conj() @ Rx @ ill[q2]).real) * np.outer(a, a.conj())
        g = g_t[q]
        sinr = to_db(t.rcs * (ill[q].conj() @ Rx @ ill[q]).real * (g.conj() @ np.linalg.solve(J, g)).real)
        out.append(sinr + ET.extended_loss_db(system, t, seed=q) if ET.is_extended() and sel else sinr)
    return out


def direct_only_snr(target):
    g = target.direct.reshape(-1)
    J = np.eye(g.shape[0]) * 10 ** ((int(settings.config.channel_model.reciever_nosie) - 30) / 10)
    return to_db(snr_from_vec(g, J, target.rcs))


def scene_positions(strategy, N, rng, greedy_pos=None):
    zc = sc.ZONE_CENTER
    if strategy == "random":
        r = np.sqrt(8.0 ** 2 + (30.0 ** 2 - 8.0 ** 2) * rng.uniform(0, 1, N))
        a = rng.uniform(0, 2 * np.pi, N)
        return [zc + np.array([ri * np.cos(ai), ri * np.sin(ai)]) for ri, ai in zip(r, a)]
    if strategy == "greedy":
        return list(greedy_pos[:N])
    if strategy == "ring_bs":          # the pre-Entry-12 deployment: ring 20-40 m around the BS
        r = rng.uniform(20, 40, N)
        a = rng.uniform(0, 2 * np.pi, N)
        return [np.array([ri * np.cos(ai), ri * np.sin(ai)]) for ri, ai in zip(r, a)]
    raise ValueError(strategy)


def planned_greedy(nx=16, ny=16, n_select=50, n_candidates=150, n_design=10, seed=0):
    cands = pl.annulus_candidates(n_candidates)
    design = sc.disk_sample(sc.ZONE_CENTER, sc.ZONE_RADIUS, n_design, np.random.default_rng(seed))
    order = pl.greedy_order(cands, design, nx, ny, n_select, seed=seed)
    return [cands[i] for i in order]
