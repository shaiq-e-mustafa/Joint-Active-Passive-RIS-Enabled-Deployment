"""Planning-time panel placement for the 100 m zone scenario (Entry 12).

Greedy on a deterministic *coverage surrogate*: for every candidate site c and design point q (a possible
target location in the zone) the planner knows the LoS geometry, hence the echo vector the panel would
contribute if it aimed at q:  g_{c,q} = G_c^T (phi_q * b_{c,q})  (M-dim, exact spherical-wave phases, echo_vec of src/sys/convention.py).
Adding candidate c to the chosen set S gives, with its panel-level phase aligned to the running sum,
    |tot_q + e^{j beta} g_{c,q}|^2 = |tot_q|^2 + |g|^2 + 2 |tot_q^H g|            (if the BS link is not blocked)
Score(c) = min over design points q of  mean over blockage trials t of  10 log10 |tot_{t,q}|^4  (radar SNR ~ |g|^4).
No system rebuilds are needed, so thousands of evaluations take seconds. The panel set is nested: the first N
chosen sites are the greedy solution for N panels.
"""
import numpy as np
from src.channel.risConfig import calibrate_lambda_b, is_blocked_line_boolean
from src.channel.geometry3d import (cfg, to3, upa_elements, panel_normal, link_3d, bs_array, point_end, ris_end)
from src.sim.deployment import BS_POS
from src.sys.convention import echo_vec, design_phases
from src.sys import scenario as sc


def annulus_candidates(n, r_min=8.0, r_max=30.0, center=None):
    """Deterministic quasi-uniform (golden-angle) candidate sites in an annulus around the zone centre."""
    c = sc.ZONE_CENTER if center is None else np.asarray(center, float)
    i = np.arange(n) + 0.5
    r = np.sqrt(r_min ** 2 + (r_max ** 2 - r_min ** 2) * i / n)
    a = i * np.pi * (3 - np.sqrt(5))
    return [c + np.array([ri * np.cos(ai), ri * np.sin(ai)]) for ri, ai in zip(r, a)]


def los_echo_vectors(ppos, q_positions, nx, ny):
    """Deterministic LoS g_{c,q} for every design point q (aimed exactly at q). Returns (Q, M)."""
    ppos = np.asarray(ppos, float)
    normal = panel_normal(ppos, BS_POS, sc.ZONE_CENTER)
    el = upa_elements(ppos, float(cfg("h_panel", 5.0)), normal, nx, ny)
    end = ris_end(normal)
    bs_el, bs_e = bs_array()
    G = link_3d(el, end, bs_el, bs_e, None, float(cfg("eta_los", 2.0)), None)
    k = 2 * np.pi / sc.wavelength()
    d_bs = np.linalg.norm(el - bs_el.mean(axis=0), axis=1)
    out = []
    for q_pos in q_positions:
        q3 = to3(q_pos, float(cfg("h_target", 1.5)))[None, :]
        b = link_3d(el, end, q3, point_end(), None, float(cfg("eta_los", 2.0)), None)[:, 0]
        phi = np.exp(1j * design_phases(d_bs, np.linalg.norm(el - q3, axis=1), k, "radar"))
        out.append(echo_vec(G, phi, b[:, None])[:, 0])
    return np.stack(out)


def greedy_order(candidates, design_points, nx, ny, n_select, p_block=0.30, n_trials=24, seed=0, d_ref=100.0,
                 mean_len=5.0):
    """Returns the chosen candidate indices in greedy order (nested)."""
    rng = np.random.default_rng(seed)
    nC, nQ = len(candidates), len(design_points)
    g = np.stack([los_echo_vectors(c, design_points, nx, ny) for c in candidates])   # (C, Q, M)
    lam_b = calibrate_lambda_b(p_block, d_ref, mean_len)
    ok = np.array([[not is_blocked_line_boolean(np.linalg.norm(np.asarray(c) - BS_POS), rng, lam_b, mean_len)
                    for c in candidates] for _ in range(n_trials)])                                   # (T, C)
    M = g.shape[2]
    tot = np.zeros((n_trials, nQ, M), dtype=complex)
    chosen, avail = [], list(range(nC))
    for _ in range(min(n_select, nC)):                                                 # cannot choose more sites than there are candidates
        best, best_score = avail[0], -np.inf                                           # fallback if no candidate has a usable score
        for c in avail:
            gc = g[c]                                                                 # (Q, M)
            ip = np.einsum("tqm,qm->tq", tot.conj(), gc)                              # <tot, g>
            cross = 2 * np.abs(ip)
            new_pow = np.sum(np.abs(tot) ** 2, axis=2) + np.sum(np.abs(gc) ** 2, axis=1)[None, :] + cross
            old_pow = np.sum(np.abs(tot) ** 2, axis=2)
            pw = np.where(ok[:, c][:, None], new_pow, old_pow)                        # (T, Q)
            score = np.min(np.mean(20 * np.log10(np.maximum(pw, 1e-300)), axis=0))
            if score > best_score:
                best, best_score = c, score
        gc = g[best]
        ip = np.einsum("tqm,qm->tq", tot.conj(), gc)
        phase = np.where(np.abs(ip) > 0, np.exp(-1j * np.angle(ip)) * 1.0, 1.0)       # align g to the running sum
        add = gc[None, :, :] * phase[:, :, None] * ok[:, best][:, None, None]
        tot = tot + add
        chosen.append(best); avail.remove(best)
    return chosen
