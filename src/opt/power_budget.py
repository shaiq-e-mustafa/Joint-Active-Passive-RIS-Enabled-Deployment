"""Network-wide RIS power budget (configs/default.yaml `ris_network_budget_w`).

Power accounting follows src/opt/active_gains.py (Long et al., arXiv:2103.00709): every element costs P_c; an active element costs P_c + P_DC,
and an active panel also draws the amplifier output  min(P_out, a^2 (P_in + L sigma_v^2)) / upsilon.

enforce_network_budget(system)  -- take a built system, measure the power drawn, and if it exceeds the budget downgrade active panels to passive
    (least benefit per extra watt first), then switch off the weakest panels (a = 0) if still over.
plan_under_budget(...)          -- planning-time knapsack-greedy on the coverage surrogate of placement3d: at each step choose the best of
    {add passive, add active, upgrade a passive to active} by coverage gain per watt, until the budget is spent.
"""
import numpy as np
from src.utils.channel_utils import to_linear
from src.channel.geometry3d import cfg, to3, upa_elements, panel_normal, link_3d, bs_array, point_end, ris_end
from src.channel.risConfig import calibrate_lambda_b, is_blocked_line_boolean
from src.sim.deployment import BS_POS
from src.sys import scenario as sc
from src.sys.convention import echo_vec, illumination
from src.opt import active_gains as ag
from src.opt import placement3d as pl
from src.opt.phase_alignment import coherent_phases_upa


def budget_w():
    b = cfg("ris_network_budget_w", None)
    return None if b is None else float(b)


def network_power_w(system):
    """Power drawn by the whole network now (W): per-panel circuits plus amplifier output / efficiency for active panels. Blocked panels still draw."""
    q = ag.params()
    sv = to_linear(float(cfg("active_ris_noise", -96.0)) - 30)
    beams = ag.target_beams(system) if system.targets else []
    total = 0.0
    for p in system.panels:
        if getattr(p.state, "off", False):
            continue
        L = p.state.phases.shape[0]
        if not p.state.active or p.state.gains is None:
            total += ag.panel_power_w(L, False)
            continue
        pin = (ag.incident_power_w(system, p, beams) if beams else 0.0) + L * sv
        a = float(np.max(p.state.gains))
        total += ag.panel_power_w(L, True) + min(ag.p_out_w(L), a ** 2 * pin) / q["upsilon"]
    return total


def _benefit(system, p):
    """Benefit proxy of one panel: norm of its passive, phase-aligned echo vector toward the best target."""
    best = 0.0
    for t in system.targets:
        phi = np.exp(1j * coherent_phases_upa(p, t.pos, "radar"))
        best = max(best, float(np.linalg.norm(echo_vec(p.channels.G, phi, p.channels.b_by_target[t.target_id]))))
    return best


def enforce_network_budget(system, budget=None):
    """Bring the drawn power under the budget. Returns dict(before_w, after_w, downgraded, switched_off). budget=None uses the config; unlimited -> no change."""
    B = budget_w() if budget is None else budget
    before = network_power_w(system)
    rep = dict(before_w=before, after_w=before, downgraded=0, switched_off=0)
    if B is None or before <= B:
        return rep
    cur = before
    # 0) blocked panels (no line of sight to the BS) are useless: power them down first
    for p in system.panels:
        if not p.state.a and not getattr(p.state, "off", False):
            p.state.off = True
    cur = network_power_w(system)
    sel = [p for p in system.panels if p.state.a and not getattr(p.state, "off", False)]
    ben = {id(p): _benefit(system, p) for p in sel}
    # 1) downgrade active -> passive, least benefit per extra watt first (the extra-watt cost is similar across panels, so rank by benefit)
    for p in sorted([p for p in sel if p.state.active], key=lambda p: ben[id(p)]):
        if cur <= B:
            break
        p.state.active = False
        p.state.gains = None
        cur = network_power_w(system)
        rep["downgraded"] += 1
    # 2) switch off the weakest panels
    for p in sorted(sel, key=lambda p: ben[id(p)]):
        if cur <= B:
            break
        p.state.a = 0
        p.state.off = True
        cur = network_power_w(system)
        rep["switched_off"] += 1
    rep["after_w"] = cur
    return rep


# ------------------------------------------------------------------------------------------------ planning
def _candidate_G(c, nx, ny):
    c = np.asarray(c, float)
    normal = panel_normal(c, BS_POS, sc.ZONE_CENTER)
    el = upa_elements(c, float(cfg("h_panel", 5.0)), normal, nx, ny)
    bs_el, bs_e = bs_array()
    return link_3d(el, ris_end(normal), bs_el, bs_e, None, float(cfg("eta_los", 2.0)), None)


def estimate_active_draw_w(G, g_cq, P):
    """Amplifier draw of an active panel (W) when the BS beams at the panel's best design point (planning-time estimate)."""
    q = ag.params()
    L = G.shape[0]
    sv = to_linear(float(cfg("active_ris_noise", -96.0)) - 30)
    pin = 0.0
    for g in g_cq:
        v = illumination(g[:, None])[:, 0]
        x = np.sqrt(P) * v / np.linalg.norm(v)
        pin = max(pin, float(np.sum(np.abs(G @ x) ** 2)))
    pin += L * sv
    return min(ag.p_out_w(L), q["a_max"] ** 2 * pin) / q["upsilon"]


def plan_under_budget(candidates, design_points, nx, ny, budget, max_panels=100, p_block=0.30, n_trials=24, seed=0, d_ref=100.0, mean_len=5.0, P=None):
    """Returns a list of (candidate_index, active) in the order they were chosen, total nominal power <= budget."""
    rng = np.random.default_rng(seed)
    P = float(P if P is not None else 10 ** ((float(cfg("P_max", 30.0)) - 30) / 10))
    L = nx * ny
    q = ag.params()
    nC, nQ = len(candidates), len(design_points)
    g = np.stack([pl.los_echo_vectors(c, design_points, nx, ny) for c in candidates])           # (C, Q, M)
    lam_b = calibrate_lambda_b(p_block, d_ref, mean_len)
    ok = np.array([[not is_blocked_line_boolean(np.linalg.norm(np.asarray(c) - BS_POS), rng, lam_b, mean_len) for c in candidates] for _ in range(n_trials)])
    p_pass = ag.panel_power_w(L, False)
    p_act = np.array([ag.panel_power_w(L, True) + estimate_active_draw_w(_candidate_G(c, nx, ny), g[i], P) for i, c in enumerate(candidates)])
    a_cap = q["a_max"]
    M = g.shape[2]

    def score(tot):
        pw = np.sum(np.abs(tot) ** 2, axis=2)
        return float(np.min(np.mean(20 * np.log10(np.maximum(pw, 1e-300)), axis=0)))

    tot = np.zeros((n_trials, nQ, M), dtype=complex)
    chosen, contrib, used = [], {}, 0.0                  # chosen: list of [candidate, active]; contrib[c]: unit-gain aligned contribution
    avail = list(range(nC))

    def aligned(vec_cq, tot_now, c):
        ip = np.einsum("tqm,qm->tq", tot_now.conj(), vec_cq)
        ph = np.where(np.abs(ip) > 0, np.exp(-1j * np.angle(ip)), 1.0)
        return vec_cq[None, :, :] * ph[:, :, None] * ok[:, c][:, None, None]

    while True:
        best = None
        if not chosen:                                         # first panel: best passive site that fits
            if p_pass > budget:
                break
            sc_best = -np.inf
            for c in avail:
                add = aligned(g[c], tot, c)
                s_new = score(add)
                if s_new > sc_best:
                    sc_best, best = s_new, ("add", c, False, p_pass, add)
        else:
            s_now = score(tot)
            best_ratio = 0.0
            if len(chosen) < max_panels:
                for c in avail:
                    for act in (False, True):
                        cost = p_act[c] if act else p_pass
                        if used + cost > budget:
                            continue
                        add = aligned(g[c] * (a_cap if act else 1.0), tot, c)
                        gain = score(tot + add) - s_now
                        if gain > 0 and gain / cost > best_ratio:
                            best, best_ratio = ("add", c, act, cost, add), gain / cost
            for idx, (c, act) in enumerate(chosen):
                if act:
                    continue
                cost = p_act[c] - p_pass
                if used + cost > budget:
                    continue
                add = (a_cap - 1.0) * contrib[c]
                gain = score(tot + add) - s_now
                if gain > 0 and gain / cost > best_ratio:
                    best, best_ratio = ("up", c, idx, cost, add), gain / cost
        if best is None:
            break
        kind, c = best[0], best[1]
        tot = tot + best[4]
        used += best[3]
        if kind == "add":
            chosen.append([c, best[2]])
            contrib[c] = best[4] / (a_cap if best[2] else 1.0)
            avail.remove(c)
        else:
            chosen[best[2]][1] = True
    return [(c, a) for c, a in chosen], used
