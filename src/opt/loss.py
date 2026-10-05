"""Hard constraints, soft loss and feasibility projection for a Design (settings: configs/default.yaml, block `optimizer`).

HARD (a design that breaks one is infeasible, not penalized):
    power      network power drawn in the `power_percentile` scene <= ris_network_budget_w, and the design's nominal power <= the budget
    validity   extended-target intercepted power <= validity_max_intercept in the `validity_percentile` scene (worst target in each scene)
    sites      number of panels <= max_panels
    overlap    panels at least one panel-width apart
SOFT (weighted terms, each averaged over the worst `cvar_alpha` fraction of scenes = robustness):
    sensing    dB by which the worst target falls short of (detection requirement + radar_margin_db)
    comm       dB by which the worst user falls short of sinr_target_db
    power      power drawn / budget (a small push toward frugal designs)
    panel_cost, active_cost   per-panel / per-active-panel deployment cost (weights default to 0)
"""
from dataclasses import replace
import numpy as np
from src.channel.channel_model import wavelength
from src.opt import power_budget as pb
from src.opt.design import opt_cfg
from src.opt import evaluate as ev
from src.utils.detection import required_snr_db

BIG = 1.0e3
NAN_SHORTFALL_DB = 60.0          # a target / user that cannot be served at all counts as a 60 dB shortfall


def weights():
    w = opt_cfg("loss_weights")
    return {k: float(getattr(w, k, 0.0)) for k in ("sensing", "comm", "power", "panel_cost", "active_cost")}


def aggregate(draws):
    return dict(radar=np.array([d["radar_db"] for d in draws], float), sinr=np.array([d["sinr_db"] for d in draws], float),
                power=np.array([d["power_w"] for d in draws], float), frac=np.array([d["frac_ext"] for d in draws], float),
                rate=np.array([d["rate_sum"] for d in draws], float), n_los=np.array([d["n_los"] for d in draws], float))


def panel_width(design):
    return design.nx * wavelength() / 2


def overlap_pairs(design):
    w, S = panel_width(design), design.sites
    return [(i, j) for i in range(len(S)) for j in range(i + 1, len(S)) if np.linalg.norm(S[i] - S[j]) < w]


def constraint_report(design, draws):
    """Violations as non-negative relative excesses (0 = satisfied). `feasible` is True only if every excess is 0."""
    agg = aggregate(draws)
    B = pb.budget_w()
    lim = float(opt_cfg("validity_max_intercept", 0.10))
    p_pow = float(np.percentile(agg["power"], float(opt_cfg("power_percentile", 95))))
    p_val = float(np.percentile(agg["frac"], float(opt_cfg("validity_percentile", 95))))
    rep = dict(
        power_excess=0.0 if B is None else max(0.0, (p_pow - B) / B),
        nominal_power_excess=0.0 if B is None else max(0.0, (design.nominal_power_w() - B) / B),
        validity_excess=max(0.0, (p_val - lim) / lim),
        sites_excess=max(0.0, (design.n_panels - int(opt_cfg("max_panels", 100))) / float(opt_cfg("max_panels", 100))),
        overlap_excess=len(overlap_pairs(design)) / max(1, design.n_panels),                   # overlapping pairs per panel (relative, like the others)
    )
    rep["power_p"], rep["validity_p"] = p_pow, p_val
    rep["feasible"] = all(rep[k] <= 1e-12 for k in ("power_excess", "nominal_power_excess", "validity_excess", "sites_excess", "overlap_excess"))
    return rep


def _cvar(x, alpha):
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, int(np.ceil(alpha * len(x))))
    return float(np.mean(x[:k]))


def loss_terms(design, draws):
    agg = aggregate(draws)
    alpha = float(opt_cfg("cvar_alpha", 0.2))
    need = required_snr_db() + float(opt_cfg("radar_margin_db", 3))
    sens = np.maximum(0.0, need - np.where(np.isnan(agg["radar"]), need - NAN_SHORTFALL_DB, agg["radar"]))       # (D, Q)
    comm = np.maximum(0.0, float(opt_cfg("sinr_target_db", 10)) - np.where(np.isnan(agg["sinr"]), -1e3, agg["sinr"]))
    comm = np.minimum(comm, NAN_SHORTFALL_DB)
    B = pb.budget_w()
    terms = dict(
        sensing=_cvar(sens.max(axis=1), alpha),
        comm=_cvar(comm.max(axis=1), alpha),
        power=float(np.mean(agg["power"]) / B) if B else 0.0,
        panel_cost=float(design.n_panels), active_cost=float(design.n_active),
    )
    return terms


def total_loss(design, draws):
    """dict(loss, feasible, violations, terms). Infeasible designs get BIG * (1 + sum of excesses) + the soft loss, so a search can still rank them."""
    terms, rep, w = loss_terms(design, draws), constraint_report(design, draws), weights()
    soft = sum(w[k] * terms[k] for k in terms)
    if rep["feasible"]:
        return dict(loss=soft, feasible=True, violations=rep, terms=terms)
    exc = sum(rep[k] for k in ("power_excess", "nominal_power_excess", "validity_excess", "sites_excess", "overlap_excess"))
    return dict(loss=BIG * (1.0 + exc) + soft, feasible=False, violations=rep, terms=terms)


def evaluate_and_score(design, scenes):
    draws = ev.evaluate_design(design, scenes)
    out = total_loss(design, draws)
    out["draws"] = draws
    return out


# ------------------------------------------------------------------------------------------------ projection
def _subset(design, keep):
    keep = list(keep)
    return replace(design, sites=design.sites[keep], active=design.active[keep])


def project_to_feasible(design, scenes, max_iter=60):
    """Repair a design until every hard constraint holds. Returns (design, report, n_repairs).
    overlap: drop the later panel of each overlapping pair.  power: downgrade the weakest active panel, else remove the weakest panel.
    validity: downgrade the strongest active panel, else remove the strongest panel (strong panels drive the intercepted power)."""
    d = design.copy()
    w, keep = panel_width(d), []
    for i in range(d.n_panels):
        if all(np.linalg.norm(d.sites[i] - d.sites[j]) >= w for j in keep):
            keep.append(i)
    d = _subset(d, keep)
    cap = int(opt_cfg("max_panels", 100))
    if d.n_panels > cap:
        d = _subset(d, range(cap))
    n_rep = len(design.sites) - d.n_panels
    for _ in range(max_iter):
        if d.n_panels == 0:
            return d, dict(feasible=True), n_rep
        draws = ev.evaluate_design(d, scenes)
        rep = constraint_report(d, draws)
        if rep["feasible"]:
            return d, rep, n_rep
        system = ev.build_system(d, scenes[0])
        ben = np.array([pb._benefit(system, p) for p in system.panels])
        too_strong = rep["validity_excess"] > 0
        order = np.argsort(-ben) if too_strong else np.argsort(ben)             # strongest first for validity, weakest first for power
        act = [i for i in order if d.active[i]]
        if act:
            d.active[act[0]] = False
        else:
            d = _subset(d, [i for i in range(d.n_panels) if i != order[0]])
        n_rep += 1
    return d, constraint_report(d, ev.evaluate_design(d, scenes)), n_rep
