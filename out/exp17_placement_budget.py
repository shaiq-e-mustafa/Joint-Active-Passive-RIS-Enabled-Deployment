"""Entry 17: placement re-run under the network power budget (configs/default.yaml `ris_network_budget_w`).

  python out/exp17_placement_budget.py plan   -> out/exp17_placement_budget/designs.json   (budget-aware planner at 1, 2, 4 W)
  python out/exp17_placement_budget.py eval   -> out/exp17_placement_budget/results.json   (planned designs + equal-power baselines, scored)

plan_under_budget (src/opt/power_budget.py) chooses WHICH sites are used and WHICH of them are active under the power budget, on the
deterministic coverage surrogate of placement3d. eval scores every design with the optimizer's evaluator (src/opt/evaluate.py) on the same
fixed scenes (users, targets, blockage, fading are paired), under the hard limits of src/opt/loss.py.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from concurrent.futures import ProcessPoolExecutor
import numpy as np

OUT = ROOT / "out" / "exp17_placement_budget"
BUDGETS = [1.0, 2.0, 4.0]


def _init():
    from src.utils.config import settings
    settings.load_config()


def _plan(B):
    from src.opt import design as dz
    t0 = time.time()
    d = dz.budget_design(budget_w=B, beamformer="joint", radar_fraction=0.5, max_panels=100)
    return dict(budget_w=B, sites=d.sites.tolist(), active=d.active.tolist(), n_panels=d.n_panels, n_active=d.n_active,
                nominal_w=d.nominal_power_w(), seconds=time.time() - t0)


def stage_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(len(BUDGETS), initializer=_init) as ex:
        res = list(ex.map(_plan, BUDGETS))
    for r in res:
        print(f"budget {r['budget_w']:.1f} W: {r['n_panels']} panels ({r['n_active']} active), nominal {r['nominal_w']:.3f} W, {r['seconds']:.0f} s")
    json.dump(res, open(OUT / "designs.json", "w"))


def _make(spec, planned):
    from src.opt import design as dz
    kind = spec["kind"]
    if kind == "planned":
        p = planned[spec["budget_w"]]
        return dz.Design(p["sites"], p["active"], 0.5, 16, 16, spec.get("beamformer", "joint"), spec.get("rho", 0.0), spec.get("policy", "advantage"), spec.get("balance", 0.0))
    if kind == "greedy":
        return dz.greedy_design(spec["n"], active_fraction=spec["active_fraction"], radar_fraction=0.5, beamformer="joint")
    if kind == "random":
        return dz.random_design(spec["n"], active_fraction=0.0, radar_fraction=0.5, beamformer="joint", seed=0)
    raise ValueError(kind)


def _eval(args):
    spec, planned = args
    from src.opt import evaluate as ev, loss as ls
    t0 = time.time()
    d = _make(spec, planned)
    scenes = ev.sample_scenes()
    r = ls.evaluate_and_score(d, scenes)
    dr = r["draws"]
    radar = np.array([x["radar_db"] for x in dr], float)
    sinr = np.array([x["sinr_db"] for x in dr], float)
    return dict(label=spec["label"], n_panels=d.n_panels, n_active=d.n_active, nominal_w=d.nominal_power_w(), feasible=bool(r["feasible"]),
                power_p95=r["violations"]["power_p"], intercept_p95=r["violations"]["validity_p"],
                power_excess=r["violations"]["power_excess"], validity_excess=r["violations"]["validity_excess"],
                sensing_short_db=r["terms"]["sensing"], comm_short_db=r["terms"]["comm"], loss=r["loss"],
                radar_mean_db=float(np.nanmean(radar)), radar_min_db=float(np.nanmin(radar)), sinr_min_db=float(np.nanmin(sinr)) if np.isfinite(sinr).any() else float("nan"),
                n_los_mean=float(np.mean([x["n_los"] for x in dr])), n_radar_mean=float(np.mean([x["n_radar"] for x in dr])),
                joint_ok=bool(all(x["joint_ok"] for x in dr)), seconds=time.time() - t0)


def stage_eval():
    planned = {r["budget_w"]: r for r in json.load(open(OUT / "designs.json"))}
    n_pass = int(2.0 / 0.0256)            # passive panels that fit in the 2 W budget (circuits only)
    specs = [
        dict(label="planner 2 W (role rule: advantage, balance 0)", kind="planned", budget_w=2.0, policy="advantage", balance=0.0),
        dict(label="planner 2 W (role rule: first half radar)", kind="planned", budget_w=2.0, policy="fixed_order"),
        dict(label="planner 2 W, RZF + 2% sensing beam (Entry 16 row)", kind="planned", budget_w=2.0, beamformer="rzf", rho=0.02, policy="advantage", balance=0.0),
        dict(label=f"greedy, {n_pass} passive panels (fills 2 W)", kind="greedy", n=n_pass, active_fraction=0.0),
        dict(label="greedy, 15 active panels (fills 2 W)", kind="greedy", n=15, active_fraction=1.0),
        dict(label="greedy, 20 panels, half active (Entry 16 best)", kind="greedy", n=20, active_fraction=0.5),
        dict(label=f"random, {n_pass} passive panels", kind="random", n=n_pass),
        dict(label="planner 1 W", kind="planned", budget_w=1.0, policy="advantage", balance=0.0),
        dict(label="planner 4 W", kind="planned", budget_w=4.0, policy="advantage", balance=0.0),
    ]
    with ProcessPoolExecutor(min(8, len(specs)), initializer=_init) as ex:
        res = list(ex.map(_eval, [(s, planned) for s in specs]))
    json.dump(res, open(OUT / "results.json", "w"))
    print(f"{'design':52s} {'N(act)':>8s} {'nom W':>6s} {'feas':>5s} {'pwr95':>6s} {'icpt95':>7s} {'radar short':>11s} {'comm short':>10s} {'loss':>8s} {'radar mean':>10s} {'min SINR':>8s} {'s':>5s}")
    for r in res:
        print(f"{r['label']:52s} {r['n_panels']:4d}({r['n_active']:2d}) {r['nominal_w']:6.2f} {str(r['feasible']):>5s} {r['power_p95']:6.2f} {100*r['intercept_p95']:6.1f}% {r['sensing_short_db']:11.1f} {r['comm_short_db']:10.1f} {r['loss']:8.1f} {r['radar_mean_db']:10.1f} {r['sinr_min_db']:8.1f} {r['seconds']:5.0f}")


if __name__ == "__main__":
    {"plan": stage_plan, "eval": stage_eval}[sys.argv[1]]()
