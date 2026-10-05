"""Entry 17 diagnostics: (a) why the random 78-panel design is infeasible, (b) solver status / achieved SINR of the 2 W planner design,
(c) the 1 W and 4 W planner designs scored against THEIR OWN budget.   python out/exp17b_diagnostics.py"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from concurrent.futures import ProcessPoolExecutor
import numpy as np

OUT = ROOT / "out" / "exp17_placement_budget"


def _init():
    from src.utils.config import settings
    settings.load_config()


def planned(B):
    from src.opt import design as dz
    p = {r["budget_w"]: r for r in json.load(open(OUT / "designs.json"))}[B]
    return dz.Design(p["sites"], p["active"], 0.5, 16, 16, "joint", 0.0, "advantage", 0.0)


def job_overlap(_):
    from src.opt import design as dz, loss as ls
    out = {}
    for name, d in (("random 78", dz.random_design(78, active_fraction=0.0, radar_fraction=0.5, beamformer="joint", seed=0)), ("greedy 78 passive", dz.greedy_design(78, active_fraction=0.0, radar_fraction=0.5, beamformer="joint")), ("planner 2 W", planned(2.0))):
        out[name] = dict(n=d.n_panels, overlap_pairs=len(ls.overlap_pairs(d)), min_dist=float(min(np.linalg.norm(d.sites[i] - d.sites[j]) for i in range(d.n_panels) for j in range(i + 1, d.n_panels))),
                         panel_width_m=ls.panel_width(d))
    return out


def job_status(_):
    from src.opt import evaluate as ev, joint_beamforming as jb
    rec, orig = [], jb.joint_design

    def wrapped(*a, **k):
        r = orig(*a, **k)
        rec.append(dict(status=r.get("status")))
        return r
    jb.joint_design = wrapped
    d = planned(2.0)
    scenes = ev.sample_scenes()
    per_scene = []
    for s in scenes:
        n0 = len(rec)
        r = ev.evaluate_scene(d, s)
        per_scene.append(dict(statuses=[x["status"] for x in rec[n0:]], sinr_db=[float(v) for v in r["sinr_db"]], radar_db=[float(v) for v in r["radar_db"]], joint_ok=bool(r["joint_ok"])))
    return per_scene


def job_budget(B):
    from src.utils.config import settings
    settings.config.channel_model.ris_network_budget_w = B          # judge this design against its own budget
    from src.opt import evaluate as ev, loss as ls
    d = planned(B)
    r = ls.evaluate_and_score(d, ev.sample_scenes())
    v = r["violations"]
    return dict(budget_w=B, n_panels=d.n_panels, n_active=d.n_active, feasible=bool(r["feasible"]), power_p95=v["power_p"], intercept_p95=v["validity_p"],
                power_excess=v["power_excess"], validity_excess=v["validity_excess"], sensing_short_db=r["terms"]["sensing"], comm_short_db=r["terms"]["comm"], loss=r["loss"],
                radar_mean_db=float(np.nanmean([x["radar_db"] for x in r["draws"]])))


if __name__ == "__main__":
    with ProcessPoolExecutor(4, initializer=_init) as ex:
        f_ov, f_st = ex.submit(job_overlap, 0), ex.submit(job_status, 0)
        f_b = [ex.submit(job_budget, B) for B in (1.0, 4.0)]
        ov, st, bud = f_ov.result(), f_st.result(), [f.result() for f in f_b]
    print("== (a) overlap check (rule: panels at least one panel width apart)")
    for k, v in ov.items():
        print(f"  {k:18s} N={v['n']:3d} overlapping pairs={v['overlap_pairs']:3d} closest pair {v['min_dist']:.2f} m (panel width {v['panel_width_m']:.2f} m)")
    print("== (b) planner 2 W: joint-SDP status and achieved worst-user SINR per scene (target 10 dB)")
    for i, s in enumerate(st):
        print(f"  scene {i}: statuses {sorted(set(s['statuses']))} | SINR per target {[round(x, 2) for x in s['sinr_db']]} | radar {[round(x, 1) for x in s['radar_db']]} dB | joint_ok {s['joint_ok']}")
    print("== (c) planner designs judged against their own budget")
    for b in bud:
        print(f"  {b['budget_w']:.0f} W: {b['n_panels']} panels ({b['n_active']} active) feasible={b['feasible']} power p95 {b['power_p95']:.2f} W intercept p95 {100*b['intercept_p95']:.1f}% "
              f"(excess: power {b['power_excess']:.3f}, validity {b['validity_excess']:.3f}) radar short {b['sensing_short_db']:.1f} comm short {b['comm_short_db']:.1f} loss {b['loss']:.1f} radar mean {b['radar_mean_db']:.1f} dB")
    json.dump(dict(overlap=ov, status=st, budget=bud), open(OUT / "diagnostics.json", "w"))
