"""Optimizer groundwork checks (Design, evaluator, hard constraints, loss, projection). Run: python tests/test_optimizer_groundwork.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "out"))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.opt import design as dz, evaluate as ev, loss as ls, active_gains as ag
from src.sys import scenario as sc
import exp12_lib as X

ok = True
cm = settings.config.channel_model
oc = settings.config.optimizer


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


L = 256
# 1. Design basics
d = dz.random_design(10, active_fraction=0.3, seed=1)
exp_power = 10 * ag.panel_power_w(L, False) + 3 * (ag.panel_power_w(L, True) - ag.panel_power_w(L, False))
check("nominal power arithmetic", abs(d.nominal_power_w() - exp_power) < 1e-12, f"({1e3*d.nominal_power_w():.1f} mW)")
v = d.vector()
d = d.copy(radar_share=0.3, balance=0.4)
v = d.vector()
d2 = dz.Design.from_vector(v, d.n_panels, d)
check("vector round trip", np.allclose(d2.sites, d.sites) and (d2.active == d.active).all() and d2.radar_share == d.radar_share and d2.balance == d.balance and d2.policy == d.policy)
check("policy parameters are 2 numbers, not one label per panel", len(v) == 3 * d.n_panels + 2, f"(vector length {len(v)} for {d.n_panels} panels)")

# 2. scenes are deterministic and paired
sA, sB, sC = ev.sample_scenes(2, 7, 4, 1), ev.sample_scenes(2, 7, 4, 1), ev.sample_scenes(2, 8, 4, 1)
check("same seed -> same scenes", np.allclose(sA[0].targets[0].pos, sB[0].targets[0].pos) and not np.allclose(sA[0].targets[0].pos, sC[0].targets[0].pos))

# 3. evaluator == reference pipeline (one target, all panels radar, all power on the matched sensing beam, passive -> rcs P |u|^2 u^H J^-1 u)
scenes = ev.sample_scenes(2, 11, 4, 1)
_tm = cm.target_model
cm.target_model = 'point'          # compare the focus-point SNR exactly (the extended loss uses random scatterer positions)
pos = X.scene_positions("greedy", 12, np.random.default_rng(1), X.planned_greedy(16, 16, 12))
dsn = dz.Design(pos, np.zeros(12, bool), 1.0, 16, 16, "rzf", rho=1.0)      # radar_share 1: every usable panel serves radar
res = ev.evaluate_design(dsn, scenes)
for sc_, r in zip(scenes, res):
    system = ev.build_system(dsn, sc_)
    ref = X.radar_timemux(system, 0, "intra")
    check("evaluator radar SNR == radar_timemux (reference pipeline)", abs(r["radar_db"][0] - ref) < 1e-6, f"({r['radar_db'][0]:.3f} vs {ref:.3f} dB)")
    break
cm.target_model = _tm
check("intercepted fraction is positive and finite", all(0 < r["frac_ext"] < 1e3 for r in res))

# 4. constraints
scenes = ev.sample_scenes(3, 21, 4, 2)
small = dz.random_design(6, active_fraction=0.0, seed=2)
draws = ev.evaluate_design(small, scenes)
rep = ls.constraint_report(small, draws)
check("small passive design is feasible", rep["feasible"], f"(power {rep['power_p']:.3f} W, intercept {100*rep['validity_p']:.2f}%)")
saved = oc.validity_max_intercept
oc.validity_max_intercept = 1e-6
check("validity limit is a hard constraint (tiny limit -> infeasible)", not ls.constraint_report(small, draws)["feasible"])
oc.validity_max_intercept = saved
saved_b = cm.ris_network_budget_w
cm.ris_network_budget_w = 0.05
rep2 = ls.constraint_report(small, draws)
check("power budget is a hard constraint (0.05 W -> infeasible)", rep2["power_excess"] > 0 and rep2["nominal_power_excess"] > 0 and not rep2["feasible"])
cm.ris_network_budget_w = saved_b
ov = dz.Design(np.array([[100.0, 0.0], [100.5, 0.0]]), np.zeros(2, bool), 1.0, 16, 16)
rep_ov = ls.constraint_report(ov, ev.evaluate_design(ov, scenes))
check("overlapping panels are infeasible; the excess is overlapping pairs per panel (relative)", rep_ov["overlap_excess"] == 0.5 and not rep_ov["feasible"], f"({rep_ov['overlap_excess']})")

# 5. loss: infeasible is always worse than feasible; a better design has lower loss
good = ls.total_loss(small, draws)
oc.validity_max_intercept = 1e-6
bad = ls.total_loss(small, draws)
oc.validity_max_intercept = saved
check("infeasible loss > feasible loss", bad["loss"] > good["loss"] + 100 and good["feasible"] and not bad["feasible"], f"({good['loss']:.2f} vs {bad['loss']:.0f})")
bigger = dz.random_design(10, active_fraction=0.0, seed=2)
lb = ls.total_loss(bigger, ev.evaluate_design(bigger, scenes))
check("more panels reduce the sensing shortfall (soft loss)", lb["terms"]["sensing"] <= good["terms"]["sensing"] + 1e-9, f"({good['terms']['sensing']:.2f} -> {lb['terms']['sensing']:.2f} dB)")

# 6. projection returns a feasible design from an infeasible one
oc.validity_max_intercept = 0.001
dd, rp, nrep = ls.project_to_feasible(dz.random_design(8, active_fraction=1.0, seed=3), scenes, max_iter=40)
check("projection repairs a design that violates the validity limit", rp["feasible"] and nrep > 0, f"({nrep} repairs, {dd.n_panels} panels left, intercept p95 {100*rp.get('validity_p', 0):.3f}%)")
oc.validity_max_intercept = saved

# 7. per-scene role policy
rs = ev.sample_scenes(2, 31, 4, 2)
base = dz.random_design(12, active_fraction=0.4, seed=5)      # some amplified panels, so the gain handling of the score is exercised
sysA = ev.build_system(base, rs[0])
usable = np.array([bool(p.state.a) for p in sysA.panels])
for share in (0.0, 0.25, 0.5, 1.0):
    rad, _, _, _ = ev.assign_roles(sysA, base.copy(radar_share=share))
    check(f"radar_share {share}: radar count = round(share * usable panels), none on blocked panels", rad.sum() == int(round(share * usable.sum())) and not rad[~usable].any(), f"({rad.sum()} of {usable.sum()} usable)")
rad, ph_r, ph_c, score = ev.assign_roles(sysA, base.copy(radar_share=0.5))
check("radar panels have a higher radar/comm advantage than comm panels", score[rad].min() >= score[usable & ~rad].max(), f"(min radar {score[rad].min():.2f} >= max comm {score[usable & ~rad].max():.2f})")
# the score formula itself, recomputed independently (finite scores only)
from src.sys.convention import echo_vec, comm_vec
from src.opt.phase_alignment import coherent_phases_upa
ind = []
for p in sysA.panels:
    amp = p.state.gains.astype(float) if p.state.active else np.ones(256)
    nrm = max(np.linalg.norm(echo_vec(p.channels.G, amp * np.exp(1j * coherent_phases_upa(p, t.pos, "radar")), p.channels.b_by_target[t.target_id])) for t in sysA.targets)
    k = int(np.argmin([np.linalg.norm(u.pos - p.pos) for u in sysA.users]))
    c = np.linalg.norm(comm_vec(p.channels.G, amp * np.exp(1j * coherent_phases_upa(p, sysA.users[k].pos, "comm")), p.channels.f_by_user[k]))
    ind.append(2 * np.log(nrm) - np.log(c) if nrm > 0 and c > 0 else np.nan)
ind = np.array(ind); fin = np.isfinite(score) & np.isfinite(ind)
check("advantage score = 2 log(radar benefit) - log(comm benefit), gains included", fin.sum() > 0 and np.allclose(score[fin], ind[fin]), f"({fin.sum()} finite scores)")
# roles change with the scene (they are not hardcoded per panel)
rad0, _, _, _ = ev.assign_roles(sysA, base.copy(radar_share=0.5))
sysB = ev.build_system(base, rs[1])
rad1, _, _, _ = ev.assign_roles(sysB, base.copy(radar_share=0.5))
check("roles depend on the scene (different scene, different assignment)", not np.array_equal(rad0, rad1),f"(scene 0: {rad0.astype(int)}, scene 1: {rad1.astype(int)})")
# fixed_order reproduces the old static split
fx, _, _, _ = ev.assign_roles(sysA, base.copy(radar_share=0.25, policy="fixed_order"))
check("fixed_order = first round(share*N) panels", fx.sum() == 3 and fx[:3].all())
# result carries the radar count; share 0 and share 1 bracket the radar SNR
r_all = ev.evaluate_scene(base.copy(radar_share=1.0), rs[0])
r_none = ev.evaluate_scene(base.copy(radar_share=0.0), rs[0])
check("active panels exist in the policy test design", base.n_active > 0)
check("evaluator reports n_radar", r_all["n_radar"] >= r_none["n_radar"] == 0, f"({r_all['n_radar']} vs {r_none['n_radar']})")
check("more radar panels give more radar SNR (share 1 vs 0)", np.nanmean(r_all["radar_db"]) > np.nanmean(r_none["radar_db"]), f"({np.nanmean(r_all['radar_db']):.1f} vs {np.nanmean(r_none['radar_db']):.1f} dB)")
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
