"""Network power budget checks. Run: python tests/test_power_budget.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "out"))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.sys import scenario as sc
from src.opt import power_budget as pb
from src.opt import active_gains as ag
from src.opt import placement3d as pl
import exp12_lib as X

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


nx = ny = 16; L = 256
cands = pl.annulus_candidates(60)
design = sc.disk_sample(sc.ZONE_CENTER, sc.ZONE_RADIUS, 6, np.random.default_rng(0))
p_pass = ag.panel_power_w(L, False)

# 1. planner: never exceeds the budget; counts follow the budget; tiny budget buys nothing; huge budget upgrades every panel
res = {}
for B in (0.01, 0.3, 1.0, 2.0, 50.0):
    chosen, used = pb.plan_under_budget(cands, design, nx, ny, B, max_panels=12, n_trials=8)
    res[B] = (chosen, used)
    check(f"budget {B} W: nominal power used <= budget", used <= B + 1e-12, f"({used:.3f} W, {len(chosen)} panels, {sum(a for _, a in chosen)} active)")
check("budget below one passive panel buys nothing", len(res[0.01][0]) == 0)
check("huge budget: every panel is active", all(a for _, a in res[50.0][0]) and len(res[50.0][0]) == 12)
n_prev = -1
mono = True
for B in (0.3, 1.0, 2.0, 50.0):
    n = len(res[B][0]) + sum(a for _, a in res[B][0])
    mono &= n >= n_prev
    n_prev = n
check("more budget buys at least as many panels + upgrades", mono)

# 2. enforce_network_budget on a built system
users, tg = sc.make_users_targets(4, 2, np.random.default_rng(3), direct_mode="none")
pos = X.scene_positions("greedy", 20, np.random.default_rng(1), X.planned_greedy(16, 16, 20))
s = X.make_scene(20, 16, 16, pos, users, tg, 1.0, np.random.default_rng(4))
p0 = pb.network_power_w(s)
check("20 active 16x16 panels draw more than 2 W", p0 > 2.0, f"({p0:.2f} W)")
check("unlimited budget leaves the system unchanged", pb.enforce_network_budget(s, budget=1e9)["downgraded"] == 0 and abs(pb.network_power_w(s) - p0) < 1e-12)
ben_before = {id(p): pb._benefit(s, p) for p in s.panels}
rep = pb.enforce_network_budget(s, budget=1.5)
check("enforce: drawn power <= budget afterwards", rep["after_w"] <= 1.5 + 1e-9 and abs(pb.network_power_w(s) - rep["after_w"]) < 1e-12,
      f"({rep['before_w']:.2f} -> {rep['after_w']:.2f} W, {rep['downgraded']} downgraded, {rep['switched_off']} off)")
kept = [ben_before[id(p)] for p in s.panels if p.state.a and p.state.active]          # panels in line of sight only
dropped = [ben_before[id(p)] for p in s.panels if p.state.a and not p.state.active]
check("downgraded panels are the least beneficial ones", (min(kept) >= max(dropped) - 1e-18) if kept and dropped else True)
rep2 = pb.enforce_network_budget(s, budget=0.3)
check("tiny budget switches panels off, still within budget", rep2["after_w"] <= 0.3 + 1e-9 and rep2["switched_off"] > 0, f"({rep2['after_w']:.3f} W, {rep2['switched_off']} off)")
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
