"""Active-gain budget checks (Long et al., arXiv:2103.00709). Run: python tests/test_active_gains.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "out"))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.utils.channel_utils import to_linear, to_linear_dbm
from src.sys import scenario as sc
from src.opt import active_gains as ag
import exp12_lib as X

cm = settings.config.channel_model
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


q = ag.params()
# 1. feasibility arithmetic (Eq. 17): 16x16 active needs 256*(0.1+0.316) mW = 106 mW, leaving 0.8*(1 - 0.106) W
check("circuit power of a 16x16 active panel", abs(ag.panel_power_w(256, True) - 256 * (1e-4 + 10 ** (-0.5) * 1e-3)) < 1e-9, f"({1e3*ag.panel_power_w(256, True):.1f} mW)")
check("P_out = upsilon (P_RIS - L (P_c + P_DC))", abs(ag.p_out_w(256) - 0.8 * (1.0 - ag.panel_power_w(256, True))) < 1e-12)
check("64x64 active panel is infeasible at 30 dBm (needs 1.7 W)", ag.p_out_w(4096) < 0, f"(P_out = {1e3*ag.p_out_w(4096):.0f} mW)")

users, tg = sc.make_users_targets(4, 2, np.random.default_rng(3), direct_mode="none")
pos = X.scene_positions("greedy", 12, np.random.default_rng(1), X.planned_greedy(16, 16, 12))
s = X.make_scene(12, 16, 16, pos, users, tg, 1.0, np.random.default_rng(4))
rep = ag.assign_budget_gains(s)
check("gains never exceed the element cap", max(rep["gains"]) <= q["a_max"] + 1e-12, f"(max {max(rep['gains']):.3f}, cap {q['a_max']:.3f})")
check("with a 30 dBm budget at 100 m the cap binds, not the power", all(abs(g - q["a_max"]) < 1e-9 for g in rep["gains"]))

# 2. force the budget to bind: choose P_RIS so that a^2 = P_out / P_in = 4 for the first active panel (a = 2 < cap), then check the closed form
beams = ag.target_beams(s)
sv = to_linear(float(cm.active_ris_noise) - 30)
first = next(p for p in s.panels if p.state.active)
pin = ag.incident_power_w(s, first, beams) + 256 * sv
cm.ris_panel_power_dbm = 10 * np.log10((ag.panel_power_w(256, True) + 4 * pin / 0.8) * 1e3)
rep2 = ag.assign_budget_gains(s)
out_power = first.state.gains[0] ** 2 * pin
check("budget-limited: amplifier output power == P_out", abs(out_power / ag.p_out_w(256) - 1) < 1e-9, f"(a = {first.state.gains[0]:.4f}, expected 2, cap {q['a_max']:.3f})")
check("budget-limited gain equals sqrt(P_out / P_in) (Long et al.)", abs(first.state.gains[0] - 2.0) < 1e-6)
check("budget-limited gains are below the cap", all(g < q["a_max"] for g in rep2["gains"]))
cm.ris_panel_power_dbm = 30.0

# 3. infeasible panels fall back to passive
s3 = X.make_scene(4, 64, 64, X.scene_positions("random", 4, np.random.default_rng(2)), users, tg, 1.0, np.random.default_rng(5))
rep3 = ag.assign_budget_gains(s3)
check("64x64 active panels are passive (budget cannot power them)", not any(p.state.active for p in s3.panels) and all(p.state.gains is None for p in s3.panels))
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
