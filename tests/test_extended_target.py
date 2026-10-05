"""Extended-target evaluator checks. Run: python tests/test_extended_target.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "out"))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.channel.geometry3d import cfg, to3
from src.sys import scenario as sc
from src.sys import extended_target as ET
from src.sys.convention import echo_vec, illumination
from src.opt.phase_alignment import coherent_phases_upa
import exp12_lib as X

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


users, tg = sc.make_users_targets(4, 1, np.random.default_rng(3), direct_mode="none")
pos = X.scene_positions("greedy", 30, np.random.default_rng(1), X.planned_greedy(16, 16, 30))
s = X.make_scene(30, 16, 16, pos, users, tg, 0.0, np.random.default_rng(4))
t = tg[0]
check("zero-size target == point target (0 dB)", abs(ET.extended_loss_db(s, t, size=(1e-9, 1e-9, 1e-9))) < 1e-6)
losses = [ET.extended_loss_db(s, t, size=(a, a, a), K=40, seed=1) for a in (0.05, 0.15, 0.5, 1.0)]
check("loss grows monotonically with target size and is <= 0", all(l <= 1e-9 for l in losses) and all(np.diff(losses) < 0.5 * 0 + 1e-9),
      f"({', '.join(f'{l:.1f}' for l in losses)} dB for 5, 15, 50, 100 cm cubes)")
v = ET.validity(s, t, size=(0.5, 0.5, 1.7))
check("intercepted power: extended <= point", v["frac_ext"] <= v["frac_point"] + 1e-12, f"(point {100*v['frac_point']:.2f}%, extended {100*v['frac_ext']:.2f}%)")

# Monte Carlo with explicit random scatterer amplitudes (Swerling-like), whole chain: x -> echo -> matched filter
K = 24; size = np.array([0.5, 0.5, 1.7])
rho, uc = ET._rho(s, t, size, K, 5)
formula = np.mean(rho ** 4)
sel = [p for p in s.panels if p.state.a]
from src.channel.geometry3d import link_3d, bs_array, point_end, ris_end
eta = float(cfg("eta_los", 2.0)); bs_el, bs_e = bs_array()
ph = [coherent_phases_upa(p, t.pos, "radar") for p in sel]
Gd = [link_3d(p.elements, ris_end(p.normal), bs_el, bs_e, None, eta, None) for p in sel]
c3 = to3(t.pos, float(cfg("h_target", 1.5)))
pts = c3 + np.random.default_rng(5).uniform(-0.5, 0.5, (K, 3)) * size
U = np.stack([sum(echo_vec(G, np.exp(1j * f), link_3d(p.elements, ris_end(p.normal), q[None, :], point_end(), None, eta, None)) for p, G, f in zip(sel, Gd, ph))[:, 0] for q in pts])
v_tx = illumination(uc[:, None])[:, 0] / np.linalg.norm(uc)
rng = np.random.default_rng(6)
sig = 0.0; n = 20000
for _ in range(n):
    a = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2 * K)     # E sum|a|^2 = 1
    y = sum(a[k] * U[k] * (U[k] @ v_tx) for k in range(K))                            # echo vector for unit-power matched beam
    sig += abs(uc.conj() @ y) ** 2 / np.linalg.norm(uc) ** 2
mc = sig / n / np.linalg.norm(uc) ** 4
check("closed form mean(rho^4) == Monte Carlo over random scatterer amplitudes", abs(10 * np.log10(mc / formula)) < 0.3,
      f"(MC {10*np.log10(mc):.2f} dB, formula {10*np.log10(formula):.2f} dB)")
# --- loss with the phases the panels actually have (current=True) ---
for pn in s.panels:
    pn.state.phases = coherent_phases_upa(pn, t.pos, "radar")
l_all = ET.extended_loss_db(s, t, seed=2)
l_cur = ET.extended_loss_db(s, t, seed=2, current=True)
check("current=True equals the re-aimed loss when every panel is aimed at the target", abs(l_all - l_cur) < 1e-9, f"({l_all:.3f} vs {l_cur:.3f} dB)")
l_links = ET.extended_loss_db(s, t, seed=2, current=True, links=ET.panel_links(s))
check("precomputed panel links give the identical result", abs(l_links - l_cur) < 1e-12)
for pn in s.panels[::2]:                                   # every other panel aimed at a user instead (a comm-role panel)
    pn.state.phases = coherent_phases_upa(pn, users[0].pos, "comm")
l_mixed = ET.extended_loss_db(s, t, seed=2, current=True)
check("with half the panels aimed elsewhere the loss changes and stays <= 0 dB", abs(l_mixed - l_cur) > 1e-3 and l_mixed <= 1e-9, f"({l_cur:.2f} -> {l_mixed:.2f} dB)")
v_mixed = ET.validity(s, t, seed=2)
check("the intercepted-power bound ignores the current phases (conservative: all aimed at the target)", abs(v_mixed["ext_loss_db"] - l_all) < 1e-9)
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
