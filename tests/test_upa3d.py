"""Entry 12 verification of the 3-D UPA / exact-wave model. Run: python tests/test_upa3d.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.utils.channel_utils import to_linear, to_linear_dbm
from src.channel.channel_model import wavelength
from src.channel.geometry3d import (cfg, to3, upa_elements, link_3d, bs_array, point_end, ris_end, spacing)
from src.sys import scenario as sc
from src.opt.phase_alignment import set_coherent_phases
from src.sys.convention import echo_vec, echo_operator
from src.sys.system import ISACSystem

ok = True
_cm = settings.config.channel_model
_saved_cell = (_cm.ris_cell_gain, _cm.ris_cell_q)
_cm.ris_cell_gain, _cm.ris_cell_q = 8.0, 3.0      # Tang's measured cell, needed to reproduce his formulas
lam = wavelength(); k = 2 * np.pi / lam
rng = np.random.default_rng(0)


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 1. one forward pass through a RIS reproduces Tang et al.'s far-field beamforming formula
#    Pr/Pt = G dx dy lambda^2 F(t) F(r) (MN)^2 / (64 pi^3 d1^2 d2^2),  G = 8, F = cos^3, Gt = Gr = A = 1
for (nx, ny, d1, d2, th_t, th_r) in [(4, 4, 50., 60., 0.0, 0.0), (6, 4, 80., 40., 0.5, 0.3), (8, 8, 120., 100., 0.8, -0.4)]:
    n = np.array([1.0, 0.0])
    z = float(cfg("h_panel", 5.0))
    el = upa_elements(np.array([0.0, 0.0]), z, n, nx, ny)
    tx = to3(d1 * np.array([np.cos(th_t), np.sin(th_t)]), z)[None, :]
    rx = to3(d2 * np.array([np.cos(th_r), -np.sin(th_r)]), z)[None, :]
    end = ris_end(n)
    h1 = link_3d(el, end, tx, point_end(), 1e12, 2.0, rng)[:, 0]
    h2 = link_3d(el, end, rx, point_end(), 1e12, 2.0, rng)[:, 0]
    d_tot = np.linalg.norm(el - tx, axis=1) + np.linalg.norm(el - rx, axis=1)
    S = np.sum(h1 * h2 * np.exp(1j * k * d_tot))                 # phases chosen to align every element
    ours = abs(S) ** 2
    dx = dy = spacing()
    tang = 8.0 * dx * dy * lam ** 2 * np.cos(th_t) ** 3 * np.cos(th_r) ** 3 * (nx * ny) ** 2 / (64 * np.pi ** 3 * d1 ** 2 * d2 ** 2)
    check(f"Tang far-field formula, {nx}x{ny}, d1={d1:.0f} d2={d2:.0f}", abs(ours / tang - 1) < 5e-3,
          f"(ours/Tang = {ours / tang:.4f})")

# 2. direct LoS monostatic echo reproduces the radar equation  SNR = P G^2 lambda^2 sigma / ((4 pi)^3 R^4 N), G = M*pi
bs_el, bs_e = bs_array()
M = bs_el.shape[0]
zbs = float(bs_el[:, 2].mean())
R = 100.0
tgt = to3(np.array([R, 0.0]), zbs)[None, :]
H = link_3d(tgt, point_end(), bs_el, bs_e, 1e12, 2.0, rng)       # (1, M)
g = H.conj().T[:, 0]
P = to_linear_dbm(settings.config.channel_model.P_max)
N0 = to_linear(int(settings.config.channel_model.reciever_nosie) - 30)
sigma = 1.0
snr_model = sc.effective_rcs(sigma) * P * np.linalg.norm(g) ** 2 * np.linalg.norm(g) ** 2 / N0
G_arr = M * 4 * np.pi * spacing() ** 2 / lam ** 2
snr_radar_eq = P * G_arr ** 2 * lam ** 2 * sigma / ((4 * np.pi) ** 3 * R ** 4 * N0)
check("direct LoS echo == radar equation (Skolnik)", abs(snr_model / snr_radar_eq - 1) < 1e-2,
      f"(model/radar-eq = {snr_model / snr_radar_eq:.4f}; {10 * np.log10(snr_model):.2f} dB vs {10 * np.log10(snr_radar_eq):.2f} dB)")

# 3. exact phase design gives ~perfect coherence for UPAs (links forced LoS: a Rayleigh link cannot be phase-aligned)
_orig_glp = sc.get_link_params
sc.get_link_params = lambda a, b, rng: (1e12, 2.5, False)
bs_c = bs_el.mean(axis=0)
for nx, ny in [(8, 8), (16, 16), (32, 32)]:
    users, targets = sc.make_users_targets(2, 1, np.random.default_rng(1), direct_mode="none")
    pan = sc.make_panel(0, np.array([95.0, 30.0]), targets, users, nx, ny, False, np.random.default_rng(2))
    Gm, b = pan.channels.G, pan.channels.b_by_target[0][:, 0]
    sysm = ISACSystem(panels=[pan], users=users, targets=targets, p_total_linear=P, rng=rng)
    set_coherent_phases(sysm, targets[0].pos, "radar")
    gv = np.abs(echo_vec(Gm, pan.state.phi_vec, b[:, None])[:, 0])
    ideal = np.abs(Gm).T @ np.abs(b)
    eff = float(np.median(gv / ideal))
    check(f"exact phases, UPA {nx}x{ny}: coherence efficiency", eff > 0.9, f"(median {eff:.3f} of perfect; {nx * ny} elements)")

# 4. flat RIS cannot serve a BS and a zone on opposite sides (angle 180 deg) -> ~no contribution
users, targets = sc.make_users_targets(2, 1, np.random.default_rng(3), direct_mode="none")
amp = {}
for label, pos in [("on BS-zone line (50,0)", [50.0, 0.0]), ("beside the zone (100,25)", [100.0, 25.0])]:
    pan = sc.make_panel(0, np.array(pos), targets, users, 8, 8, False, np.random.default_rng(4))
    sysm = ISACSystem(panels=[pan], users=users, targets=targets, p_total_linear=P, rng=rng)
    set_coherent_phases(sysm, targets[0].pos, "radar")
    amp[label] = np.linalg.norm(echo_vec(pan.channels.G, pan.state.phi_vec, pan.channels.b_by_target[0]))
a_line, a_side = amp.values()
check("panel on the BS-zone line contributes far less than a side panel", a_line < 0.05 * a_side,
      f"(|g| line {a_line:.2e} vs side {a_side:.2e})")

# 5. amplitude grows with element count (rows + columns add coherently) while far-field-ish
res = {}
for nx, ny in [(8, 8), (16, 16), (32, 32)]:
    pan = sc.make_panel(0, np.array([95.0, 30.0]), targets, users, nx, ny, False, np.random.default_rng(2))
    sysm = ISACSystem(panels=[pan], users=users, targets=targets, p_total_linear=P, rng=rng)
    set_coherent_phases(sysm, targets[0].pos, "radar")
    res[nx * ny] = np.linalg.norm(echo_vec(pan.channels.G, pan.state.phi_vec, pan.channels.b_by_target[0]))
els = sorted(res)
for a, b in zip(els[:-1], els[1:]):
    ratio = res[b] / res[a]
    check(f"|g| scales with element count {a}->{b}", 0.7 * (b / a) < ratio < 1.05 * (b / a), f"(ratio {ratio:.2f}, ideal {b / a:.0f})")

sc.get_link_params = _orig_glp

# 6. direct path enters T: diagonal mode adds g_d g_d^H, all-pairs mode adds to the summed vector
users, targets = sc.make_users_targets(2, 1, np.random.default_rng(5), direct_mode="los")
sysm = sc.build_system([np.array([95.0, 30.0]), np.array([110.0, -25.0])], users, targets, 8, 8,
                       lambda i: False, P, np.random.default_rng(6))
sysm.build_hbar(); sysm.build_w_mrt(); sysm.build_J()
gd = targets[0].direct
gs = [echo_vec(p.channels.G, p.state.phi_vec, p.channels.b_by_target[0]) for p in sysm.panels]
sysm.cross_panel_T = False; sysm.build_T()
check("diagonal T includes direct path", np.allclose(targets[0].T, echo_operator(gd) + sum(echo_operator(g) for g in gs), rtol=1e-9))
sysm.cross_panel_T = True; sysm.build_T()
gt = gd + sum(gs)
check("all-pairs T includes direct path", np.allclose(targets[0].T, echo_operator(gt), rtol=1e-9))

# 7. NEAR-FIELD end-to-end: one BS element -> one 16x16 RIS (~15 m from the target) -> isotropic target, versus
#    Tang's near-field expression  Pr/Pt = Gt G dx dy lambda^2 |sum_n sqrt(Gt_n F1_n F2_n)/(r1_n r2_n)|^2 / (64 pi^3)
from src.channel.geometry3d import panel_normal
cm = settings.config.channel_model
_saved = (cm.M, getattr(cm, "bs_cols", None))
cm.M, cm.bs_cols = 1, 1
bs_el1, bs_e1 = bs_array()
ppos = np.array([100.0, 25.0]); tpos = np.array([105.0, 12.0])
nrm = panel_normal(ppos, np.array([0.0, 0.0]), sc.ZONE_CENTER)
el = upa_elements(ppos, float(cfg("h_panel", 5.0)), nrm, 16, 16)
end = ris_end(nrm)
t3 = to3(tpos, float(cfg("h_target", 1.5)))[None, :]
h1 = link_3d(el, end, bs_el1, bs_e1, 1e12, 2.0, rng)[:, 0]
h2 = link_3d(el, end, t3, point_end(), 1e12, 2.0, rng)[:, 0]
r1 = np.linalg.norm(el - bs_el1[0], axis=1); r2 = np.linalg.norm(el - t3[0], axis=1)
S = np.sum(h1 * h2 * np.exp(1j * k * (r1 + r2)))
ours = abs(S) ** 2
n3 = end["normal"]
F1 = np.clip(((bs_el1[0] - el) @ n3) / r1, 0, None) ** 3
F2 = np.clip(((t3[0] - el) @ n3) / r2, 0, None) ** 3
Gt_n = np.pi * np.clip(((el - bs_el1[0]) @ bs_e1["normal"]) / r1, 0, None)      # BS element: gain pi, F = cos
tang_near = 8.0 * spacing() ** 2 * lam ** 2 * abs(np.sum(np.sqrt(Gt_n * F1 * F2) / (r1 * r2))) ** 2 / (64 * np.pi ** 3)
check("near-field (Tang) 16x16 RIS ~15 m from the target", abs(ours / tang_near - 1) < 5e-3,
      f"(ours/Tang-near = {ours / tang_near:.4f}; one-way gain {10 * np.log10(ours):.1f} dB)")
cm.M, cm.bs_cols = _saved

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
