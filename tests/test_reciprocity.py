"""Audit C1: verification of the reciprocal channel convention. Run: python tests/test_reciprocity.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "out"))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.utils.channel_utils import to_linear, to_linear_dbm, to_db
from src.channel.channel_model import wavelength
from src.channel.geometry3d import bs_array
from src.sys import scenario as sc
from src.sys.convention import reciprocal, echo_vec, comm_vec
from src.opt.phase_alignment import coherent_phases_upa, set_coherent_phases
from src.sys.system import ISACSystem

ok = True
cm = settings.config.channel_model
K0 = 2 * np.pi / wavelength()
P = to_linear_dbm(cm.P_max)


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


check("convention is reciprocal", reciprocal())
orig = sc.get_link_params
_saved_kappa = getattr(cm, 'kappa_los_db', 5.0)
cm.kappa_los_db = 120.0                                              # BS-panel link pure LoS for the phase checks
sc.get_link_params = lambda a, b, rng: (1e12, 2.0, False)           # pure LoS so phases are exact
users, targets = sc.make_users_targets(2, 1, np.random.default_rng(1), direct_mode="none")
pos = [np.array([95.0, 25.0]), np.array([112.0, -18.0])]
system = sc.build_system(pos, users, targets, 4, 4, lambda i: False, P, np.random.default_rng(2), direct_link_scale=0.0, cross_panel_T=True)
sc.get_link_params = orig
bs_el, _ = bs_array(); M = bs_el.shape[0]
for p in system.panels:
    p.state.phases = np.random.default_rng(5).uniform(0, 2 * np.pi, p.elements.shape[0])
system.build_hbar(); system.build_w_mrt(); system.build_T(); system.build_J()

# 1. channel matrices carry the physical propagation phase exp(-j k d)
p0 = system.panels[0]
d = np.linalg.norm(p0.elements[:, None, :] - bs_el[None, :, :], axis=2)
check("G[l,m] phase == -k d(panel element l, BS element m)", np.allclose(np.angle(p0.channels.G * np.exp(1j * K0 * d)), 0, atol=1e-4))

# 2. explicit element-level path sum  BS_m -> (panel i, element l) -> target -> (panel j, element l') -> BS_m'
t = system.targets[0]
T_brute = np.zeros((M, M), dtype=complex)
for pi in system.panels:
    for pj in system.panels:
        for l, (Gi, bi, phi_i) in enumerate(zip(pi.channels.G, pi.channels.b_by_target[0][:, 0], np.exp(1j * pi.state.phases))):
            out_m = Gi * phi_i * bi                                   # (M,) outbound amplitude per transmit element m
            for lp, (Gj, bj, phi_j) in enumerate(zip(pj.channels.G, pj.channels.b_by_target[0][:, 0], np.exp(1j * pj.state.phases))):
                T_brute += np.outer(Gj * phi_j * bj, out_m)            # [m', m]
check("T (all-pairs, reciprocal) == explicit sum over every (i, l, j, l') path", np.allclose(t.T, T_brute, rtol=1e-9, atol=0),
      f"(max rel diff {np.abs(t.T - T_brute).max() / np.abs(T_brute).max():.1e})")
check("echo operator is symmetric, T = T^T (reciprocity of the two-way link)", np.allclose(t.T, t.T.T))

# 3. forward channel from first principles: y_k = sum_m (f^T Phi G)_m x_m
u = system.users[0]
x = np.random.default_rng(3).standard_normal(M) + 1j * np.random.default_rng(4).standard_normal(M)
y_first = sum((pn.channels.f_by_user[0][:, 0] * np.exp(1j * pn.state.phases)) @ pn.channels.G @ x for pn in system.panels)
check("h_bar^H x == sum_i f^T Phi G x", np.allclose(u.h_bar.conj().T @ x, y_first, rtol=1e-9))

# 4. one phase set serves radar and comm (user standing on the target)
sc.get_link_params = lambda a, b, rng: (1e12, 2.0, False)
users2, targets2 = sc.make_users_targets(1, 1, np.random.default_rng(7), direct_mode="none")
users2[0].pos = targets2[0].pos.copy()
pan = sc.make_panel(0, np.array([96.0, 28.0]), targets2, users2, 16, 16, False, np.random.default_rng(8))
sc.get_link_params = orig
G, b, f = pan.channels.G, pan.channels.b_by_target[0], pan.channels.f_by_user[0]
ph_r = coherent_phases_upa(pan, targets2[0].pos, "radar"); ph_c = coherent_phases_upa(pan, users2[0].pos, "comm")
check("radar and comm design phases coincide", np.allclose(np.exp(1j * ph_r), np.exp(1j * ph_c)))
rad = lambda ph: np.linalg.norm(echo_vec(G, np.exp(1j * ph), b)) ** 2
com = lambda ph: np.linalg.norm(comm_vec(G, np.exp(1j * ph), f)) ** 2
check("radar gain with comm phases == radar-optimal", abs(to_db(rad(ph_c) / rad(ph_r))) < 1e-6)
check("comm gain with radar phases == comm-optimal", abs(to_db(com(ph_r) / com(ph_c))) < 1e-6)
rng = np.random.default_rng(9)
better = max(to_db(rad(ph_r + 0.3 * rng.standard_normal(ph_r.size)) / rad(ph_r)) for _ in range(20))
check("design phases are a maximum (random perturbations lose gain)", better < 0, f"(best perturbation {better:.2f} dB)")

# 5. a genuine conflict appears only when user and target are in different places
users3, targets3 = sc.make_users_targets(1, 1, np.random.default_rng(10), direct_mode="none")
users3[0].pos = targets3[0].pos + np.array([3.0, 4.0])
sc.get_link_params = lambda a, b_, rng: (1e12, 2.0, False)
pan3 = sc.make_panel(0, np.array([96.0, 28.0]), targets3, users3, 16, 16, False, np.random.default_rng(8))
sc.get_link_params = orig
ph_r3 = coherent_phases_upa(pan3, targets3[0].pos, "radar"); ph_c3 = coherent_phases_upa(pan3, users3[0].pos, "comm")
pen = to_db(np.linalg.norm(comm_vec(pan3.channels.G, np.exp(1j * ph_r3), pan3.channels.f_by_user[0])) ** 2 /
            np.linalg.norm(comm_vec(pan3.channels.G, np.exp(1j * ph_c3), pan3.channels.f_by_user[0])) ** 2)
check("5 m apart: serving the user with the target's phases costs gain (real conflict)", pen < -3, f"({pen:.1f} dB)")

# 6. noise generator covariance == J (active panels only; amplifier noise raised so it is visible)
saved = cm.active_ris_noise
cm.active_ris_noise = -50
sc.get_link_params = lambda a, b_, rng: (1e12, 2.0, False)
users4, targets4 = sc.make_users_targets(2, 1, np.random.default_rng(11), direct_mode="none")
s4 = sc.build_system(pos, users4, targets4, 4, 4, lambda i: i == 0, P, np.random.default_rng(12), direct_link_scale=0.0, cross_panel_T=True)
sc.get_link_params = orig
s4.build_hbar(); s4.build_w_mrt(); s4.build_T(); s4.build_J()
sv = to_linear(cm.active_ris_noise - 30); sr = to_linear(int(cm.reciever_nosie) - 30)
rng = np.random.default_rng(13); reps = 40000
acc = np.zeros((M, M), dtype=complex)
for _ in range(reps):
    for pn in s4.panels:
        if pn.state.active:
            Lc = pn.state.phases.shape[0]
            pn.state.noise = np.sqrt(sv / 2) * (rng.standard_normal(Lc) + 1j * rng.standard_normal(Lc))
            pn.state.noise_2 = np.sqrt(sv / 2) * (rng.standard_normal(Lc) + 1j * rng.standard_normal(Lc))
    y = s4.build_y_r(np.zeros((M, 1), complex), rng)[0]
    acc += y @ y.conj().T
C = acc / reps
Jm = s4.targets[0].J
rel = np.linalg.norm(C - Jm) / np.linalg.norm(Jm)
check("generator covariance == J within Monte Carlo error", rel < 0.05, f"(relative difference {rel:.3f} over {reps} draws)")
cm.active_ris_noise = saved

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
