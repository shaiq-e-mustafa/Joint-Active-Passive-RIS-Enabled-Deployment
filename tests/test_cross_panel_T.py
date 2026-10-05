"""Verification of the all-pairs radar operator (ISACSystem.cross_panel_T=True).
Run: python tests/test_cross_panel_T.py"""
import sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.utils.config import settings
from src.utils.channel_utils import to_linear, to_linear_dbm, to_db
from src.sys.factory import build_system
from src.sys.system import ISACSystem
from src.sys.convention import echo_vec, echo_operator, reciprocal
_c = lambda v: v if reciprocal() else v.conj()

settings.load_config()
L = settings.config.channel_model.L
M = settings.config.channel_model.M
P = to_linear_dbm(settings.config.channel_model.P_max)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def make(n, seed=5, active=True):
    rng = np.random.default_rng(seed)
    return build_system(n, 4, 2, np.array([active] * n), L, M, P, rng, target_p_block=0.0)


def gvecs(sysm, tgt):
    return [echo_vec(p.channels.G, p.state.phi_vec, p.channels.b_by_target[tgt.target_id]).reshape(-1)
            for p in sysm.panels if p.state.a != 0]


s = make(8)
s.build_hbar(); s.build_w_mrt()

# 1. flag=False must reproduce the ORIGINAL formula T_i = G^H Phi b b^H Phi^* ... exactly as before the edit
s.cross_panel_T = False; s.build_T()
for t in s.targets:
    T_orig = sum(echo_operator(echo_vec(p.channels.G, p.state.phi_vec, p.channels.b_by_target[t.target_id]))
                 for p in s.panels if p.state.a != 0)
    check("regression: flag off == original build_T formula", np.allclose(t.T, T_orig, rtol=1e-10, atol=0),
          f"(max rel diff {np.abs(t.T - T_orig).max() / np.abs(T_orig).max():.1e})")
T_diag = [t.T.copy() for t in s.targets]

# 2. brute-force double loop over every (i, j) panel pair == closed form (sum g)(sum g)^H
s.cross_panel_T = True; s.build_T()
for t, Td in zip(s.targets, T_diag):
    gs = gvecs(s, t)
    brute = sum(np.outer(gi, _c(gj)) for gi in gs for gj in gs)
    check("all-pairs double loop == (sum g)(sum g)^H", np.allclose(t.T, brute, rtol=1e-10))
    off = sum(np.outer(gi, _c(gj)) for a, gi in enumerate(gs) for b, gj in enumerate(gs) if a != b)
    check("T_cross - T_diag == sum over i!=j cross terms", np.allclose(t.T - Td, off, rtol=1e-8, atol=1e-30))
    # 3. structure: Hermitian PSD, rank 1; Cauchy-Schwarz bound on coherent gain
    sv = np.linalg.svd(t.T, compute_uv=False)
    sym = np.allclose(t.T, t.T.T) if reciprocal() else np.allclose(t.T, t.T.conj().T)
    check('T_cross rank 1 and ' + ('symmetric (reciprocal)' if reciprocal() else 'Hermitian'), sym and bool((sv[1:] < 1e-9 * sv[0]).all()))
    n = len(gs)
    check("coherent power <= N x incoherent power (Cauchy-Schwarz)",
          np.real(np.trace(t.T)) <= n * np.real(np.trace(Td)) * (1 + 1e-9),
          f"(ratio {np.real(np.trace(t.T)) / np.real(np.trace(Td)):.2f} of max {n})")

# 4. N = 1: nothing to cross -> identical
s1 = make(1); s1.build_hbar(); s1.build_w_mrt()
s1.cross_panel_T = False; s1.build_T(); a = s1.targets[0].T.copy()
s1.cross_panel_T = True; s1.build_T()
check("N=1: all-pairs == diagonal", np.allclose(a, s1.targets[0].T, rtol=1e-10))

# 5. two IDENTICAL panels: T_diag = 2 g g^H, T_cross = 4 g g^H -> radar SNR (quadratic in T) differs by exactly 16/4 = 6.02 dB
s2 = make(1); p0 = s2.panels[0]
s2.panels = [p0, copy.deepcopy(p0)]
s2.build_hbar(); s2.build_w_mrt(); s2.build_J()
res = {}
for flag in (False, True):
    s2.cross_panel_T = flag; s2.build_T(); s2.compute_radar_snr(multi_target_interference=False)
    res[flag] = to_db(s2.snr_per_target[0])
check("two identical panels: coherent gain = 6.02 dB", abs((res[True] - res[False]) - 6.0206) < 1e-6,
      f"(measured {res[True] - res[False]:.4f} dB)")

# 6. cost
s = make(40, seed=9); s.build_hbar(); s.build_w_mrt()
for flag in (False, True):
    s.cross_panel_T = flag
    t0 = time.perf_counter()
    for _ in range(20): s.build_T()
    print(f"      build_T N=40, {len(s.targets)} targets, M={M}: {'all-pairs' if flag else 'diagonal '} "
          f"{(time.perf_counter() - t0) / 20 * 1e3:.2f} ms")

# 7. is leaving J unchanged acceptable? ratio of amplified-noise part of J to thermal part (all-active, worst case)
s.build_J(); sig_r = to_linear(int(settings.config.channel_model.reciever_nosie) - 30)
J = s.targets[0].J
print(f"      J: thermal part = {sig_r:.2e} per diagonal entry; largest eigenvalue of (J - thermal I) = "
      f"{np.linalg.eigvalsh(J - sig_r * np.eye(M)).max():.2e}  (ratio {np.linalg.eigvalsh(J - sig_r * np.eye(M)).max() / sig_r:.2e})")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
