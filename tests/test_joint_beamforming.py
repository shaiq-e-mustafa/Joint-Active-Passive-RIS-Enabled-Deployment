"""Joint comm+sensing SDP checks. Run: python tests/test_joint_beamforming.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.opt import joint_beamforming as jb
from src.opt import beamforming as bf

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


rng = np.random.default_rng(0)
cn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
M, K, P = 8, 3, 1.0
H = 20 * cn(M, K)
ill = cn(M); ill /= np.linalg.norm(ill)
noise = np.full(K, 1.0)
gam = 10 ** (np.array([6.0, 6.0, 6.0]) / 10)

r = jb.joint_design(H, ill, P, noise, gam)
check("solved", r["status"] in ("optimal", "optimal_inaccurate"), f"({r['status']})")
check("power constraint holds", r["power"] <= P * (1 + 1e-6), f"({r['power']:.6f})")
sinr = jb.user_sinr(H, r["W"], r["R0"], noise)
check("every user meets its SINR target", bool(np.all(sinr >= gam * (1 - 1e-4))), f"(min {10*np.log10(sinr.min()):.2f} dB vs 6.00)")

# exactness of the dimension reduction (reduced d = K+1 = 4 against the full 8-dim problem)
rf = jb.joint_design(H, ill, P, noise, gam, full_space=True)
check("reduced solve == full-space solve", abs(r["radar"] - rf["radar"]) < 1e-5 * max(1, rf["radar"]), f"({r['radar']:.6f} vs {rf['radar']:.6f})")

# no feasible hand-built design beats the optimum: RZF comm beams + all remaining power on a sensing beam in the null space of the users
Pc = P * 0.5
V = bf.rzf(H, Pc, noise)
Qn = np.eye(M) - H @ np.linalg.solve(H.conj().T @ H, H.conj().T)
vs = Qn @ ill; vs = vs / np.linalg.norm(vs)
Rx_hand = V @ V.conj().T + (P - Pc) * np.outer(vs, vs.conj())
sinr_hand = bf.sinr(H, V, noise, (P - Pc) * np.outer(vs, vs.conj()))
feasible = bool(np.all(sinr_hand >= gam))
val_hand = float(np.real(ill.conj() @ Rx_hand @ ill))
check("hand-built feasible design <= optimum", (not feasible) or val_hand <= r["radar"] + 1e-6, f"(hand {val_hand:.4f}, optimum {r['radar']:.4f}, feasible={feasible})")

# monotone in the SINR requirement
vals = []
for g_db in (0.0, 6.0, 12.0, 18.0):
    rr = jb.joint_design(H, ill, P, noise, 10 ** (g_db / 10))
    vals.append(rr["radar"] if rr["status"].startswith("optimal") else 0.0)
check("radar value is non-increasing as the SINR target rises", bool(np.all(np.diff(vals) <= 1e-6)), f"({', '.join(f'{v:.3f}' for v in vals)})")

# rank-one form: real beams v_k plus a (possibly higher-rank) sensing covariance reproduce the SDP solution exactly
okx = True; detail = []
for sd in range(5):
    Hs = 20 * cn(M, K); il = cn(M); il /= np.linalg.norm(il)
    gm = 10 ** (8 / 10)
    rr = jb.joint_design(Hs, il, P, noise, gm)
    if not rr["status"].startswith("optimal"):
        continue
    Vs, R0n = jb.rank1_form(rr, Hs)
    Rx_new = Vs @ Vs.conj().T + R0n
    s_new = bf.sinr(Hs, Vs, noise, R0n)
    okx &= bool(np.allclose(Rx_new, rr["Rx"], atol=1e-8) and np.all(s_new >= gm * (1 - 1e-4)) and np.min(np.linalg.eigvalsh(R0n)) > -1e-5)
    detail.append(int(np.sum(np.linalg.eigvalsh(R0n) > 1e-6)))
check("rank-one form: same Rx, SINRs met, sensing covariance PSD", okx, f"(sensing-covariance ranks: {detail})")

# known answer: one user, sensing direction parallel to the user's channel -> all power on the user beam, radar = P
h1 = cn(M, 1) * 20
rk = jb.joint_design(h1, h1[:, 0] / np.linalg.norm(h1), P, [1.0], 10 ** (3 / 10))
check("K=1, sensing direction == user direction: radar value = P", abs(rk["radar"] - P) < 1e-5, f"({rk['radar']:.6f})")
# --- rank1_form with a user that gets no signal power (W_k = 0) must not divide by zero ---
_H = np.eye(4, 2, dtype=complex)
_res = dict(W=[np.zeros((4, 4), complex), np.diag([0, 1.0, 0, 0]).astype(complex)], R0=np.zeros((4, 4), complex))
_V, _R0 = jb.rank1_form(_res, _H)
check("rank1_form: a zero user covariance gives no stream and no NaN", np.all(np.isfinite(_V)) and np.all(np.isfinite(_R0)) and np.allclose(_V[:, 0], 0))

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
