"""Beamformer checks (MRT / ZF / RZF / WMMSE). Run: python tests/test_beamforming.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from scipy.optimize import minimize
from src.opt import beamforming as bf

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


rng = np.random.default_rng(0)
cn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
M, K, P, s2 = 16, 4, 1.0, 0.05

# 1. K = 1: optimum is MRT, rate log2(1 + P |h|^2 / sigma^2)
h = cn(M, 1)
V = bf.wmmse(h, P, s2)
check("K=1: WMMSE rate == log2(1 + P|h|^2/sigma^2)", abs(bf.sum_rate(h, V, [s2]) - np.log2(1 + P * np.linalg.norm(h) ** 2 / s2)) < 1e-6)

# 2. power constraint holds and binds; objective is monotone
H = cn(M, K)
V, hist = bf.wmmse(H, P, s2, return_history=True)
check("total power == P", abs(np.linalg.norm(V) ** 2 - P) < 1e-8, f"({np.linalg.norm(V) ** 2:.9f})")
check("weighted sum rate is non-decreasing over iterations", bool(np.all(np.diff(hist) > -1e-9)), f"({len(hist)} iterations, {hist[0]:.3f} -> {hist[-1]:.3f})")

# 3. never worse than the closed-form baselines (i.i.d. and strongly correlated users)
for label, Hc in (("i.i.d. users", cn(64, 8)),
                  ("correlated users (share a dominant direction)", cn(64, 1) @ np.ones((1, 8)) * 3 + cn(64, 8))):
    s2c = 0.1; Pc = 1.0
    r_w = bf.sum_rate(Hc, bf.wmmse(Hc, Pc, s2c), np.full(8, s2c))
    r_z = bf.sum_rate(Hc, bf.rzf(Hc, Pc, np.full(8, s2c)), np.full(8, s2c))
    r_m = bf.sum_rate(Hc, bf.mrt(Hc, Pc), np.full(8, s2c))
    check(f"{label}: WMMSE >= RZF >= MRT", r_w >= r_z - 1e-6 and r_z >= r_m, f"(WMMSE {r_w:.2f}, RZF {r_z:.2f}, MRT {r_m:.2f} bit/s/Hz)")

# 4. orthogonal users: interference-free, equal power split is optimal for equal gains
Ho = np.sqrt(M) * np.linalg.qr(cn(M, K))[0]
Vo = bf.wmmse(Ho, P, s2)
check("orthogonal equal users: equal split", np.allclose(np.linalg.norm(Vo, axis=0) ** 2, P / K, rtol=1e-3))

# 5. a fixed sensing-beam interference covariance reduces the sum rate and WMMSE adapts to it
vs = cn(M, 1); R0 = 0.5 * (vs @ vs.conj().T) / np.linalg.norm(vs) ** 2
r0_no = bf.sum_rate(H, bf.wmmse(H, P, s2), np.full(K, s2), R0)
r0_ad = bf.sum_rate(H, bf.wmmse(H, P, s2, R0=R0), np.full(K, s2), R0)
check("WMMSE aware of R0 >= WMMSE unaware of R0 (evaluated with R0)", r0_ad >= r0_no - 1e-9, f"({r0_ad:.3f} vs {r0_no:.3f})")

# 6. small problem vs brute-force multi-start optimization of the same objective
M2, K2 = 3, 2
H2 = cn(M2, K2)
def neg_rate(x):
    V = (x[:M2 * K2] + 1j * x[M2 * K2:]).reshape(M2, K2)
    V = np.sqrt(P) * V / max(np.linalg.norm(V), 1e-12)
    return -bf.sum_rate(H2, V, np.full(K2, s2))
best = max(-minimize(neg_rate, rng.standard_normal(2 * M2 * K2), method="BFGS").fun for _ in range(30))
r_w = bf.sum_rate(H2, bf.wmmse(H2, P, s2), np.full(K2, s2))
check("small problem: WMMSE within 0.1% of multi-start brute force", r_w >= best * 0.999, f"(WMMSE {r_w:.4f}, brute force {best:.4f})")
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
