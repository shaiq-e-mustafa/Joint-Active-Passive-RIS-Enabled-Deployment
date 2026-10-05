"""Downlink beamformers for the RIS-composite channels H = [hbar_1 ... hbar_K] (M x K), received y_k = hbar_k^H x + noise.

mrt, zf, rzf : closed-form baselines.
wmmse        : weighted-sum-rate maximization (Shi, Razaviyayn, Luo, He, IEEE TSP 2011), MISO downlink, total power constraint
               sum_k ||v_k||^2 <= P. A fixed interference covariance R0 (e.g. a dedicated sensing beam) enters every user's
               denominator. Iterations: u_k (MMSE receiver), w_k = 1/e_k (MSE weight), v_k = alpha_k w_k u_k (A + mu I)^-1 hbar_k with
               A = sum_j alpha_j w_j |u_j|^2 hbar_j hbar_j^H and mu >= 0 by bisection so that the power constraint holds.
sinr / rates : evaluation under any precoder.
"""
import numpy as np


def _normalize(V, P):
    return np.sqrt(P) * V / np.linalg.norm(V)


def mrt(H, P):
    K = H.shape[1]
    return np.sqrt(P / K) * H / np.linalg.norm(H, axis=0)


def zf(H, P):
    V = H @ np.linalg.pinv(H.conj().T @ H)               # pinv = inv for full-rank H; finite (min-norm) when two users have the same channel
    return np.sqrt(P / H.shape[1]) * V / np.linalg.norm(V, axis=0)


def rzf(H, P, noise):
    K = H.shape[1]
    lam = K * float(np.mean(noise)) / P
    V = H @ np.linalg.inv(H.conj().T @ H + lam * np.eye(K))
    return np.sqrt(P / K) * V / np.linalg.norm(V, axis=0)


def sinr(H, V, noise, R0=None):
    """Per-user SINR (linear). R0: fixed interference covariance (M x M) from other transmit components."""
    G = np.abs(H.conj().T @ V) ** 2
    sig = np.diag(G)
    intf = G.sum(axis=1) - sig
    extra = np.zeros(H.shape[1]) if R0 is None else np.real(np.einsum("mk,mn,nk->k", H.conj(), R0, H))
    return sig / (intf + extra + np.asarray(noise))


def sum_rate(H, V, noise, R0=None, weights=None):
    r = np.log2(1 + sinr(H, V, noise, R0))
    return float(np.sum(r if weights is None else np.asarray(weights) * r))


def wmmse(H, P, noise, R0=None, weights=None, iters=300, tol=1e-9, V0=None, return_history=False):
    M, K = H.shape
    noise = np.broadcast_to(np.asarray(noise, float), (K,)).copy()
    alpha = np.ones(K) if weights is None else np.asarray(weights, float)
    R0h = np.zeros(K) if R0 is None else np.real(np.einsum("mk,mn,nk->k", H.conj(), R0, H))
    V = (rzf(H, P, noise) if V0 is None else V0).astype(complex)
    hist, prev = [], -np.inf
    for _ in range(iters):
        HV = H.conj().T @ V                                           # (K, K): [k, j] = h_k^H v_j
        a = np.sum(np.abs(HV) ** 2, axis=1) + R0h + noise
        u = np.diag(HV) / a
        e = 1 - np.abs(np.diag(HV)) ** 2 / a
        w = 1.0 / e
        A = (H * (alpha * w * np.abs(u) ** 2)[None, :]) @ H.conj().T   # sum_j alpha_j w_j |u_j|^2 h_j h_j^H
        lam, Uv = np.linalg.eigh(A)
        keep = lam > 1e-12 * max(float(lam.max()), 1e-300)             # A = sum_j c_j h_j h_j^H has rank <= K < M: its null space carries no part of H
        lam, Uv = lam[keep], Uv[:, keep]                               # (dropping it is exact and avoids the 0/0 of (lam + mu)^-2 at mu = 0)
        Hproj = Uv.conj().T @ H                                        # (M, K)
        coef = alpha * w * u                                           # v_k = coef_k (A + mu I)^-1 h_k

        def power(mu):
            return float(np.sum(np.abs(coef) ** 2 * np.sum(np.abs(Hproj) ** 2 / (lam[:, None] + mu) ** 2, axis=0)))
        mu = 0.0
        if power(0.0) > P:
            lo, hi = 0.0, 1.0
            while power(hi) > P:
                hi *= 2
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if power(mid) > P else (lo, mid)
            mu = hi
        V = Uv @ ((Hproj / (lam[:, None] + mu)) * coef[None, :])
        obj = sum_rate(H, V, noise, R0, alpha)
        hist.append(obj)
        if abs(obj - prev) < tol * max(1.0, abs(obj)):
            break
        prev = obj
    return (V, hist) if return_history else V
