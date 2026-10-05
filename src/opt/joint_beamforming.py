"""Joint (shared) comm + sensing transmit design as a semidefinite program (Liu, Huang, Zhang style).

Variables: comm covariances W_k >= 0 (k = 1..K) and a sensing covariance R0 >= 0. Transmit covariance Rx = sum_k W_k + R0.
    maximize   ill^H Rx ill                                           (radar: SNR = sigma_eff (ill^H Rx ill) (u^H J^-1 u) for a fixed target)
    subject to h_k^H W_k h_k  >=  gamma_k ( sum_{j != k} h_k^H W_j h_k + h_k^H R0 h_k + sigma_k^2 )      (user SINR)
               tr(Rx) <= P.
`ill` is the unit transmit direction that maximizes the echo (conj(u) in the reciprocal convention). Channels are scaled by the noise so the problem is
well conditioned. Exact dimension reduction: every constraint only involves h_k and ill, and projecting each W_k, R0 onto S = span{h_1..h_K, ill}
leaves them unchanged and cannot increase power, so we solve in d = dim S <= K+1 dimensions and map back.
"""
import numpy as np
import cvxpy as cp


def _basis(H, ill):
    A = np.hstack([H, ill[:, None]])
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > 1e-9 * s[0]]


def joint_design(H, ill, P, noise, gammas, allow_R0=True, solver="CLARABEL", full_space=False):
    """Returns dict(status, radar (= ill^H Rx ill), W (list of M x M), R0, Rx, ranks) or status 'infeasible'. gammas: linear SINR targets per user."""
    M, K = H.shape
    noise = np.broadcast_to(np.asarray(noise, float), (K,))
    gam = np.broadcast_to(np.asarray(gammas, float), (K,))
    Hs = H / np.sqrt(noise)[None, :]                                  # unit-noise scaling
    nk = np.linalg.norm(Hs, axis=0)
    hn = Hs / nk[None, :]                                             # unit-norm channels; each SINR constraint is divided by its own |h_k|^2,
    eff_noise = 1.0 / nk ** 2                                         # so every coefficient is O(1) and the noise term becomes 1/|h_k|^2
    B = np.eye(M) if full_space else _basis(hn, ill)
    h = B.conj().T @ hn
    v = B.conj().T @ ill
    d = B.shape[1]
    W = [cp.Variable((d, d), hermitian=True) for _ in range(K)]
    R0 = cp.Variable((d, d), hermitian=True)
    cons = [w >> 0 for w in W] + [R0 >> 0]
    if not allow_R0:
        cons.append(R0 == 0)
    Rx = sum(W) + R0
    q = lambda X, vec: cp.real(vec.conj() @ X @ vec)
    for k in range(K):
        intf = sum(q(W[j], h[:, k]) for j in range(K) if j != k) + q(R0, h[:, k]) + float(eff_noise[k])
        cons.append(q(W[k], h[:, k]) >= gam[k] * intf)
    cons.append(cp.real(cp.trace(Rx)) <= P)
    prob = cp.Problem(cp.Maximize(q(Rx, v)), cons)
    try:
        if solver == "CLARABEL":
            prob.solve(solver=solver, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11, max_iter=400)
        else:
            prob.solve(solver=solver)
    except cp.SolverError:
        return dict(status="solver_error")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return dict(status=prob.status)
    Wf = [B @ w.value @ B.conj().T for w in W]
    R0f = B @ R0.value @ B.conj().T
    Rxf = sum(Wf) + R0f
    ranks = [int(np.sum(np.linalg.eigvalsh((w.value + w.value.conj().T) / 2) > 1e-6 * max(1e-30, np.linalg.eigvalsh(w.value).max()))) for w in W]
    r0rank = int(np.sum(np.linalg.eigvalsh((R0.value + R0.value.conj().T) / 2) > 1e-6 * max(1e-30, np.linalg.eigvalsh(R0.value).max())))
    return dict(status=prob.status, radar=float(np.real(ill.conj() @ Rxf @ ill)), W=Wf, R0=R0f, Rx=Rxf, ranks=ranks, r0_rank=r0rank,
                power=float(np.real(np.trace(Rxf))))


def rank1_form(result, H):
    """Exact rank-one equivalent of an optimal solution: v_k v_k^H = W_k h_k h_k^H W_k / (h_k^H W_k h_k), with the leftover W_k - v_k v_k^H moved
    into the sensing covariance. Every user's signal power, every interference term and the total covariance Rx are unchanged
    (h_k^H v_k v_k^H h_k = h_k^H W_k h_k and the leftover is PSD), so SINRs, radar value and power are identical.
    Returns (V (M x K), R0_new)."""
    M, K = H.shape
    V = np.zeros((M, K), dtype=complex)
    R0 = result["R0"].copy()
    for k in range(K):
        Wk, h = result["W"][k], H[:, k]
        a = Wk @ h
        g = np.real(h.conj() @ a)
        V[:, k] = a / np.sqrt(g)
        R0 = R0 + (Wk - np.outer(V[:, k], V[:, k].conj()))
    return V, (R0 + R0.conj().T) / 2


def user_sinr(H, W_list, R0, noise):
    """Per-user SINR (linear) for covariance-form transmit signals."""
    M, K = H.shape
    noise = np.broadcast_to(np.asarray(noise, float), (K,))
    out = []
    for k in range(K):
        h = H[:, k]
        sig = np.real(h.conj() @ W_list[k] @ h)
        intf = sum(np.real(h.conj() @ W_list[j] @ h) for j in range(K) if j != k) + np.real(h.conj() @ R0 @ h)
        out.append(sig / (intf + noise[k]))
    return np.array(out)
