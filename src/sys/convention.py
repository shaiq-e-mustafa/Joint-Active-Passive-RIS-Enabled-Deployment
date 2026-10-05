"""Channel-reciprocity convention (audit C1, configs/default.yaml `reciprocity`).

Physical (reciprocal) model, element l of panel i, BS element m:
    forward  BS -> panel -> user k :   c_k = f_k^T Phi G           (row, 1 x M)       y_k = c_k x
    echo     BS -> panel -> target -> panel -> BS :   y = alpha u u^T x,   u = G^T Phi b      (M-vector)
The code convention is y_k = hbar_k^H x, so hbar_k = c_k^H = conj(G^T Phi f_k) = conj(u-form of f_k).
One RIS phase set  phi_l = k (d1_l + d2_l)  maximizes the radar echo and the user's channel alike.

The old ("hermitian") convention used G^H Phi b and G^H Phi^H f, which makes the radar-optimal phases k(d2-d1) and the comm-optimal
phases the negative of that: it invented a radar/comm conflict. It stays selectable (`reciprocity: hermitian`) to reproduce Entries 1-12.
"""
import numpy as np
from src.utils.config import settings


def reciprocal() -> bool:
    return getattr(settings.config.channel_model, "reciprocity", "reciprocal") == "reciprocal"


def echo_vec(G, phi_vec, b):
    """(M,1) echo vector of one panel toward a target: u_i = G^T Phi b  (hermitian: G^H Phi b)."""
    Gm = G.T if reciprocal() else G.conj().T
    return Gm @ (phi_vec[:, None] * b)


def comm_vec(G, phi_vec, f):
    """(M,1) contribution of one panel to hbar_k (y_k = hbar_k^H x):  conj(G^T Phi f)  (hermitian: G^H Phi^H f)."""
    if reciprocal():
        return np.conj(G.T @ (phi_vec[:, None] * f))
    return G.conj().T @ (phi_vec.conj()[:, None] * f)


def direct_comm(hdk):
    """(M,1) direct BS-user contribution to hbar (hdk is the BS-element x user channel)."""
    return np.conj(hdk) if reciprocal() else hdk


def direct_echo(H_row):
    """(M,1) direct BS-target echo vector from the forward row H (1 x M, link target<-BS)."""
    return H_row.T if reciprocal() else H_row.conj().T


def illumination(u):
    """Transmit direction that maximizes |u^T x| (reciprocal) or |u^H x| (hermitian); matched beam = this / norm."""
    return np.conj(u) if reciprocal() else u


def echo_operator(u):
    """T such that the echo is alpha T x:  u u^T (reciprocal) or u u^H (hermitian). u is (M,1)."""
    return u @ (u.T if reciprocal() else u.conj().T)


def design_phases(d_bs, d_aim, k, mode):
    """Per-element phases from exact distances. Reciprocal: k (d1 + d2) for BOTH radar and comm.
    Hermitian (legacy): radar k (d2 - d1), comm the negative."""
    if reciprocal():
        if mode not in ("radar", "comm"):
            raise ValueError(mode)
        return k * (d_bs + d_aim)
    base = k * (d_aim - d_bs)
    if mode == "radar":
        return base
    if mode == "comm":
        return -base
    raise ValueError(mode)
