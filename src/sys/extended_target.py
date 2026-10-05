"""Extended-target correction (configs/default.yaml `target_model: extended`).

The point-target model puts the whole RCS in one dot at the focus. With many panels the coherent focal spot is centimetres wide,
far smaller than a person or a car, so the target does not return the focus-point echo: its scattering centres sit off the focus,
where the echo vector u_k is only partly correlated with the focus echo u_c. For incoherent scatterers (total RCS fixed) with
transmit beam and receive filter matched to the focus,
    SNR_ext / SNR_point = mean_k rho_k^4,     rho_k = |u_c^H u_k| / |u_c|^2   (0..1)
(one factor rho^2 from the transmit side, one from the receive side). The intercepted-power fraction scales as
    frac_ext = frac_point * mean_k rho_k^2,   frac_point = sigma_eff |u_c|^2     (> 1 means the point model is unphysical).
Evaluated on the deterministic line-of-sight geometry (a ceiling), with the phases designed for the focus.
"""
import numpy as np
from src.channel.geometry3d import cfg, to3, link_3d, bs_array, point_end, ris_end
from src.sys.convention import echo_vec
from src.opt.phase_alignment import coherent_phases_upa


def target_size():
    return np.array(cfg("target_size_m", [0.5, 0.5, 1.7]), dtype=float)


def is_extended():
    return str(cfg("target_model", "point")) == "extended"


def _rho(system, target, size, K, seed):
    sel = [p for p in system.panels if p.state.a]
    if not sel:
        return None, None
    eta = float(cfg("eta_los", 2.0))
    bs_el, bs_e = bs_array()
    phases = [coherent_phases_upa(p, target.pos, "radar") for p in sel]
    amps = [(p.state.gains.astype(float) if (p.state.active and p.state.gains is not None) else 1.0) for p in sel]
    G = [link_3d(p.elements, ris_end(p.normal), bs_el, bs_e, None, eta, None) for p in sel]

    def u_at(q3):
        u = 0
        for p, Gi, ph, am in zip(sel, G, phases, amps):
            b = link_3d(p.elements, ris_end(p.normal), q3[None, :], point_end(), None, eta, None)
            u = u + echo_vec(Gi, am * np.exp(1j * ph), b)
        return u[:, 0]

    c3 = to3(target.pos, float(cfg("h_target", 1.5)))
    uc = u_at(c3)
    pts = c3 + np.random.default_rng(seed).uniform(-0.5, 0.5, (K, 3)) * np.asarray(size)
    rho = np.array([abs(uc.conj() @ u_at(q)) / np.linalg.norm(uc) ** 2 for q in pts])
    return rho, uc


def extended_loss_db(system, target, size=None, K=24, seed=0):
    """10 log10 mean rho^4  (<= 0 dB). NaN if no panel is in line of sight to the BS."""
    rho, _ = _rho(system, target, target_size() if size is None else size, K, seed)
    return float("nan") if rho is None else float(10 * np.log10(np.mean(rho ** 4) + 1e-300))


def validity(system, target, size=None, K=24, seed=0):
    """dict(frac_point, frac_ext, ext_loss_db). frac > 1 -> the model violates energy conservation."""
    rho, uc = _rho(system, target, target_size() if size is None else size, K, seed)
    if rho is None:
        return dict(frac_point=0.0, frac_ext=0.0, ext_loss_db=float("nan"))
    fp = float(target.rcs * np.linalg.norm(uc) ** 2)
    return dict(frac_point=fp, frac_ext=fp * float(np.mean(rho ** 2)), ext_loss_db=float(10 * np.log10(np.mean(rho ** 4) + 1e-300)))
