"""'100 m surveillance zone' scenario with 3-D UPA panels (Entry 12).

BS (8x8 UPA, 10 m, facing +x) at the origin. A hidden zone (disk) centred ~100 m away holds the targets and
users at 1.5 m height. RIS panels (flat UPAs on facades, 5 m height) are placed around the zone; each
panel's normal bisects its directions to the BS and the zone centre.
"""
import numpy as np
from src.utils.config import settings
from src.utils.channel_utils import to_db
from src.channel.risConfig import get_link_params
from src.channel.channel_model import wavelength
from src.channel.geometry3d import (cfg, to3, upa_elements, panel_normal, link_3d, bs_array,
                                    point_end, ris_end)
from src.sys.system import RISPanel, UserLink, TargetLink, ISACSystem
from src.sys.channels import PanelChannels, UserChannels
from src.sys.factory import make_panel_state
from src.sys.convention import echo_vec, direct_echo
from src.sim.deployment import BS_POS

ZONE_CENTER = np.array([100.0, 0.0])
ZONE_RADIUS = 15.0


def effective_rcs(rcs_m2):
    """sigma [m^2] -> dimensionless scattering factor 4 pi sigma / lambda^2 (isotropic-equivalent), so that
    SNR = P (G_t G_r)(lambda/4 pi R)^4 * 4 pi sigma / lambda^2 / noise  equals the radar equation
    P G^2 lambda^2 sigma / ((4 pi)^3 R^4 N) (Skolnik). Disabled with radar_scatter_gain: false."""
    if not bool(cfg("radar_scatter_gain", True)):
        return rcs_m2
    return rcs_m2 * 4 * np.pi / wavelength() ** 2


def disk_sample(center, radius, n, rng):
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return [np.asarray(center, float) + np.array([ri * np.cos(ai), ri * np.sin(ai)]) for ri, ai in zip(r, a)]


def direct_vector(tpos2, mode, rng):
    """BS<->target direct path as an (M,1) echo vector, in the convention of src/sys/convention.py (direct_echo: the forward
    row transposed when reciprocal, conjugate-transposed in the legacy hermitian convention). mode: 'los' | 'nlos' | 'none'."""
    if mode == "none":
        return None
    kappa, eta = (10 ** (float(cfg("kappa_los_db", 5.0)) / 10), float(cfg("eta_los", 2.0))) if mode == "los" else (0.0, float(cfg("eta_nlos", 3.5)))
    bs_el, bs_e = bs_array()
    tgt = to3(tpos2, float(cfg("h_target", 1.5)))[None, :]
    H = link_3d(tgt, point_end(), bs_el, bs_e, kappa, eta, rng)          # (1, M)
    return direct_echo(H)


def make_users_targets(k_users, q_targets, rng, direct_mode="nlos", zone_center=None, zone_radius=None,
                       rcs_m2=1.0):
    zc = ZONE_CENTER if zone_center is None else np.asarray(zone_center, float)
    zr = ZONE_RADIUS if zone_radius is None else zone_radius
    bs_el, bs_e = bs_array()
    user_pos = disk_sample(zc, zr, k_users, rng)
    target_pos = disk_sample(zc, zr, q_targets, rng)
    users = []
    for k, u in enumerate(user_pos):
        eta = get_link_params(BS_POS, u, rng)[1]
        hdk = link_3d(bs_el, bs_e, to3(u, float(cfg("h_user", 1.5)))[None, :], point_end(), 0.0, eta, rng)
        users.append(UserLink(user_id=k, channels=UserChannels(hdk=hdk), pos=u))
    targets = [TargetLink(target_id=q, rcs=effective_rcs(rcs_m2), pos=t,
                          direct=direct_vector(t, direct_mode, rng))
               for q, t in enumerate(target_pos)]
    return users, targets


def make_panel(panel_id, ppos, targets, users, nx, ny, active, rng, a=1, zone_center=None):
    zc = ZONE_CENTER if zone_center is None else np.asarray(zone_center, float)
    ppos = np.asarray(ppos, float)
    normal = panel_normal(ppos, BS_POS, zc)
    el = upa_elements(ppos, float(cfg("h_panel", 5.0)), normal, nx, ny)
    end = ris_end(normal)
    bs_el, bs_e = bs_array()
    # BS<->panel: RIS sites are chosen with line of sight to the BS, so this link is LoS-Rician and blockage is
    # carried ONLY by the panel selection flag a_i (Line-Boolean, calibrated in the experiments). The logistic
    # model in get_link_params would block ~half of all 100 m links and double-count blockage.
    kappa_g, eta_g = 10 ** (float(cfg("kappa_los_db", 5.0)) / 10), float(cfg("eta_los", 2.0))
    G = link_3d(el, end, bs_el, bs_e, kappa_g, eta_g, rng)
    b = {}
    for t in targets:
        kb, eb, _ = get_link_params(ppos, t.pos, rng)
        b[t.target_id] = link_3d(el, end, to3(t.pos, float(cfg("h_target", 1.5)))[None, :], point_end(), kb, eb, rng)
    f = {}
    for u in users:
        kf, ef, _ = get_link_params(ppos, u.pos, rng)
        f[u.user_id] = link_3d(el, end, to3(u.pos, float(cfg("h_user", 1.5)))[None, :], point_end(), kf, ef, rng)
    state = make_panel_state(active, a, nx * ny, rng)
    return RISPanel(panel_id=panel_id, channels=PanelChannels(G=G, b_by_target=b, f_by_user=f),
                    state=state, pos=ppos, elements=el, normal=normal)


def build_system(panel_positions, users, targets, nx, ny, active_fn, p_total_linear, rng,
                 direct_link_scale=0.0, cross_panel_T=True):
    panels = [make_panel(i, p, targets, users, nx, ny, active_fn(i), rng) for i, p in enumerate(panel_positions)]
    return ISACSystem(panels=panels, users=users, targets=targets, p_total_linear=p_total_linear, rng=rng,
                      direct_link_scale=direct_link_scale, cross_panel_T=cross_panel_T)


def radar_snr_matched(system, target_idx=0):
    """SNR with the best single transmit beam for this target: R_x = P g g^H / |g|^2, g = total echo vector
    (direct + all panels). T = g g^H so tr(T Rx T^H J^-1) = P |g|^2 (g^H J^-1 g)."""
    t = system.targets[target_idx]
    M = t.J.shape[0]
    g = np.zeros((M, 1), dtype=complex)
    if t.direct is not None:
        g += t.direct
    for p in system.panels:
        if p.state.a == 0:
            continue
        g += echo_vec(p.channels.G, p.state.phi_vec, p.channels.b_by_target[t.target_id])
    g = g.reshape(-1)
    val = system.p_total_linear * (np.linalg.norm(g) ** 2) * (g.conj() @ np.linalg.solve(t.J, g)).real
    return t.rcs * val
