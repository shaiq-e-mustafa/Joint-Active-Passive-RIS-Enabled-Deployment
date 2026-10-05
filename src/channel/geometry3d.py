"""3-D UPA geometry and exact spherical-wave channels (docs/phase_a_log.html Entry 12).

Every entry of a link matrix uses the true 3-D distance d between a transmit element and a receive
element, both for the phase exp(-j 2 pi d / lambda) and for the amplitude

    |h|^2 = beta(d) * g_rx * F_rx(theta_rx) * g_tx * F_tx(theta_tx)           (LoS part)

where beta(d) is the project's path-loss law, F(theta) = max(cos theta, 0)^q is the element power
pattern (theta = angle from the array normal), and g is the element gain. Element models (all values come from configs/default.yaml; the defaults below equal the config values):
  RIS unit cell : F = cos^q with q = ris_cell_q (1.0), per-link gain sqrt(G_rx G_tx) with G_rx = 4 pi dx dy / lambda^2 (= pi for lambda/2 cells)
                  and G_tx = ris_cell_gain (pi, the aperture-limited gain of a lambda/2 cell). Tang et al. (arXiv:1911.05326) measured a larger cell:
                  ris_cell_q = 3, ris_cell_gain = 8, which reproduces their  Pr/Pt = G dx dy lambda^2 F F /((4 pi)^3 d1^2 d2^2)  and is
                  8.1 dB more optimistic per radar round trip; tests/test_upa3d.py sets those two values locally.
  BS element    : ideal half-wavelength aperture, g = 4 pi dx dy / lambda^2 = pi, F = cos theta (assumption).
  user / target : isotropic point, g = 1, F = 1.
"""
import numpy as np
from src.utils.config import settings
from src.channel.channel_model import get_path_loss_linear, _get_H_tilde, wavelength


def cfg(name, default):
    return getattr(settings.config.channel_model, name, default)


def spacing():
    return wavelength() / 2


def end_model(kind):
    """(peak per-link gain, cosine exponent q) of an array end."""
    cell_gain_rx = 4 * np.pi * spacing() ** 2 / wavelength() ** 2        # = pi for lambda/2 cells
    if kind == "ris":
        return float(np.sqrt(cell_gain_rx * float(cfg("ris_cell_gain", np.pi)))), float(cfg("ris_cell_q", 1.0))
    if kind == "bs":
        return float(cell_gain_rx), float(cfg("bs_cell_q", 1.0))
    if kind == "point":
        return 1.0, 0.0
    raise ValueError(kind)


def to3(pos2, z):
    p = np.asarray(pos2, dtype=float)
    return np.array([p[0], p[1], z])


def upa_elements(center2, z, normal2, nx, ny):
    """nx columns along the in-plane horizontal axis a = (-n_y, n_x), ny rows along z, spacing lambda/2,
    centered on (center2, z). Row-major (row j, column i -> index j*nx + i). Returns (nx*ny, 3)."""
    n = np.asarray(normal2, dtype=float)
    n = n / np.linalg.norm(n)
    a = np.array([-n[1], n[0]])
    s = spacing()
    cols = (np.arange(nx) - (nx - 1) / 2) * s
    rows = (np.arange(ny) - (ny - 1) / 2) * s
    jj, ii = np.meshgrid(rows, cols, indexing="ij")
    x = center2[0] + ii.ravel() * a[0]
    y = center2[1] + ii.ravel() * a[1]
    z_ = z + jj.ravel()
    return np.stack([x, y, z_], axis=1)


def panel_normal(ppos2, bs2, zone2):
    """Horizontal normal of a facade-mounted flat RIS: bisector of the directions to the BS and to the
    zone, so both lie in the front half-space whenever geometrically possible. A flat reflector cannot
    serve a BS and a zone on opposite sides (angle 180 deg): the normal is then arbitrary and the cos^q
    pattern zeroes the panel, which is the physically correct outcome."""
    p = np.asarray(ppos2, float)
    u1 = np.asarray(bs2, float) - p
    u2 = np.asarray(zone2, float) - p
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / (np.linalg.norm(u2) + 1e-12)
    n = u1 + u2
    if np.linalg.norm(n) < 1e-6:
        n = np.array([-u1[1], u1[0]])
    return n / np.linalg.norm(n)


def link_3d(rx_el, rx_end, tx_el, tx_end, kappa, eta, rng):
    """(R x T) hybrid Rician channel. rx_end/tx_end = dict(kind=..., normal=3-vector|None)."""
    diff = tx_el[None, :, :] - rx_el[:, None, :]                   # rx -> tx vectors, (R,T,3)
    d = np.linalg.norm(diff, axis=2)
    amp2 = get_path_loss_linear(d, eta)
    for end, sign in ((rx_end, +1), (tx_end, -1)):
        g, q = end_model(end["kind"])
        amp2 = amp2 * g
        if end["kind"] != "point":
            cos_t = np.clip(sign * (diff @ np.asarray(end["normal"], float)) / d, 0.0, None)
            amp2 = amp2 * cos_t ** q
    phase = np.exp(-1j * 2 * np.pi * d / wavelength())
    if rng is None:                         # deterministic pure-LoS channel (planning geometry)
        return np.sqrt(amp2) * phase
    los = np.sqrt(kappa / (kappa + 1)) * phase
    nlos = np.sqrt(1.0 / (kappa + 1)) * _get_H_tilde(rx_el.shape[0], tx_el.shape[0], rng)
    return np.sqrt(amp2) * (los + nlos)


def bs_array():
    """BS UPA: M = bs_cols * (M / bs_cols) elements, at height h_bs, facing bs_normal."""
    M = int(cfg("M", 64))
    cols = int(cfg("bs_cols", 8))
    rows = M // cols
    normal = np.array(cfg("bs_normal", [1.0, 0.0]), dtype=float)
    el = upa_elements(np.array([0.0, 0.0]), float(cfg("h_bs", 10.0)), normal, cols, rows)
    return el, dict(kind="bs", normal=np.array([normal[0], normal[1], 0.0]) / np.linalg.norm(normal))


def point_end():
    return dict(kind="point", normal=None)


def ris_end(normal2):
    n = np.asarray(normal2, float)
    return dict(kind="ris", normal=np.array([n[0], n[1], 0.0]))
