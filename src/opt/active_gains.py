"""Power-budgeted active-RIS gains (configs/default.yaml `gain_mode: budget`).

Model of Long, Liang, Pei, Larsson, "Active RIS-aided wireless communications" (arXiv:2103.00709):
  every active element needs circuit power P_c + DC power P_DC, so a panel of L active elements leaves an amplification-output budget
      P_out = upsilon * (P_RIS - L (P_c + P_DC))                                   (their Eq. 17)
  and the amplified signal plus amplified noise must fit it:
      sum_l a_l^2 (|(G x)_l|^2 + sigma_v^2) <= P_out,   a_l <= a_max                 (their Eq. 21b/21c)
  For phase-aligned elements the optimum is a uniform amplitude (their Eq. after 34):
      a = min( a_max, sqrt( P_out / sum_l (|(G x)_l|^2 + sigma_v^2) ) ).
A panel whose budget cannot even power its active circuits (P_out <= 0) falls back to passive. Passive elements still cost P_c each.
Our parameters: upsilon = 0.8, P_c = -10 dBm, P_DC = -5 dBm (Long et al. examples); P_RIS (per panel) and the element gain cap are conventions
in configs/default.yaml (Long et al. use P_RIS = 10 dBm and a_max = 40 dB; we use 30 dBm and 10 dB power gain, the draft's p_max).
"""
import numpy as np
from src.utils.channel_utils import to_linear, to_linear_dbm
from src.channel.geometry3d import cfg
from src.sys.convention import echo_vec, illumination
from src.opt.phase_alignment import coherent_phases_upa


def params():
    return dict(
        upsilon=float(cfg("amplifier_efficiency", 0.8)),
        p_c=to_linear_dbm(float(cfg("ris_element_power_dbm", -10.0))),
        p_dc=to_linear_dbm(float(cfg("ris_active_dc_dbm", -5.0))),
        p_ris=to_linear_dbm(float(cfg("ris_panel_power_dbm", 30.0))),
        a_max=float(np.sqrt(to_linear(float(cfg("pmax_dB", 10.0))))),
    )


def panel_power_w(L, active):
    """(circuit + DC) power in watts that a panel of L elements needs just to operate."""
    q = params()
    return L * (q["p_c"] + (q["p_dc"] if active else 0.0))


def p_out_w(L):
    q = params()
    return q["upsilon"] * (q["p_ris"] - panel_power_w(L, True))


def incident_power_w(system, panel, beams):
    """Largest power the panel's elements receive over the given transmit beams (each a length-M vector with ||x||^2 = P)."""
    return max(float(np.sum(np.abs(panel.channels.G @ x) ** 2)) for x in beams)


def target_beams(system):
    """Matched transmit beams (one per target, all selected panels phase-aligned to it, unit gain): the strongest illumination a panel sees."""
    beams = []
    for t in system.targets:
        u = np.zeros((system.panels[0].channels.G.shape[1], 1), dtype=complex)
        if t.direct is not None:
            u += t.direct
        for p in system.panels:
            if p.state.a:
                phi = np.exp(1j * coherent_phases_upa(p, t.pos, "radar"))
                u += echo_vec(p.channels.G, phi, p.channels.b_by_target[t.target_id])
        v = illumination(u)[:, 0]
        nrm = np.linalg.norm(v)
        beams.append(np.sqrt(system.p_total_linear) * v / nrm if nrm > 0 else np.zeros_like(v))
    return beams


def assign_budget_gains(system):
    """Set every active panel's per-element amplitude by the budget rule; panels that cannot power their active circuits become passive.
    Returns a report dict (gain per active panel, number downgraded, power used)."""
    q = params()
    sv = to_linear(float(cfg("active_ris_noise", -96.0)) - 30)
    beams = target_beams(system) if system.targets else []
    gains, downgraded, power = [], 0, []
    for p in system.panels:
        L = p.state.phases.shape[0]
        if not p.state.active:
            power.append(panel_power_w(L, False))
            continue
        pout = p_out_w(L)
        if pout <= 0:
            p.state.active = False
            p.state.gains = None
            downgraded += 1
            power.append(panel_power_w(L, False))
            continue
        pin = (incident_power_w(system, p, beams) if beams else 0.0) + L * sv
        a = min(q["a_max"], float(np.sqrt(pout / pin)))
        p.state.gains = np.full(L, a)
        gains.append(a)
        power.append(panel_power_w(L, True) + min(pout, a ** 2 * pin) / q["upsilon"])
    return dict(gains=gains, downgraded=downgraded, panel_power_w=power)
