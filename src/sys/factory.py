from src.channel.risConfig import get_link_params, is_blocked, is_blocked_line_boolean, calibrate_lambda_b
from src.channel.channel_model import get_path_loss_linear, get_hybrid_channel_exact, wavefront_is_exact
from src.channel.fading import get_Gi, get_bi, get_fi, get_hdk, get_rtt
from src.sys.risInfo import PanelState
from src.sys.system import RISPanel, UserLink, ISACSystem, TargetLink
from src.sys.channels import  PanelChannels, UserChannels
from src.utils.channel_utils import to_linear, distance, sample_polar, bearing, random_rcs
from src.utils.config import settings
from src.sim.deployment import BS_POS, PANEL_RADIUS, USER_RADIUS, TARGET_RADIUS, TARGET_RCS_LIM
import numpy as np

def sample_geometry(n_panels, k_users, k_targets, rng, target_p_block=0.0, mean_obstacle_len=5.0,
                     panel_radius=None, target_radius=None, user_radius=None):
    """Wraps your existing deployment.py sampling — positions + blockage only.

    panel_radius/target_radius/user_radius override the deployment.py defaults
    (PANEL_RADIUS/TARGET_RADIUS/USER_RADIUS) when given, for geometry
    sensitivity sweeps (see src/opt/geometry_sensitivity.py).

    Panel blockage uses the Line Boolean Model (PPP obstacles), calibrated so
    that a panel at the mean BS-panel distance is blocked with probability
    target_p_block. Returns panel_active as 1=active/0=blocked, matching
    PanelState.a semantics directly.
    """
    panel_radius = panel_radius if panel_radius is not None else PANEL_RADIUS
    target_radius = target_radius if target_radius is not None else TARGET_RADIUS
    user_radius = user_radius if user_radius is not None else USER_RADIUS

    panel_pos = [sample_polar(panel_radius, (0, 2*np.pi), rng) for _ in range(n_panels)]

    d_ref = 0.5 * (panel_radius[0] + panel_radius[1])
    lambda_b = calibrate_lambda_b(target_p_block, d_ref, mean_obstacle_len)
    panel_active = [
        0 if is_blocked_line_boolean(distance(BS_POS, p), rng, lambda_b, mean_obstacle_len) else 1
        for p in panel_pos
    ]

    user_pos = [sample_polar(user_radius, (0, 2*np.pi), rng) for _ in range(k_users)]
    target_pos = [sample_polar(target_radius, (0, 2*np.pi), rng) for _ in range(k_targets)]
    target_rcs = [random_rcs((0.1, TARGET_RCS_LIM), rng) for _ in range(k_targets)]
    return panel_pos, panel_active, user_pos, target_pos, target_rcs


def make_panel_state(active, panel_active, L, rng):
    return PanelState(
        active=bool(active), a=int(panel_active),
        phases=rng.uniform(0, 2*np.pi, size=L),
        gains=(rng.uniform(0, np.sqrt(to_linear(settings.config.channel_model.pmax_dB)), size=L)
               if active else None),
        noise=(
            (np.sqrt(10 ** ((int(settings.config.channel_model.active_ris_noise) - 30) / 10) / 2)
             * (rng.standard_normal(L) + 1j * rng.standard_normal(L))) if active else 0),
        noise_2=(
            (np.sqrt(10 ** ((int(settings.config.channel_model.active_ris_noise) - 30) / 10) / 2)
             * (rng.standard_normal(L) + 1j * rng.standard_normal(L))) if active else 0
        )
    )


def build_one_panel_channel(ppos, target_pos, user_pos, active, panel_active, L, M, rng):
    """Build (PanelChannels, PanelState) for a single panel at ppos. Extracted
    from realize_channels()'s loop body so greedy placement (src/opt/
    greedy_placement.py) can add one candidate panel at a time without
    rebuilding every already-placed panel's channels from scratch.
    """
    # BS <-> panel
    kappa_g, eta_g, blocked_g = get_link_params(BS_POS, ppos, rng)
    beta_g = get_path_loss_linear(distance(BS_POS, ppos), eta_g)
    exact = wavefront_is_exact()
    if exact:
        G_i = get_hybrid_channel_exact(ppos, L, BS_POS, M, kappa_g, eta_g, rng)
    else:
        G_i = get_Gi(beta_g, kappa_g, L, M,
                      angle_rx=bearing(ppos, BS_POS),
                      angle_tx=bearing(BS_POS, ppos), rng=rng)

    # panel <-> target
    b_by_target = {}
    for k, tpos in enumerate(target_pos):
        kappa_b, eta_b, blocked_b = get_link_params(ppos, tpos, rng)
        beta_b = get_path_loss_linear(distance(ppos, tpos), eta_b)
        b_by_target[k] = (get_hybrid_channel_exact(ppos, L, tpos, 1, kappa_b, eta_b, rng) if exact else
                          get_bi(beta_b, kappa_b, L,
                                 angle_rx=bearing(ppos, tpos),
                                 angle_tx=bearing(tpos, ppos), rng=rng))

    # panel <-> each user (independent blockage per user!)
    f_by_user = {}
    for k, upos in enumerate(user_pos):
        kappa_f, eta_f, blocked_f = get_link_params(ppos, upos, rng)
        beta_f = get_path_loss_linear(distance(ppos, upos), eta_f)
        f_by_user[k] = (get_hybrid_channel_exact(ppos, L, upos, 1, kappa_f, eta_f, rng) if exact else
                        get_fi(beta_f, kappa_f, L,
                               angle_rx=bearing(ppos, upos),
                               angle_tx=bearing(upos, ppos), rng=rng))

    channels = PanelChannels(G=G_i, b_by_target=b_by_target, f_by_user=f_by_user)

    state = make_panel_state(active, panel_active, L, rng)
    return channels, state


def make_direct_channel(upos, M, rng):
    """Direct BS<->user link (Eq. 16): kappa = 0 (pure NLoS fading); eta from the link's LoS/NLoS draw."""
    eta = get_link_params(BS_POS, upos, rng)[1]
    if wavefront_is_exact():
        return UserChannels(hdk=get_hybrid_channel_exact(BS_POS, M, upos, 1, 0.0, eta, rng))
    return UserChannels(hdk=get_hdk(
        beta_hdk=get_path_loss_linear(distance(BS_POS, upos), eta),
        rx_elements=M, angle_rx=bearing(upos, BS_POS), rng=rng))


def realize_channels(panel_pos, user_pos, target_pos, active_mask, L, M, rng, panel_active=None):
    if panel_active is None:
        panel_active = [1] * len(panel_pos)
    panels_channels, panels_state = [], []

    for i, ppos in enumerate(panel_pos):
        channels, state = build_one_panel_channel(
            ppos, target_pos, user_pos, active_mask[i], panel_active[i], L, M, rng
        )
        panels_channels.append(channels)
        panels_state.append(state)

    users_channels = [make_direct_channel(upos, M, rng) for upos in user_pos]

    return panels_channels, panels_state, users_channels

def build_system(n_panels, k_users, k_targets, active_mask, L, M, p_total_linear, rng,
                  target_p_block=0.0, mean_obstacle_len=5.0,
                  panel_radius=None, target_radius=None, user_radius=None):
    panel_pos, panel_active, user_pos, target_pos, target_rcs = sample_geometry(
        n_panels, k_users, k_targets, rng, target_p_block=target_p_block, mean_obstacle_len=mean_obstacle_len,
        panel_radius=panel_radius, target_radius=target_radius, user_radius=user_radius,
    )
    panels_ch, panels_st, users_ch = realize_channels(
        panel_pos=panel_pos, user_pos=user_pos, active_mask=active_mask, panel_active=panel_active,
        target_pos=target_pos, L=L, M=M, rng=rng
    )

    panels = [RISPanel(panel_id=i, channels=c, state=s, pos=p)
              for i, (c, s, p) in enumerate(zip(panels_ch, panels_st, panel_pos))]
    users = [UserLink(user_id=k, channels=c, pos=p) for k, (c, p) in enumerate(zip(users_ch, user_pos))]
    targets = [TargetLink(target_id=k, rcs=target_rcs[k], pos=target_pos[k]) for k in range(k_targets)]

    return ISACSystem(panels=panels, users=users, p_total_linear=p_total_linear, targets=targets, rng=rng)