from src.channel.risConfig import get_link_params, is_blocked
from src.channel.channel_model import get_path_loss_linear
from src.channel.fading import get_Gi, get_bi, get_fi, get_hdk, get_rtt
from src.sys.risInfo import PanelState
from src.sys.entities import RISPanel, UserLink, ISACSystem
from src.sys.channels import  PanelChannels, UserChannels, TargetChannels
from src.utils.channel_utils import to_linear, distance, sample_polar, bearing
from src.utils.config import settings
from src.sim.deployment import BS_POS, PANEL_RADIUS, USER_RADIUS, TARGET_RADIUS
import numpy as np

def sample_geometry(n_panels, k_users, rng):
    """Wraps your existing deployment.py sampling — positions + blockage only."""
    panel_pos = [sample_polar(PANEL_RADIUS, (0, 2*np.pi), rng) for _ in range(n_panels)]
    panel_blocked = [is_blocked(distance(BS_POS, p), rng) for p in panel_pos]
    user_pos = [sample_polar(USER_RADIUS, (0, 2*np.pi), rng) for _ in range(k_users)]
    target_pos = sample_polar(TARGET_RADIUS, (0, 2*np.pi), rng)
    return panel_pos, panel_blocked, user_pos, target_pos


def realize_channels(panel_pos, user_pos, target_pos, active_mask, L, M, rng):
    panels_channels, panels_state, link_blockage = [], [], []

    for i, ppos in enumerate(panel_pos):
        # BS <-> panel i
        kappa_g, eta_g, blocked_g = get_link_params(BS_POS, ppos, rng)
        beta_g = get_path_loss_linear(distance(BS_POS, ppos), eta_g)
        G_i = get_Gi(beta_g, kappa_g, L, M,
                      angle_rx=bearing(ppos, BS_POS),
                      angle_tx=bearing(BS_POS, ppos), rng=rng)

        # panel i <-> target
        kappa_b, eta_b, blocked_b = get_link_params(ppos, target_pos, rng)
        beta_b = get_path_loss_linear(distance(ppos, target_pos), eta_b)
        b_i = get_bi(beta_b, kappa_b, L,
                      angle_rx=bearing(ppos, target_pos),
                      angle_tx=bearing(target_pos, ppos), rng=rng)

        # panel i <-> each user k  (independent blockage per user!)
        f_by_user = {}
        for k, upos in enumerate(user_pos):
            kappa_f, eta_f, blocked_f = get_link_params(ppos, upos, rng)
            beta_f = get_path_loss_linear(distance(ppos, upos), eta_f)
            f_by_user[k] = get_fi(beta_f, kappa_f, L,
                                    angle_rx=bearing(ppos, upos),
                                    angle_tx=bearing(upos, ppos), rng=rng)

        panels_channels.append(PanelChannels(G=G_i, b=b_i, f_by_user=f_by_user))

        state = PanelState(
            active=bool(active_mask[i]), a=1,
            phases=rng.uniform(0, 2*np.pi, size=L),
            gains=(rng.uniform(0, np.sqrt(to_linear(settings.config.channel_model.pmax_dB)), size=L)
                   if active_mask[i] else None),
            noise=(
                (
                    np.sqrt(10 ** ((int(settings.config.channel_model.active_ris_noise) - 30) / 10) / 2) * (rng.standard_normal(L) + 1j * rng.standard_normal(L))) if active_mask[i] else 0)
            
        )
        panels_state.append(state)

    users_channels = [
    UserChannels(
        hdk=get_hdk(
            beta_hdk=get_path_loss_linear(
                distance(BS_POS, upos),
                get_link_params(BS_POS, upos, rng)[1]
            ),
            rx_elements=M,
            angle_rx=bearing(upos, BS_POS),
            rng=rng
        )
    )
        for upos in user_pos
    ]

    target_channels = [
        TargetChannels(
            rtt=get_rtt(

            )
        )
    ]

    return panels_channels, panels_state, users_channels

def build_system(n_panels, k_users, active_mask, L, M, p_total_linear, rng):
    panel_pos, panel_blocked, user_pos, target_pos = sample_geometry(n_panels, k_users, rng)
    panels_ch, panels_st, users_ch = realize_channels(panel_pos=panel_pos, user_pos=user_pos, active_mask=active_mask, target_pos=target_pos, L=L, M=M, rng=rng)

    panels = [RISPanel(panel_id=i, channels=c, state=s) for i, (c, s) in enumerate(zip(panels_ch, panels_st))]
    users = [UserLink(user_id=k, channels=c) for k, c in enumerate(users_ch)]

    return ISACSystem(panels=panels, users=users, p_total_linear=p_total_linear, rng=rng)