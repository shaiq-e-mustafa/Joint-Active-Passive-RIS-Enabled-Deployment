from src.sys.channels import PanelChannels, UserChannels
from src.sys.risInfo import PanelState
from src.utils.channel_utils import to_linear, q_function
import numpy as np
from dataclasses import dataclass
from src.utils.config import settings
from src.sys.convention import (reciprocal, echo_vec, comm_vec, direct_comm, echo_operator)

@dataclass
class RISPanel:
    panel_id: int
    channels: PanelChannels
    state: PanelState
    pos: np.ndarray = None      # (x, y) in meters, set by factory.build_system()
    elements: np.ndarray = None  # (L, 3) element positions (3-D UPA model, see src/channel/geometry3d.py)
    normal: np.ndarray = None    # (2,) horizontal panel normal

@dataclass
class UserLink:
    user_id: int
    channels: UserChannels
    h_bar: np.ndarray = None            # derived — set by build_hbar()
    w: np.ndarray = None                # derived — set by build_w_mrt()
    sinr: int = None
    air: int = None
    ber: int = None
    pos: np.ndarray = None              # (x, y) in meters, set by factory.build_system()

@dataclass
class TargetLink:
    target_id: int
    rcs: int
    b_i: np.ndarray = None
    pos: np.ndarray = None              # (x, y) in meters, set by factory.build_system()
    direct: np.ndarray = None           # (M,1) BS<->target direct (LoS or NLoS) path in the g_i convention, or None
    


@dataclass
class ISACSystem:
    panels: list
    users: list
    targets: list
    p_total_linear: float
    rng: np.random.Generator
    ber_comm: float = None
    # Amplitude scale on the direct BS-user link. 1.0 = unchanged (default), 0.0 = direct link
    # fully blocked (RIS-only comm). See docs/phase_a_log.html Entry 12.
    direct_link_scale: float = 1.0
    # False = current model (echo returns through the SAME panel it left from: T = sum_i g_i g_i^H).
    # True  = all-pairs model (echo may return through ANY panel: T = (sum_i g_i)(sum_i g_i)^H).
    cross_panel_T: bool = False

    """
    USER CHANNEL
    """

    def build_hbar(self):
        """hbar_k with y_k = hbar_k^H x. Reciprocal convention: hbar_k = conj(h_d,k) + sum_i a_i conj(G_i^T Phi_i f_ik)."""
        active_panels = [panel for panel in self.panels if panel.state.a != 0]
        for user in self.users:
            h_bar = self.direct_link_scale * direct_comm(user.channels.hdk)
            for panel in active_panels:
                f_i = panel.channels.f_by_user[user.user_id]
                h_bar = h_bar + panel.state.a * comm_vec(panel.channels.G, panel.state.phi_vec, f_i)
            user.h_bar = h_bar

    def build_w_mrt(self, equal_split: bool = True, weights: np.ndarray = None):
        K = len(self.users)
        p_k_arr = (np.full(K, self.p_total_linear / K) if equal_split
                   else self.p_total_linear * weights)
        
        for user, p_k in zip(self.users, p_k_arr):
            norm = np.linalg.norm(user.h_bar)
            user.w = np.sqrt(p_k) * (user.h_bar / norm)

    def transmit_waveform(self, s: np.ndarray) -> np.ndarray:
        W = np.hstack([u.w for u in self.users])   # M x K
        return W @ s
    
    def get_received_signal(self, s, rng):
        """
        s: K x N_symbols
        Returns y: K x N_symbols, plus breakdown for diagnostics.
        """
        K, N = s.shape
        x = self.transmit_waveform(s)          # M x N

        y = np.zeros((K, N), dtype=complex)
        desired = np.zeros((K, N), dtype=complex)
        interference = np.zeros((K, N), dtype=complex)

        noise_power_linear = to_linear(
            float(settings.config.channel_model.reciever_nosie) - 30
        )

        for k, user in enumerate(self.users):
            h_bar_k = user.h_bar                      # M x 1

            # full channel output (desired + inter-user interference), Eq. 6 first term
            y_k = (h_bar_k.conj().T @ x).flatten()     # (N,)

            # isolate the desired-only component for plotting/sanity-check
            desired[k, :] = (h_bar_k.conj().T @ self.users[k].w).item() * s[k, :]
            interference[k, :] = y_k - desired[k, :]

            # active-RIS amplified noise, per symbol (resampled each call)
            amp_noise = np.zeros(N, dtype=complex)
            for panel in self.panels:
                if not panel.state.active:
                    continue
                f_i = panel.channels.f_by_user[k]
                v_i = panel.state.noise
                f_row = f_i.T if reciprocal() else f_i.conj().T      # noise v_i is reflected: f^T Phi v
                amp_noise += (f_row @ (panel.state.phi_vec * v_i)).flatten()

            # thermal noise, per symbol
            n_k = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) \
                * np.sqrt(noise_power_linear / 2)

            y[k, :] = y_k + amp_noise + n_k

        return y, desired, interference
    
    def build_sinr(self):
        sigma_v_sq = to_linear(float(settings.config.channel_model.active_ris_noise) - 30)
        sigma_k_sq = to_linear(float(settings.config.channel_model.reciever_nosie) - 30)

        for k, user_k in enumerate(self.users):
            signal_power = np.abs((user_k.h_bar.conj().T @ user_k.w)).item() ** 2

            interference_power = 0.0
            for j, user_j in enumerate(self.users):
                if j == k:
                    continue
                interference_power += np.abs((user_k.h_bar.conj().T @ user_j.w)).item() ** 2

            active_noise_power = 0.0
            for panel in self.panels:
                if not panel.state.active or panel.state.a == 0:
                    continue
                f_i = panel.channels.f_by_user[user_k.user_id]
                gain = f_i.conj().T * panel.state.phi_vec[None, :]
                active_noise_power += (np.linalg.norm(gain) ** 2) * sigma_v_sq

            denom = interference_power + active_noise_power + sigma_k_sq
            user_k.sinr = signal_power / denom

    def get_achievable_rate(self):
        for user_k in self.users:
            user_k.air = np.log2(1 + user_k.sinr) 

    def get_ber_per_user(self):
        for user_k in self.users:
            user_k.ber = q_function(np.sqrt(user_k.sinr))   # Gray QPSK, symbol SINR (the BPSK form Q(sqrt(2 SINR)) was 3 dB optimistic)
    
    def get_total_ber(self):
        tot_ber = 0
        for user_k in self.users:
            tot_ber += user_k.ber
        self.ber_comm = tot_ber / len(self.users)

    """
    TARGET CHANNEL
    """



    def build_T(self):
        """Echo operator per target:  y_r = alpha T x.
        Reciprocal: u_i = G_i^T Phi_i b_i, T = u u^T. all-pairs (cross_panel_T): u = u_d + sum_i a_i u_i (every out/return panel pair,
        plus the direct path); otherwise T = sum_i u_i u_i^T (+ u_d u_d^T), i.e. the echo returns through the panel it left from.
        Hermitian (legacy): u_i = G_i^H Phi_i b_i, T = u u^H."""
        M = settings.config.channel_model.M
        active_panels = [(panel, panel.channels.G, panel.state.phi_vec) for panel in self.panels if panel.state.a != 0]
        for target in self.targets:
            T = np.zeros((M, M), dtype=complex)
            u_sum = np.zeros((M, 1), dtype=complex)
            if target.direct is not None:
                if self.cross_panel_T:
                    u_sum += target.direct
                else:
                    T += echo_operator(target.direct)
            for panel, G_i, Phi_i in active_panels:
                u_i = echo_vec(G_i, Phi_i, panel.channels.b_by_target[target.target_id])
                if self.cross_panel_T:
                    u_sum += u_i
                else:
                    T += echo_operator(u_i)
            if self.cross_panel_T:
                T = echo_operator(u_sum)
            target.T = T

    def build_y_r(self, x, rng):
        """Echo plus noise for one snapshot x (M x N). Amplified noise of active panel i: the outbound v_i is reflected, scattered by
        the target (random amplitude alpha_t) and returns through every panel; the return noise v_i' goes straight to the BS."""
        M, N = x.shape
        sigma_r_sq = to_linear(float(settings.config.channel_model.reciever_nosie) - 30)
        results = {}
        for target in self.targets:
            alpha_t = (rng.standard_normal() + 1j * rng.standard_normal()) * np.sqrt(target.rcs / 2)
            y_echo = alpha_t * (target.T @ x)
            outbound_noise = np.zeros((M, 1), dtype=complex)
            return_noise = np.zeros((M, 1), dtype=complex)
            u_tot = target.direct.copy() if target.direct is not None else np.zeros((M, 1), dtype=complex)
            for panel in self.panels:
                if panel.state.a != 0:
                    u_tot = u_tot + echo_vec(panel.channels.G, panel.state.phi_vec, panel.channels.b_by_target[target.target_id])
            for panel in self.panels:
                if not panel.state.active or panel.state.a == 0:
                    continue
                G_i = panel.channels.G
                phi = panel.state.phi_vec[:, None]
                b_i = panel.channels.b_by_target[target.target_id]
                v_i = panel.state.noise.reshape(-1, 1)
                v_ret = panel.state.noise_2.reshape(-1, 1)
                b_row = b_i.T if reciprocal() else b_i.conj().T
                s_out = (b_row @ (phi * v_i)).item()
                u_i = echo_vec(G_i, panel.state.phi_vec, b_i)
                outbound_noise += alpha_t * s_out * (u_tot if self.cross_panel_T else u_i)
                G_ret = G_i.T if reciprocal() else G_i.conj().T
                return_noise += G_ret @ ((phi if reciprocal() else phi.conj()) * v_ret)
            n_r = ((rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))) * np.sqrt(sigma_r_sq / 2))
            results[target.target_id] = y_echo + outbound_noise + return_noise + n_r
        return results

    def get_Rx(self) -> np.ndarray:
        """R_x = sum_k w_k w_k^H, Eq. (2)."""
        return sum(u.w @ u.w.conj().T for u in self.users)

    
    def build_J(self, x=None):
        """Noise-plus-interference covariance at the BS receive array, ACTIVE panels only (passive panels have no amplifier):
            J = sigma_r^2 I + sigma_v^2 sum_i G_i^T diag(|phi_i|^2) conj(G_i)                (return-hop amplifier noise)
                + sigma_t^2 sigma_v^2 (sum_i c_i) u u^H,   c_i = sum_l |phi_il|^2 |b_il|^2    (outbound noise scattered by the target)
        u = total echo vector (all-pairs); the diagonal operator uses c_i u_i u_i^H per panel. Hermitian legacy: G^H in the first term."""
        M = settings.config.channel_model.M
        sigma_v_sq = to_linear(float(settings.config.channel_model.active_ris_noise) - 30)
        sigma_r_sq = to_linear(float(settings.config.channel_model.reciever_nosie) - 30)

        amp = [p for p in self.panels if p.state.a != 0 and p.state.active]
        J_return = np.zeros((M, M), dtype=complex)
        for p in amp:
            G_i, w2 = p.channels.G, np.abs(p.state.phi_vec) ** 2
            if reciprocal():
                J_return += sigma_v_sq * ((G_i.T * w2[None, :]) @ G_i.conj())
            else:
                J_return += sigma_v_sq * ((G_i.conj().T * w2[None, :]) @ G_i)

        for target in self.targets:
            sigma_t_sq = target.rcs
            J = J_return.copy()
            if amp:
                c_tot = 0.0
                u_tot = np.zeros((M, 1), dtype=complex)
                if target.direct is not None:
                    u_tot += target.direct
                for p in self.panels:
                    if p.state.a != 0:
                        u_tot += echo_vec(p.channels.G, p.state.phi_vec, p.channels.b_by_target[target.target_id])
                for p in amp:
                    b = p.channels.b_by_target[target.target_id]
                    c_i = float(np.sum((np.abs(p.state.phi_vec) ** 2)[:, None] * np.abs(b) ** 2))
                    if self.cross_panel_T:
                        c_tot += c_i
                    else:
                        u_i = echo_vec(p.channels.G, p.state.phi_vec, b)
                        J += sigma_t_sq * sigma_v_sq * c_i * (u_i @ u_i.conj().T)
                if self.cross_panel_T:
                    J += sigma_t_sq * sigma_v_sq * c_tot * (u_tot @ u_tot.conj().T)
            J += sigma_r_sq * np.eye(M)
            target.J = J

    def compute_radar_snr(self, multi_target_interference: bool = True):
        """
        Args:
            multi_target_interference: If True and |targets| > 1, compute SINR with
                                      mutual-target interference; else compute plain SNR.

        Returns:
            snr_per_target: dict mapping target_id → SNR/SINR value (linear)
            avg_snr: average SNR/SINR across all targets
        """
        Rx = self.get_Rx()
        snr_per_target = {}

        if len(self.targets) == 1 or not multi_target_interference:
            # Single-target or no interference: plain SNR
            for target in self.targets:
                sigma_t_sq = target.rcs
                J_inv = np.linalg.inv(target.J)
                snr_per_target[target.target_id] = (
                    sigma_t_sq * np.trace(target.T @ Rx @ target.T.conj().T @ J_inv).real
                )
        else:
            # Multi-target: SINR with mutual-target interference
            for q, target_q in enumerate(self.targets):
                sigma_q_sq = target_q.rcs

                # Build J_q = Σ_{q'≠q} σ_{q'}² T^(q') R_x (T^(q'))† + J_noise
                J_q = target_q.J.copy()  # Start with noise-only part

                # Add mutual-target interference from all other targets
                for q_prime, target_q_prime in enumerate(self.targets):
                    if q_prime == q:
                        continue
                    sigma_q_prime_sq = target_q_prime.rcs
                    interference_term = (
                        sigma_q_prime_sq * (
                            target_q_prime.T @ Rx @ target_q_prime.T.conj().T
                        )
                    )
                    J_q += interference_term

                J_q_inv = np.linalg.inv(J_q)
                sinr_radar = (
                    sigma_q_sq * np.trace(
                        target_q.T @ Rx @ target_q.T.conj().T @ J_q_inv
                    ).real
                )
                snr_per_target[target_q.target_id] = sinr_radar

        self.snr_per_target = snr_per_target

        total_targets = len(self.targets)
        avg_snr = sum(snr_per_target.values()) / total_targets

        return avg_snr, snr_per_target

        

