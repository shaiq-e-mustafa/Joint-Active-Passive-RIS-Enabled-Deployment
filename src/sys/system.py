from src.sys.channels import PanelChannels, UserChannels
from src.sys.risInfo import PanelState
from src.utils.channel_utils import to_linear, q_function
import numpy as np
from dataclasses import dataclass
from src.utils.config import settings

@dataclass
class RISPanel:
    panel_id: int
    channels: PanelChannels
    state: PanelState
    pos: np.ndarray = None      # (x, y) in meters, set by factory.build_system()

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
    


@dataclass
class ISACSystem:
    panels: list
    users: list
    targets: list
    p_total_linear: float
    rng: np.random.Generator
    ber_comm: float = None

    """
    USER CHANNEL 
    """

    def build_hbar(self):
        # G_i^H @ Phi_i^H depends only on the panel, not the user -- precompute
        # once per active panel instead of recomputing it inside the user loop.
        active_panels = [
            (panel, panel.channels.G.conj().T @ panel.state.phi.conj().T)
            for panel in self.panels if panel.state.a != 0
        ]
        for user in self.users:
            h_bar = user.channels.hdk.copy()
            for panel, M_i in active_panels:
                f_i = panel.channels.f_by_user[user.user_id]
                h_bar = h_bar + panel.state.a * (M_i @ f_i)
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
            int(settings.config.channel_model.reciever_nosie) - 30
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
                Phi_i = panel.state.phi
                v_i = panel.state.noise
                L = f_i.shape[0]
                amp_noise += (f_i.conj().T @ Phi_i @ v_i).flatten()

            # thermal noise, per symbol
            n_k = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) \
                * np.sqrt(noise_power_linear / 2)

            y[k, :] = y_k + amp_noise + n_k

        return y, desired, interference
    
    def build_sinr(self):
        sigma_v_sq = to_linear(int(settings.config.channel_model.active_ris_noise) - 30)
        sigma_k_sq = to_linear(int(settings.config.channel_model.reciever_nosie) - 30)

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
                gain = f_i.conj().T @ panel.state.phi
                active_noise_power += (np.linalg.norm(gain) ** 2) * sigma_v_sq

            denom = interference_power + active_noise_power + sigma_k_sq
            user_k.sinr = signal_power / denom

    def get_achievable_rate(self):
        for user_k in self.users:
            user_k.air = np.log2(1 + user_k.sinr) 

    def get_ber_per_user(self):
        for user_k in self.users:
            user_k.ber = q_function(np.sqrt(2 * user_k.sinr))
    
    def get_total_ber(self):
        tot_ber = 0
        for user_k in self.users:
            tot_ber += user_k.ber
        self.ber_comm = tot_ber / len(self.users)

    """
    TARGET CHANNEL
    """



    def build_T(self):
        """
        Build the deterministic round-trip sensing operator T_i for each target.

        Following Clarifications.tex correction (i): Φ_i appears on BOTH reflections
        (reciprocal RIS applies same phase on outbound and return).

        T_i = G_i† (Φ_i b_i)(Φ_i b_i)† G_i  [deterministic, RCS-free]
        T = Σ_i a_i T_i

        Eq. (31) in corrected form from Clarifications.
        """
        M = settings.config.channel_model.M
        # G_i, Phi_i are panel-only (target-independent) -- fetch once per
        # active panel instead of recomputing inside the target loop.
        active_panels = [
            (panel, panel.channels.G, panel.state.phi)
            for panel in self.panels if panel.state.a != 0
        ]
        for target in self.targets:
            T = np.zeros((M, M), dtype=complex)
            for panel, G_i, Phi_i in active_panels:
                b_i = panel.channels.b_by_target[target.target_id]

                # Φ_i on BOTH reflections (not Φ_i† on return)
                # T_i = G_i† Φ_i b_i b_i† Φ_i G_i
                reflected_b = Phi_i @ b_i
                T_i = G_i.conj().T @ reflected_b @ reflected_b.conj().T @ G_i
                T += T_i

            target.T = T

    def build_y_r(self, x, rng):
       
        M, N = x.shape
        sigma_r_sq = to_linear(int(settings.config.channel_model.reciever_nosie) - 30)  # confirm: dedicated radar-Rx noise figure, or reusing comm's?

        results = {}
        for target in self.targets:
            sigma_t_sq = target.rcs
            alpha_t = (rng.standard_normal() + 1j*rng.standard_normal()) * np.sqrt(sigma_t_sq / 2)
            y_echo = alpha_t * (target.T @ x)
            outbound_noise = np.zeros((M, 1), dtype=complex)
            return_noise = np.zeros((M, 1), dtype=complex)

            for panel in self.panels:
                if not panel.state.active or panel.state.a == 0:
                    continue

                G_i = panel.channels.G
                Phi_i = panel.state.phi
                b_i = panel.channels.b_by_target[target.target_id]
                v_i = panel.state.noise.reshape(-1, 1)               # outbound -- same event as Eq.6
                v_i_return = panel.state.noise_2.reshape(-1, 1)      # return leg -- fresh, independent

                outbound_contrib = (G_i.conj().T @ Phi_i.conj().T
                                    @ b_i @ b_i.conj().T @ Phi_i @ v_i)
                outbound_noise += panel.state.a * target.rcs * outbound_contrib

                return_contrib = G_i.conj().T @ Phi_i.conj().T @ v_i_return
                return_noise += panel.state.a * return_contrib

            n_r = ((rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N)))
                * np.sqrt(sigma_r_sq / 2))

            results[target.target_id] = y_echo + outbound_noise + return_noise + n_r

        return results
    def get_Rx(self) -> np.ndarray:
        """R_x = sum_k w_k w_k^H, Eq. (2)."""
        return sum(u.w @ u.w.conj().T for u in self.users)

    
    def build_J(self, x=None):

        M = settings.config.channel_model.M
        sigma_v_sq = to_linear(int(settings.config.channel_model.active_ris_noise) - 30)
        sigma_r_sq = to_linear(int(settings.config.channel_model.reciever_nosie) - 30)

        # G_i, Phi_i, Phi_Phi_H are panel-only (target-independent) -- fetch/compute
        # once per active panel instead of recomputing inside the target loop.
        # The "direct return" noise term (term2) doesn't touch b_i or sigma_t_sq
        # either, so it's identical for every target and gets summed once.
        active_panels = []
        J_direct_return = np.zeros((M, M), dtype=complex)
        for panel in self.panels:
            if panel.state.a == 0:
                continue
            G_i = panel.channels.G
            Phi_i = panel.state.phi
            Phi_Phi_H = Phi_i @ Phi_i.conj().T  # For passive: I_L; for active: A_i²
            active_panels.append((panel, G_i, Phi_i, Phi_Phi_H))
            J_direct_return += sigma_v_sq * (G_i.conj().T @ Phi_Phi_H @ G_i)

        for target in self.targets:
            sigma_t_sq = target.rcs
            J = J_direct_return.copy()

            # RCS-scattered amplified noise term: σ_t² σ_v² × [term] (target-dependent via b_i)
            for panel, G_i, Phi_i, Phi_Phi_H in active_panels:
                b_i = panel.channels.b_by_target[target.target_id]  # L x 1
                reflected_b = Phi_i @ b_i  # L x 1
                term1 = (G_i.conj().T @ reflected_b @ reflected_b.conj().T @ Phi_Phi_H @
                         reflected_b @ reflected_b.conj().T @ G_i)
                J += sigma_t_sq * sigma_v_sq * term1

            # Receiver thermal noise
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

        

