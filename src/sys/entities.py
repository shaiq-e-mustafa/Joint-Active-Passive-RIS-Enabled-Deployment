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

@dataclass
class UserLink:
    user_id: int
    channels: UserChannels
    h_bar: np.ndarray = None            # derived — set by build_hbar()
    w: np.ndarray = None                # derived — set by build_w_mrt()
    sinr: int = None
    air: int = None
    ber: int = None

@dataclass 
class TargetLink:
    target_id: int
    channels: TargetChannels
    


@dataclass
class ISACSystem:
    panels: list
    users: list
    p_total_linear: float
    rng: np.random.Generator
    ber_comm: float = None

    def build_hbar(self):
        for user in self.users:
            h_bar = user.channels.hdk.copy()
            for panel in self.panels:
                if panel.state.a == 0:
                    continue
                f_i = panel.channels.f_by_user[user.user_id]
                h_bar = h_bar + panel.state.a * (
                    panel.channels.G.conj().T @ panel.state.phi.conj().T @ f_i
                )
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

