import numpy as np
from src.utils.channel_utils import to_db

def validate_sinr_montecarlo(system, y, desired, tol_db=0.5):
    """
    Empirical SINR from Monte Carlo symbols vs. closed-form build_sinr().
    Uses (y - desired), not the returned `interference` array, since that
    array excludes amp_noise and thermal noise (added after it's computed).
    """
    K = len(system.users)
    print(f"{'user':>4} {'empirical (dB)':>16} {'closed-form (dB)':>18} {'diff (dB)':>10}")
    for k, user in enumerate(system.users):
        signal_energy = np.mean(np.abs(desired[k, :]) ** 2)
        noise_interference_energy = np.mean(np.abs(y[k, :] - desired[k, :]) ** 2)
        empirical_sinr = signal_energy / noise_interference_energy

        empirical_db = to_db(empirical_sinr)
        closed_form_db = to_db(user.sinr)
        diff = empirical_db - closed_form_db

        flag = "" if abs(diff) < tol_db else "  <-- MISMATCH"
        print(f"{k:>4} {empirical_db:>16.4f} {closed_form_db:>18.4f} {diff:>10.4f}{flag}")


def plot(y, desired, k, s):
    import matplotlib.pyplot as plt
    
    n_show = 1000

    # Normalize desired and y by their average RMS amplitude
    scale_desired = np.sqrt(np.mean(np.abs(desired[k, :n_show])**2))
    scale_y = np.sqrt(np.mean(np.abs(y[k, :n_show])**2))

    desired_norm = desired[k, :n_show] / scale_desired
    y_norm = y[k, :n_show] / scale_y
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Transmitted Symbols
    axes[0].scatter(s[k, :n_show].real, s[k, :n_show].imag, color="black", marker="x")
    axes[0].set_title("Transmitted Symbols (s)")
    axes[0].set_xlabel("I")
    axes[0].set_ylabel("Q")
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Unscaled Received Signal (Autoscaled to 1e-4)
    axes[1].scatter(y[k, :n_show].real, y[k, :n_show].imag, color="red", alpha=0.3, s=15, label="Rx Signal (y)")
    axes[1].scatter(desired[k, :n_show].real, desired[k, :n_show].imag, color="blue", alpha=0.5, s=20, label="Desired")
    axes[1].set_title(f"User {k} Unscaled Received Constellation (Physical Amplitudes)")
    axes[1].set_xlabel("I (Volts)")
    axes[1].set_ylabel("Q (Volts)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()