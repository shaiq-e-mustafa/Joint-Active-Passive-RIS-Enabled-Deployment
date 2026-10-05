"""The noise levels in configs/default.yaml are used as given (no integer truncation). Run: python tests/test_noise_config.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.utils.channel_utils import to_linear
from src.sys import scenario as sc

ok = True
cm = settings.config.channel_model
users, targets = sc.make_users_targets(2, 1, np.random.default_rng(0), direct_mode="none")
saved = (cm.reciever_nosie, cm.active_ris_noise)
try:
    cm.reciever_nosie, cm.active_ris_noise = -96.5, -96.5          # a value int() would truncate to -96
    system = sc.build_system([np.array([100.0, 20.0])], users, targets, 4, 4, lambda i: False, 1.0, np.random.default_rng(1))
    system.build_J()
    # a passive-only system: J = sigma_r^2 I, so every diagonal entry is the receiver noise power
    got = float(np.real(np.diag(system.targets[0].J)).mean())
    want = to_linear(-96.5 - 30)
    trunc = to_linear(-96 - 30)
finally:
    cm.reciever_nosie, cm.active_ris_noise = saved
ok = abs(got - want) / want < 1e-9 and abs(got - trunc) / trunc > 0.05
print(f"[{'PASS' if ok else 'FAIL'}] receiver noise -96.5 dBm is used as given (J diagonal {got:.4e}, -96.5 dBm {want:.4e}, truncated -96 dBm {trunc:.4e})")
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
