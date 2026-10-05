"""Does the per-scene role policy beat the old static split? Same sites, same scenes, same budget; only the role rule differs.
python out/policy_compare.py [n_scenes]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.opt import design as dz, evaluate as ev, loss as ls

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
scenes = ev.sample_scenes(n)
base = dz.greedy_design(20, active_fraction=0.5, radar_fraction=0.5, beamformer="rzf", rho=0.02)
print(f"greedy 20 panels (10 active), RZF rho 0.02, {n} scenes; same sites, different role rule")
print(f"{'role rule':34s} {'radar dB (mean)':>15s} {'worst SINR dB':>14s} {'radar short':>11s} {'comm short':>10s} {'loss':>8s} {'feasible':>8s}")
cases = [(f"fixed_order        share {sh}", base.copy(radar_share=sh, policy="fixed_order")) for sh in (0.25, 0.5, 0.75)]
cases += [(f"advantage bal {bl} share {sh}", base.copy(radar_share=sh, balance=bl)) for sh in (0.25, 0.5, 0.75) for bl in (0.0, 0.5, 1.0)]
for label, d in cases:
    draws = ev.evaluate_design(d, scenes)
    r = ls.total_loss(d, draws)
    t = r["terms"]
    print(f"{label:34s} {np.nanmean([x['radar_db'] for x in draws]):15.1f} {np.nanmean([x['sinr_db'] for x in draws]):14.1f} {t['sensing']:11.1f} {t['comm']:10.1f} {r['loss']:8.1f} {str(r['feasible']):>8s}")
