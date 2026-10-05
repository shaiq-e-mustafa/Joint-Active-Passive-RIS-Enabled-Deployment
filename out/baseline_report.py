"""Dry run of the groundwork: score baseline designs (no optimizer). python out/baseline_report.py [n_draws]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.utils.config import settings
settings.load_config()
from src.opt import design as dz, evaluate as ev, loss as ls
from src.utils.detection import required_snr_db

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
scenes = ev.sample_scenes(n)
oc = settings.config.optimizer
print(f"hard limits: intercept <= {100*oc.validity_max_intercept:.0f}% (p{oc.validity_percentile}), power <= {settings.config.channel_model.ris_network_budget_w} W (p{oc.power_percentile}); "
      f"radar needs {required_snr_db():.1f} + {oc.radar_margin_db} dB, users {oc.sinr_target_db} dB; {n} scenes")
designs = [
    ("random 20, passive, RZF rho=0", dz.random_design(20, active_fraction=0.0, beamformer="rzf", rho=0.0)),
    ("greedy 20, passive, RZF rho=0.02", dz.greedy_design(20, active_fraction=0.0, beamformer="rzf", rho=0.02)),
    ("greedy 20, half active, RZF rho=0.02", dz.greedy_design(20, active_fraction=0.5, beamformer="rzf", rho=0.02)),
    ("greedy 20, all active, RZF rho=0.02", dz.greedy_design(20, active_fraction=1.0, beamformer="rzf", rho=0.02)),
    ("greedy 20, half active, joint SDP", dz.greedy_design(20, active_fraction=0.5, beamformer="joint")),
    ("budget planner (2 W), RZF rho=0.02", dz.budget_design(beamformer="rzf", rho=0.02, max_panels=40)),
]
print(f"{'design':40s} {'panels(act)':>11s} {'feasible':>8s} {'power p95 W':>11s} {'intercept p95':>13s} {'radar shortfall dB':>19s} {'comm shortfall dB':>18s} {'loss':>10s}")
for name, d in designs:
    r = ls.evaluate_and_score(d, scenes)
    v, t = r["violations"], r["terms"]
    print(f"{name:40s} {d.n_panels:5d}({d.n_active:2d}) {str(r['feasible']):>8s} {v['power_p']:11.2f} {100*v['validity_p']:12.2f}% {t['sensing']:19.1f} {t['comm']:18.1f} {r['loss']:10.1f}")
