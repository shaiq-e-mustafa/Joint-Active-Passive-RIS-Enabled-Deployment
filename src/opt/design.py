"""The object an optimizer manipulates, plus baseline designs to compare against.

A Design fixes everything that is chosen at deployment or configuration time:
  sites     (N, 2)  panel positions on the ground plan (heights and orientations follow from the scenario rules)
  active    (N,)    bool, panel has amplifiers (gains then follow the power-budget rule of src/opt/active_gains.py)
  radar_share       fraction of the panels that serve radar in a scene (the rest serve comm); WHICH panels is decided per scene by `policy`
  policy            'advantage' (default): the panels with the largest 2 log(radar benefit) - balance * log(comm benefit) in this scene serve radar
                    'fixed_order': the first round(radar_share*N) panels in list order (the old static split; reference/baseline only)
  balance           weight of the comm benefit in that score (0 = rank by radar benefit alone, 1 = comm counts as much as its dB scaling implies)
  nx, ny            UPA size of every panel
  beamformer        'rzf' | 'wmmse' | 'joint'  (joint = shared SDP design, src/opt/joint_beamforming.py)
  rho               share of BS power on a dedicated matched sensing beam (split designs only; 'joint' ignores it)
Scene quantities (users, targets, blockage, fading) are NOT part of the design: they are drawn by src/opt/evaluate.py.
"""
from dataclasses import dataclass, field, replace
import numpy as np
from src.channel.geometry3d import cfg
from src.sys import scenario as sc
from src.opt import placement3d as pl
from src.opt import power_budget as pb
from src.opt import active_gains as ag


def opt_cfg(name, default=None):
    from src.utils.config import settings
    return getattr(getattr(settings.config, "optimizer", None), name, default)


@dataclass
class Design:
    sites: np.ndarray
    active: np.ndarray
    radar_share: float = 0.5
    nx: int = 16
    ny: int = 16
    beamformer: str = "rzf"
    rho: float = 0.0
    policy: str = "advantage"
    balance: float = 1.0

    def __post_init__(self):
        self.sites = np.asarray(self.sites, float).reshape(-1, 2)
        n = len(self.sites)
        self.active = np.asarray(self.active, bool).reshape(n)
        self.radar_share = float(np.clip(self.radar_share, 0.0, 1.0))

    @property
    def n_panels(self):
        return len(self.sites)

    @property
    def n_active(self):
        return int(self.active.sum())

    def copy(self, **kw):
        return replace(self, sites=self.sites.copy(), active=self.active.copy(), **kw)

    def nominal_power_w(self):
        """Deterministic design power (W): element circuits plus DC power of active panels; the amplifier output (about 20% on top
        for 16x16 panels) is added at evaluation time from the actual illumination."""
        L = self.nx * self.ny
        return float(self.n_panels * ag.panel_power_w(L, False) + self.n_active * (ag.panel_power_w(L, True) - ag.panel_power_w(L, False)))

    def vector(self):
        """Flat parameter vector (for gradient-free optimizers / learners): [x, y per site | active | radar_share, balance]."""
        return np.concatenate([self.sites.ravel(), self.active.astype(float), [self.radar_share, self.balance]])

    @staticmethod
    def from_vector(v, n, template):
        v = np.asarray(v, float)
        return Design(v[:2 * n].reshape(n, 2), v[2 * n:3 * n] > 0.5, float(v[3 * n]),
                      template.nx, template.ny, template.beamformer, template.rho, template.policy, float(v[3 * n + 1]))


# ------------------------------------------------------------------------------------------------ baselines


def random_design(n_panels, nx=16, ny=16, active_fraction=0.0, radar_fraction=0.5, beamformer="rzf", rho=0.0, seed=0):
    """Panels uniform in the 8-30 m annulus around the zone (the Entry 12/13 'random' siting)."""
    rng = np.random.default_rng(seed)
    zc = sc.ZONE_CENTER
    r = np.sqrt(8.0 ** 2 + (30.0 ** 2 - 8.0 ** 2) * rng.uniform(0, 1, n_panels))
    a = rng.uniform(0, 2 * np.pi, n_panels)
    sites = np.stack([zc[0] + r * np.cos(a), zc[1] + r * np.sin(a)], axis=1)
    act = np.zeros(n_panels, bool)
    act[rng.permutation(n_panels)[: int(round(active_fraction * n_panels))]] = True
    return Design(sites, act, radar_fraction, nx, ny, beamformer, rho)


def _planning_inputs(n_candidates, n_design, seed):
    cands = pl.annulus_candidates(n_candidates)
    design_pts = sc.disk_sample(sc.ZONE_CENTER, sc.ZONE_RADIUS, n_design, np.random.default_rng(seed))
    return cands, design_pts


def greedy_design(n_panels, nx=16, ny=16, active_fraction=0.0, radar_fraction=0.5, beamformer="rzf", rho=0.0, n_candidates=150, n_design=10, seed=0):
    """Greedy coverage placement (src/opt/placement3d.py), nested: the first n_panels sites of the greedy order."""
    cands, dp = _planning_inputs(n_candidates, n_design, seed)
    order = pl.greedy_order(cands, dp, nx, ny, n_panels, seed=seed)
    sites = np.array([cands[i] for i in order])
    act = np.zeros(n_panels, bool)
    act[: int(round(active_fraction * n_panels))] = True            # amplifiers on the best sites
    return Design(sites, act, radar_fraction, nx, ny, beamformer, rho)


def budget_design(budget_w=None, nx=16, ny=16, radar_fraction=0.5, beamformer="rzf", rho=0.0, n_candidates=150, n_design=10, seed=0, max_panels=None):
    """Budget-aware planner (src/opt/power_budget.py): which sites, and which of them active, under the network power budget."""
    B = pb.budget_w() if budget_w is None else budget_w
    cands, dp = _planning_inputs(n_candidates, n_design, seed)
    chosen, used = pb.plan_under_budget(cands, dp, nx, ny, B, max_panels=int(max_panels or opt_cfg("max_panels", 100)), seed=seed)
    sites = np.array([cands[c] for c, _ in chosen]).reshape(-1, 2)
    act = np.array([a for _, a in chosen], bool)
    return Design(sites, act, radar_fraction, nx, ny, beamformer, rho)
