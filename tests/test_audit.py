"""Independent audit of the multi-RIS ISAC model (RF-sensing review).
Run: python tests/test_audit.py   -> prints [PASS]/[FAIL] for verification checks and [FINDING] for demonstrated issues.
Each block states what is being checked and why."""
import sys, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "out"))
import numpy as np
from scipy.stats import ncx2
from scipy.optimize import brentq
from src.utils.config import settings
settings.load_config()
from src.utils.channel_utils import to_linear, to_linear_dbm, to_db, q_function
from src.channel.channel_model import wavelength
from src.channel import risConfig
from src.channel.geometry3d import cfg, to3, link_3d, bs_array, point_end, ris_end, spacing
from src.sys import scenario as sc
from src.sys.system import ISACSystem, TargetLink
from src.sys.factory import build_system
from src.opt.phase_alignment import coherent_phases_upa
import exp12_lib as X

ok = True
LAM = wavelength(); K0 = 2 * np.pi / LAM
P = to_linear_dbm(settings.config.channel_model.P_max)
N0 = to_linear(int(settings.config.channel_model.reciever_nosie) - 30)
M = settings.config.channel_model.M


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def finding(name, detail):
    print(f"[FINDING] {name}: {detail}")


def section(t):
    print(f"\n=== {t} ===")


# ------------------------------------------------------------------------------------------------ A1 detection thresholds
section("A1 detection threshold (Albersheim vs exact, Swerling I)")
pfa, pd = 1e-6, 0.9
A = np.log(0.62 / pfa); B = np.log(pd / (1 - pd))
alb = (6.2 + 4.54 / np.sqrt(1.44)) * np.log10(A + 0.12 * A * B + 1.7 * B)
thr = -2 * np.log(pfa)                                         # square-law detector threshold on |z|^2/(sigma^2/2)
exact_sw0 = 10 * np.log10(brentq(lambda s: ncx2.sf(thr, 2, 2 * s) - pd, 1e-3, 1e4))
exact_sw1 = 10 * np.log10(np.log(pfa) / np.log(pd) - 1)        # Pd = Pfa^(1/(1+SNR))
check("Albersheim matches exact non-fluctuating (Marcum Q) within 0.3 dB", abs(alb - exact_sw0) < 0.3,
      f"(Albersheim {alb:.2f} dB, exact Swerling-0 {exact_sw0:.2f} dB)")
rng = np.random.default_rng(0)
snr_lin = 10 ** (exact_sw1 / 10)
z = np.sqrt(snr_lin / 2) * (rng.standard_normal(400000) + 1j * rng.standard_normal(400000)) \
    + np.sqrt(0.5) * (rng.standard_normal(400000) + 1j * rng.standard_normal(400000))
pd_mc = np.mean(np.abs(z) ** 2 > thr / 2)
check("Swerling-I closed form Pd = Pfa^(1/(1+SNR)) verified by Monte Carlo", abs(pd_mc - pd) < 0.005, f"(MC Pd = {pd_mc:.4f} at {exact_sw1:.2f} dB)")
finding("threshold used is for a NON-fluctuating target",
        f"the model declares Swerling I (alpha ~ CN), which needs {exact_sw1:.2f} dB for Pd=0.9/Pfa=1e-6 single pulse, "
        f"not {alb:.2f} dB: every reported margin is {exact_sw1 - alb:.1f} dB optimistic")

# ------------------------------------------------------------------------------------------------ A2 closed-form vs Monte Carlo radar SNR
section("A2 closed-form radar SNR vs Monte Carlo (resolves the long-standing 18 dB 'mismatch')")
settings.config.channel_model.wavefront = "exact"
s = build_system(8, 4, 1, np.array([True] * 4 + [False] * 4), settings.config.channel_model.L, M, P,
                 np.random.default_rng(3))
s.build_hbar(); s.build_w_mrt(); s.build_T(); s.build_J()
t = s.targets[0]; Rx = s.get_Rx(); Jinv = np.linalg.inv(t.J); Lc = np.linalg.cholesky(t.J)
cf = t.rcs * np.trace(t.T @ Rx @ t.T.conj().T @ Jinv).real
W = np.hstack([u.w for u in s.users])
rng = np.random.default_rng(1)
n_mc = 20000
sig_w = noi_w = sig_raw = noi_raw = 0.0
for _ in range(n_mc):
    sym = (rng.choice([-1, 1], (4, 1)) + 1j * rng.choice([-1, 1], (4, 1))) / np.sqrt(2)
    x = W @ sym
    a = np.sqrt(t.rcs / 2) * (rng.standard_normal() + 1j * rng.standard_normal())
    ys = a * (t.T @ x)
    n = Lc @ ((rng.standard_normal((M, 1)) + 1j * rng.standard_normal((M, 1))) / np.sqrt(2))
    u = Jinv @ (t.T @ x); u = u / np.sqrt((u.conj().T @ t.J @ u).real)          # whitened MF, unit noise power
    sig_w += abs((u.conj().T @ ys).item()) ** 2; noi_w += abs((u.conj().T @ n).item()) ** 2
    sig_raw += np.linalg.norm(ys) ** 2; noi_raw += np.linalg.norm(n) ** 2
emp_w = to_db(sig_w / noi_w); emp_raw = to_db(sig_raw / noi_raw)
check("whitened matched-filter Monte Carlo == sigma^2 tr(T Rx T^H J^-1)", abs(emp_w - to_db(cf)) < 0.25,
      f"(MC {emp_w:.2f} dB vs closed form {to_db(cf):.2f} dB)")
finding("the 18 dB mismatch in tests/test_radar_snr.py is a metric error, not a model error",
        f"radar_validation.validate_radar_snr_montecarlo divides raw echo energy by raw noise energy summed over the "
        f"array (no receive combining): {emp_raw:.2f} dB; closed form - raw = {to_db(cf) - emp_raw:.2f} dB vs 10log10(M) = {10*np.log10(M):.2f} dB")

# ------------------------------------------------------------------------------------------------ A3 noise covariance J
section("A3 interference-plus-noise covariance J (fixed after the audit): independent reference + generator Monte Carlo")
from src.sys.convention import echo_vec, comm_vec
cmod = settings.config.channel_model
saved_noise = cmod.active_ris_noise
cmod.active_ris_noise = -40           # make amplifier noise visible above thermal
s = build_system(6, 2, 1, np.array([True, True, True, False, False, False]), cmod.L, M, P, np.random.default_rng(4))
s.cross_panel_T = True
s.build_hbar(); s.build_w_mrt(); s.build_T(); s.build_J()
t = s.targets[0]
sv = to_linear(cmod.active_ris_noise - 30); sr = N0
# independent reference: ACTIVE panels only, scalar sum |phi|^2 |b|^2, total echo vector u = sum_i G_i^T Phi_i b_i (reciprocal)
u_tot = sum(p.channels.G.T @ (p.state.phi_vec[:, None] * p.channels.b_by_target[0]) for p in s.panels if p.state.a)
J_ref = sr * np.eye(M, dtype=complex); c_tot = 0.0
for p in s.panels:
    if p.state.a == 0 or not p.state.active:
        continue
    G, phi, b = p.channels.G, p.state.phi_vec, p.channels.b_by_target[0]
    J_ref += sv * (G.T * (np.abs(phi) ** 2)[None, :]) @ G.conj()
    c_tot += float(np.sum((np.abs(phi) ** 2)[:, None] * np.abs(b) ** 2))
J_ref += t.rcs * sv * c_tot * (u_tot @ u_tot.conj().T)
check("build_J == independent reference (active panels only, correct scalar)", np.allclose(t.J, J_ref, rtol=1e-9, atol=0),
      f"(rel diff {np.linalg.norm(t.J - J_ref) / np.linalg.norm(J_ref):.1e})")
J_pass = sum(sv * (p.channels.G.T @ p.channels.G.conj()) for p in s.panels if p.state.a and not p.state.active)
check("passive panels contribute no amplifier noise to J", np.linalg.norm(t.J - J_ref) < 1e-9 * np.linalg.norm(J_ref) and np.linalg.norm(J_pass) > 0)
reps = 6000
acc = np.zeros((M, M), dtype=complex)
rng = np.random.default_rng(7)
for _ in range(reps):
    for p in s.panels:
        if p.state.active:
            L_ = p.state.phases.shape[0]
            p.state.noise = np.sqrt(sv / 2) * (rng.standard_normal(L_) + 1j * rng.standard_normal(L_))
            p.state.noise_2 = np.sqrt(sv / 2) * (rng.standard_normal(L_) + 1j * rng.standard_normal(L_))
    y = s.build_y_r(np.zeros((M, 1), complex), rng)[0]
    acc += y @ y.conj().T
rel = np.linalg.norm(acc / reps - t.J) / np.linalg.norm(t.J)
check("noise generator build_y_r covariance == J (Monte Carlo)", rel < 0.08, f"(rel diff {rel:.3f})")
from src.sys.convention import echo_operator
p0 = next(p for p in s.panels if p.state.a)
Gm, Ph, bb = p0.channels.G, np.diag(p0.state.phi_vec), p0.channels.b_by_target[0]
T_left = Gm.conj().T @ Ph @ bb @ bb.conj().T @ Ph @ Gm
T_right = Gm.conj().T @ (Ph @ bb) @ (Ph @ bb).conj().T @ Gm
finding("Clarifications.tex corrected-T equality is false (draft only; code no longer uses either form)",
        f"||left - right||/||right|| = {np.linalg.norm(T_left - T_right) / np.linalg.norm(T_right):.2f} because (Phi b)^H = b^H Phi^H, not b^H Phi. "
        f"The code now uses the reciprocal round trip G^T Phi b b^T Phi G")
cmod.active_ris_noise = saved_noise

# ------------------------------------------------------------------------------------------------ A4 reciprocity
section("A4 reciprocity (fixed after the audit): one RIS phase set serves radar and comm for a co-located user/target")
from src.sys.convention import comm_vec
orig = sc.get_link_params
sc.get_link_params = lambda a, b, rng: (1e12, 2.5, False)
_kap = getattr(cmod, "kappa_los_db", 5.0); cmod.kappa_los_db = 120.0
users, targets = sc.make_users_targets(1, 1, np.random.default_rng(11), direct_mode="none")
users[0].pos = targets[0].pos.copy()
pan = sc.make_panel(0, np.array([96.0, 28.0]), targets, users, 16, 16, False, np.random.default_rng(12))
sc.get_link_params = orig; cmod.kappa_los_db = _kap
G = pan.channels.G; b = pan.channels.b_by_target[0]; f = pan.channels.f_by_user[0]
ph_r = coherent_phases_upa(pan, targets[0].pos, "radar"); ph_c = coherent_phases_upa(pan, users[0].pos, "comm")
rad = lambda ph: np.linalg.norm(echo_vec(G, np.exp(1j * ph), b)) ** 2
com = lambda ph: np.linalg.norm(comm_vec(G, np.exp(1j * ph), f)) ** 2
check("co-located user/target: comm penalty with radar phases", abs(to_db(com(ph_r) / com(ph_c))) < 0.01, f"({to_db(com(ph_r) / com(ph_c)):.3f} dB; was -29 dB with the Hermitian convention)")
check("co-located user/target: radar penalty with comm phases", abs(to_db(rad(ph_c) / rad(ph_r))) < 0.01, f"({to_db(rad(ph_c) / rad(ph_r)):.3f} dB)")

# ------------------------------------------------------------------------------------------------ A5 energy conservation of the point-target model
section("A5 energy sanity: power intercepted by the target vs transmitted power")
pos = X.scene_positions("greedy", 50, np.random.default_rng(0), X.planned_greedy(16, 16, 50))
for nx in (16, 32):
    users, tg = sc.make_users_targets(8, 1, np.random.default_rng(1000), direct_mode="nlos")
    scn = X.make_scene(50, nx, nx, pos, users, tg, 1.0, np.random.default_rng(50000))
    t = scn.targets[0]
    gs = [X.echo_vec(p, coherent_phases_upa(p, t.pos, "radar"), t) for p in scn.panels if p.state.a]
    g = X._base(t) + sum(gs)
    frac = t.rcs * np.linalg.norm(g) ** 2                    # = intercepted power / transmitted power (matched beam)
    snr = to_db(X.snr_from_vec(g, t.J, t.rcs))
    echo_dbm = 10 * np.log10(P * frac * np.linalg.norm(g) ** 2 / t.rcs * t.rcs) + 30 if False else snr + 10 * np.log10(N0) + 30
    finding(f"UPA {nx}x{nx}, 50 greedy panels", f"SNR {snr:.1f} dB, target intercepts {100 * frac:.3g}% of transmit power, "
            f"echo power {echo_dbm:.1f} dBm; "
            + ("physically plausible but the point-scatterer/plane-wave RCS model is stretched (focal spot vs target size)" if frac < 1
               else "IMPOSSIBLE (>100%): point-target RCS model invalid in this regime"))

# ------------------------------------------------------------------------------------------------ A6 element-gain consistency
section("A6 RIS unit-cell gain: G = 8 (Tang's cos^3 cell) on a lambda/2 x lambda/2 cell")
ap = 4 * np.pi * spacing() ** 2 / LAM ** 2
finding("super-directive cell", f"aperture limit of a {spacing():.2f} m square cell is 4*pi*A/lambda^2 = {ap:.2f}; G = 8 implies an "
        f"effective area {8 * LAM ** 2 / (4 * np.pi):.3f} m^2 > physical {spacing() ** 2:.3f} m^2 (efficiency {800 / ap:.0f}%). "
        f"Per bounce this overstates gain by {10 * np.log10(8 / ap):.2f} dB; the radar round trip has two bounces -> "
        f"{20 * np.log10(8 / ap):.2f} dB optimistic. Tang's formula is reproduced, but its G was for a larger measured cell.")
users, tg = sc.make_users_targets(8, 1, np.random.default_rng(1001), direct_mode="nlos")
res = {}
for G_cell in (8.0, ap):
    cmod.ris_cell_gain = G_cell
    scn = X.make_scene(40, 16, 16, X.scene_positions("random", 40, np.random.default_rng(3)), users, tg, 0.0, np.random.default_rng(4))
    res[G_cell] = X.radar_timemux(scn, 0, "intra")
cmod.ris_cell_gain = 8.0
check("numerical check of the per-round-trip delta", abs((res[8.0] - res[ap]) - 20 * np.log10(8 / ap)) < 0.3,
      f"(SNR {res[8.0]:.2f} dB with G=8 vs {res[ap]:.2f} dB aperture-limited: {res[8.0] - res[ap]:.2f} dB)")

# ------------------------------------------------------------------------------------------------ A7 path-loss exponent on deterministic rays
section("A7 path-loss exponent 2.5 on every element-level LoS ray")
_link = sc.link_3d
def link_eta2(rx, rxe, tx, txe, kappa, eta, rng):
    return _link(rx, rxe, tx, txe, kappa, 2.0 if eta == 2.5 else eta, rng)
users, tg = sc.make_users_targets(8, 1, np.random.default_rng(1002), direct_mode="nlos")
vals = {}
for name, fn in (("eta=2.5", _link), ("eta=2.0", link_eta2)):
    sc.link_3d = fn
    scn = X.make_scene(40, 16, 16, X.scene_positions("random", 40, np.random.default_rng(5)), users, tg, 0.0, np.random.default_rng(6))
    vals[name] = X.radar_timemux(scn, 0, "intra")
sc.link_3d = _link
finding("exponent choice moves the headline by tens of dB",
        f"SNR {vals['eta=2.5']:.1f} dB (eta 2.5) vs {vals['eta=2.0']:.1f} dB (eta 2.0): {vals['eta=2.0'] - vals['eta=2.5']:+.1f} dB. "
        f"A deterministic LoS ray between two elements is free space (exponent 2); 2.5 is a statistical fit for aggregate "
        f"urban path loss and should not multiply into a four-hop coherent sum (3GPP UMi LoS is ~21log10 d, i.e. ~2.1)")

# ------------------------------------------------------------------------------------------------ A8 wideband: delay spread across panels and squint
section("A8 narrowband assumption vs the 80 MHz bandwidth implied by the -90 dBm noise floor")
users, tg = sc.make_users_targets(8, 1, np.random.default_rng(1003), direct_mode="none")
scn = X.make_scene(50, 16, 16, X.planned_greedy(16, 16, 50), users, tg, 0.0, np.random.default_rng(7))
t = scn.targets[0]
bs_el, bs_e = bs_array()
t3 = to3(t.pos, float(cfg("h_target", 1.5)))[None, :]
sel = [p for p in scn.panels if p.state.a]
pre = []
for p in sel:
    end = ris_end(p.normal)
    A_g = np.abs(link_3d(p.elements, end, bs_el, bs_e, None, 2.5, None))
    A_b = np.abs(link_3d(p.elements, end, t3, point_end(), None, 2.5, None))[:, 0]
    d_g = np.linalg.norm(p.elements[:, None, :] - bs_el[None, :, :], axis=2)
    d_b = np.linalg.norm(p.elements - t3, axis=1)
    pre.append((A_g, A_b, d_g, d_b, coherent_phases_upa(p, t.pos, "radar")))
def g_at(freq):
    k = 2 * np.pi * freq / 3e8
    tot = 0
    for A_g, A_b, d_g, d_b, ph in pre:
        Gf = A_g * np.exp(-1j * k * d_g); bf = A_b * np.exp(-1j * k * d_b)
        tot = tot + Gf.conj().T @ (np.exp(1j * ph) * bf)        # phases fixed at the carrier design
    return tot
fc = float(cmod.carrier_frequency)
g0 = g_at(fc); w = g0 / np.linalg.norm(g0)
path = [np.linalg.norm(p.pos - np.zeros(2)) + np.linalg.norm(p.pos - t.pos) for p in sel]
finding("one-way path-length spread across selected panels",
        f"{max(path) - min(path):.1f} m -> echo delays spread over {(max(path) - min(path)) * 2 / 3e8 * 1e9:.0f} ns "
        f"(range resolution at 80 MHz is c/2B = {3e8 / 160e6:.2f} m)")
for B in (1e6, 5e6, 20e6, 80e6):
    fs = fc + np.linspace(-B / 2, B / 2, 41)
    snr_f = [abs(g_at(fq).conj() @ w) ** 2 * np.linalg.norm(g_at(fq)) ** 2 for fq in fs]
    loss = to_db(np.mean(snr_f) / (abs(g0.conj() @ w) ** 2 * np.linalg.norm(g0) ** 2))
    (check if B <= 1e6 else finding)(f"band-averaged SNR loss at B = {B / 1e6:.0f} MHz (fixed RIS phases, fixed beam)",
        (abs(loss) < 0.5) if B <= 1e6 else f"{loss:.1f} dB")
    if B <= 1e6:
        print(f"         ({loss:.2f} dB)")

# ------------------------------------------------------------------------------------------------ A9 geometric design vs CSI-based design
section("A9 position-based phase design vs per-element CSI design (Rician K = 5 dB, blocked links Rayleigh)")
users, tg = sc.make_users_targets(8, 1, np.random.default_rng(1004), direct_mode="nlos")
scn = X.make_scene(40, 16, 16, X.scene_positions("random", 40, np.random.default_rng(8)), users, tg, 0.0, np.random.default_rng(9))
t = scn.targets[0]; Jinv = np.linalg.inv(t.J); base = X._base(t)
sel = [p for p in scn.panels if p.state.a]
geo = [X.echo_vec(p, coherent_phases_upa(p, t.pos, "radar"), t) for p in sel]
csi = []
for p in sel:
    C = p.channels.G.conj() * p.channels.b_by_target[0]          # (L, M): g = C^T e^{j ph}  (== G^H Phi b)
    ph = coherent_phases_upa(p, t.pos, "radar")                  # start from the geometric design
    for _ in range(30):                                          # maximize ||C^T e^{j ph}||: align to the current direction
        v = C.T @ np.exp(1j * ph)
        if np.linalg.norm(v) == 0:
            break
        v = v / np.linalg.norm(v)
        ph = -np.angle(C @ v.conj())
    csi.append(C.T @ np.exp(1j * ph))
s_geo = to_db(X.snr_from_vec(base + sum(np.exp(1j * b_) * x for b_, x in zip(X.offsets(geo, base, Jinv), geo)), t.J, t.rcs))
s_csi = to_db(X.snr_from_vec(base + sum(np.exp(1j * b_) * x for b_, x in zip(X.offsets(csi, base, Jinv), csi)), t.J, t.rcs))
finding("geometric (position-only) design leaves gain on the table", f"{s_geo:.1f} dB vs CSI-based {s_csi:.1f} dB ({s_csi - s_geo:+.1f} dB); "
        f"but per-element CSI for 40 x 256 elements needs a pilot/estimation scheme the project does not model")

# ------------------------------------------------------------------------------------------------ A10 small verifications
section("A10 small verifications")
lam_b = risConfig.calibrate_lambda_b(0.30, 100.0, 5.0)
rng = np.random.default_rng(0)
emp = np.mean([risConfig.is_blocked_line_boolean(100.0, rng, lam_b, 5.0) for _ in range(200000)])
check("Line-Boolean calibration: empirical blockage at d_ref", abs(emp - 0.30) < 0.005, f"({emp:.4f} vs 0.30)")
check("Friis beta_0 at 1 m, 1.5 GHz", abs(float(cmod.beta_0_dB) - (-20 * np.log10(4 * np.pi * 1.5e9 / 3e8))) < 0.01,
      f"(config {cmod.beta_0_dB}, exact {-20 * np.log10(4 * np.pi * 1.5e9 / 3e8):.4f} dB)")
gam = 10 ** (8 / 10); rng = np.random.default_rng(1); nsym = 400000
bits = rng.integers(0, 2, (nsym, 2)); sym = ((1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])) / np.sqrt(2)
r = sym + np.sqrt(1 / (2 * gam)) * (rng.standard_normal(nsym) + 1j * rng.standard_normal(nsym))
ber = np.mean(np.concatenate([(r.real < 0) != (bits[:, 0] == 1), (r.imag < 0) != (bits[:, 1] == 1)]))
finding("QPSK BER formula", f"Monte Carlo BER at SINR 8 dB = {ber:.2e}; Q(sqrt(gamma)) = {q_function(np.sqrt(gam)):.2e} (correct), "
        f"Q(sqrt(2 gamma)) = {q_function(np.sqrt(2 * gam)):.2e} (manuscript Eq. BER_QPSK and system.get_ber_per_user: BPSK formula, 3 dB optimistic)")
check("QPSK MC agrees with Q(sqrt(gamma))", abs(ber / q_function(np.sqrt(gam)) - 1) < 0.05)
users, tg = sc.make_users_targets(8, 1, np.random.default_rng(1005), direct_mode="nlos")
scn = X.make_scene(30, 8, 8, X.scene_positions("random", 30, np.random.default_rng(10)), users, tg, 1.0, np.random.default_rng(11))
check("simultaneous-SINR evaluator reduces to time-multiplexed SNR for one target",
      abs(X.radar_simultaneous(scn)[0] - X.radar_timemux(scn, 0, "intra+cross")) < 0.2,
      f"({X.radar_simultaneous(scn)[0]:.2f} vs {X.radar_timemux(scn, 0, 'intra+cross'):.2f} dB)")
# two-ray ground reflection on the direct LoS radar path (not modelled)
hb, ht = float(cfg("h_bs", 10.0)), float(cfg("h_target", 1.5))
ds = np.linspace(85, 115, 301)
d1 = np.sqrt(ds ** 2 + (hb - ht) ** 2); d2 = np.sqrt(ds ** 2 + (hb + ht) ** 2)
one_way = np.abs(1 - (d1 / d2) * np.exp(-1j * K0 * (d2 - d1))) ** 2      # grazing reflection coefficient ~ -1
finding("ground reflection ignored", f"two-ray one-way gain over 85-115 m swings {to_db(one_way.min()):.1f} to {to_db(one_way.max()):.1f} dB; "
        f"on a round trip that is {2 * to_db(one_way.min()):.1f} to {2 * to_db(one_way.max()):.1f} dB on the direct path (and similar on low panel-target hops)")
finding("monostatic self-interference", f"transmit {settings.config.channel_model.P_max} dBm vs noise {settings.config.channel_model.reciever_nosie} dBm: "
        f"{settings.config.channel_model.P_max - settings.config.channel_model.reciever_nosie} dB of Tx/Rx isolation + cancellation needed; not modelled")

print("\nVERIFICATION CHECKS: " + ("ALL PASS" if ok else "SOME FAILED"))
