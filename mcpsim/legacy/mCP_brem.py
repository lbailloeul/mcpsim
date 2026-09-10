// Author : Saeid Foroughi (saeidf@physics.carleton.ca)

import mpi4py.rc
mpi4py.rc.threads = False

import math
import numpy as np
import cmath
import sys
import os
import os.path
import argparse
import time
import vegas

from mpi4py import MPI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASS_FILE = os.path.join(SCRIPT_DIR, "data-backup", "mship_values.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "leo-brem-output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_LAMBDA_VALUES = [1.0, 1.5, 2.0]
DEFAULT_LAMBDA_VALUES = [1.5]
DEFAULT_NITN = 10
DEFAULT_NEVAL = 4000
DEFAULT_GRID_SIZE = 100
DEFAULT_PROGRESS_EVERY = 100

# --- MPI setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================================
# DarkQuest Beam-dump luminosity and mCP event yield for a 5-m iron target
# =============================================================================
#
#  * Proton beam: 120 GeV,  N_POT = 10^20
#  * Target:      solid iron (Fe),  L = 5 m = 500 cm
#  * Goal:        convert production cross-sections (pb) → expected number of χχ̄
#                 pairs that are produced inside the dump.
#
#  Notes
#  In order to avoid dealing with the attenuation of the proton beam we will focus on
#  collisions in the first nuclear collision length of the target
#  -----
#  • ρ, A : density and atomic mass of iron
#  • λ_int: nuclear interaction length of protons in iron
#           PDG tables give  λ_int ≃ 132 g cm⁻²  → 132 / ρ ≈ 16.8 cm
#  • P_int: probability that a beam proton undergoes ≥1 interaction
#           after traversing the 5-m dump  [ P = 1 - exp(-L/λ) ].
#  • luminosity_eff: effective “proton–nucleon luminosity’’ [cm⁻²]
#  • CS_total must already contain total production cross-sections in pb
#    (output of the VEGAS integration).
#
#  The final line gives N_events_MCP_list, one entry for each mass point.
# =============================================================================

# model input parameters
m_p = 0.9383
epsilon = 1.0                  # unit charge
alpha_em = 1.0 / 137.0   # Fine-structure constant

# -------- Conversion / constants --------
pb_to_cm2 = 1.0e-36            # 1 pb = 10⁻³⁶ cm²
mb_to_cm2 = 1.0e-27            # 1 mb = 10⁻²⁷ cm²
NA        = 6.022e23           # Avogadro, mol⁻¹

# -------- Beam & target parameters --------
s = 15.06**2                   # Center-of-mass energy squared [GeV^2]
N_POT    = 1.0e20              # protons on target
L_target = 500.0               # target length [cm]  (5 m iron)
rho_fe   = 7.87                # g cm⁻³, density of iron
A_fe     = 55.845              # g mol⁻¹, atomic mass of iron
z_det    = 40                  # detector location [m] 
bar_size = 0.05                # bar dimension [m]
n_bar    = 20                  # number of bar for a 1m x 1m detector
det_angle = (n_bar * bar_size/2)/z_det           # A dedicated MCP detector located at z = 40 m with an area of 20cm x 20cm and 1m x 1m, for benchmarks.

# proton–iron nuclear interaction length:
lambda_int_gcm2 = 132.0         # g cm⁻²  (PDG)
lambda_int_cm   = lambda_int_gcm2 / rho_fe   # ≈ 16.8 cm

# -------- Effective luminosity --------
# nuclei per cm²  =  ρ L / A × N_A
# Consider production only in the first interaction length
luminosity  = N_POT * NA * (rho_fe * lambda_int_cm / A_fe)   # [1/cm²]

# A angle cut is applied for faster integration convergence.
theta_max_vegas = np.pi/10
Lambda_p = 1.5           # Cutoff scale for off-shell FF [GeV]
s = (15.06)**2                   # Center-of-mass energy squared [GeV^2]
Ebeam = (s / 2.0 - m_p**2) / m_p # Beam energy in lab frame [GeV]

# =============================================================================

def sigma_pA_inel(s, A):
    """
    Inelastic proton-nucleus cross-section minus the target single diffraction (TSD) piece [mb].

    Parameters:
    -----------
    s : float
        Center-of-mass energy squared (GeV^2) [not directly used here].
    A : float
        Atomic mass number of the target nucleus.

    Returns:
    --------
    sigma_inel : float
        Inelastic p-A cross-section in mb.

    Reference:
    ----------
    Based on empirical fit from arXiv:2112.09814
    """
    return 43.55 * A**0.7111 - 3.84 * A**0.35

# Off-shell Proton Form Factor
def F_ppD(t, Lambda_p):
    """
    Off-shell form factor for the intermediate proton line in ISR.

    Parameters:
        t : float or ndarray
            Momentum transfer squared [GeV^2].
        Lambda_p : float
            Off-shell suppression scale [GeV].

    Returns:
        float or ndarray : Form factor value.
    """
    return Lambda_p**4 / (Lambda_p**4 + (t - m_p**2)**2)

#  Vector Meson Dominance (VMD) Proton EM Form Factor
VMD_MESON_MASSES = np.array([0.77, 1.25, 1.45])  # GeV
VMD_GAMMA_RHO = np.array([0.15, 0.3, 0.5])       # GeV
VMD_GAMMA_OMEGA = np.array([0.0085, 0.3, 0.5])   # GeV
VMD_F_RHO = np.array([0.616, 0.223, -0.339])
VMD_F_OMEGA = np.array([1.011, -0.881, 0.369])

FORM_FACTOR_MODES = ("full", "no_vmd", "rho_only", "off_resonance")
form_factor_mode = "off_resonance"
RESONANCE_VETO_N_WIDTHS = 1.0
VMD_RESONANCE_VETO_WIDTHS = np.maximum(VMD_GAMMA_RHO, VMD_GAMMA_OMEGA)


def resolve_form_factor_mode(mode):
    if mode is None:
        mode = form_factor_mode
    if mode not in FORM_FACTOR_MODES:
        raise ValueError(f"Unknown form_factor_mode={mode!r}; expected one of {FORM_FACTOR_MODES}")
    return mode


def output_dir_for_mode(mode):
    mode = resolve_form_factor_mode(mode)
    if mode == "full":
        return OUTPUT_DIR
    return f"{OUTPUT_DIR}-{mode}"


def is_on_vmd_resonance(k2, n_widths=RESONANCE_VETO_N_WIDTHS):
    """
    Return True when the mCP-pair invariant mass is in a VMD pole window.

    The conservative off_resonance mode removes these bins from proton brem so
    explicit meson production can be treated separately without double-counting.
    """
    if k2 <= 0.0:
        return False

    m_pair = np.sqrt(k2)
    for mass, width in zip(VMD_MESON_MASSES, VMD_RESONANCE_VETO_WIDTHS):
        if abs(m_pair - mass) <= n_widths * width:
            return True
    return False


def _F1_proton_VMD_scalar(t, mode):
    mode = resolve_form_factor_mode(mode)

    if mode == "no_vmd":
        return 1.0

    if mode == "off_resonance" and is_on_vmd_resonance(t):
        return 0.0

    if mode == "rho_only":
        denom_rho = VMD_MESON_MASSES[0]**2 - t - 1j * VMD_MESON_MASSES[0] * VMD_GAMMA_RHO[0]
        F1_rho = VMD_F_RHO[0] * VMD_MESON_MASSES[0]**2 / denom_rho
        return np.abs(F1_rho)

    denom_rho = VMD_MESON_MASSES**2 - t - 1j * VMD_MESON_MASSES * VMD_GAMMA_RHO
    denom_omega = VMD_MESON_MASSES**2 - t - 1j * VMD_MESON_MASSES * VMD_GAMMA_OMEGA

    F1_rho = np.sum(VMD_F_RHO * VMD_MESON_MASSES**2 / denom_rho)
    F1_omega = np.sum(VMD_F_OMEGA * VMD_MESON_MASSES**2 / denom_omega)

    return np.abs(F1_rho + F1_omega)


def F1_proton_VMD(t, mode=None):
    """
    Proton form factor switch for the bremsstrahlung calculation.

    Parameters:
        t : float or ndarray
            Momentum transfer squared [GeV^2].
        mode : str
            "full", "no_vmd", "rho_only", or "off_resonance".

    Returns:
        float or ndarray : Absolute value of the selected proton form factor.
    """
    mode = resolve_form_factor_mode(mode)
    if np.ndim(t) == 0:
        return _F1_proton_VMD_scalar(t, mode)
    return np.vectorize(lambda ti: _F1_proton_VMD_scalar(ti, mode))(t)

# =============================================================================

# === Differential Cross Section (ISR, NSD, Lab frame) ===
def Brem_MCP_Differential_CrossSection_NSD_Lab(phi, theta2, E2, theta1, E1, s, eps, m_chi, det_angle, Lambda_p, mode=None): # checked all correct!
    """
    ISR-based mCP production differential cross section in lab frame via NSD pA scattering.
    
    Parameters:
        phi, theta2, E2, theta1, E1 : float
            Final-state kinematic variables of mCPs [rad, GeV].
        s : float
            Center-of-mass energy squared [GeV^2].
        eps : float
            mCP charge fraction (Q = eps * e).
        m_chi : float
            mCP mass [GeV].
        det_angle : float
            Detector angular coverage [rad].
    
    Returns:
        float : Differential cross-section in picobarns [pb].
    """
    mode = resolve_form_factor_mode(mode)

    # Beam energy and momentum
    Eb = s / (2 * m_p) - m_p
    pb = np.sqrt(Eb**2 - m_p**2)
    s_valid = 3.0**2

    # Final-state momenta of mCPs
    p1, p2 = np.sqrt(E1**2 - m_chi**2), np.sqrt(E2**2 - m_chi**2)
    p1T, p2T = p1*np.sin(theta1), p2*np.sin(theta2)
    cos_12 = np.cos(theta1) * np.cos(theta2) + np.cos(phi) * np.sin(theta1) * np.sin(theta2)

    # Lorentz invariants
    S = Eb * E1 - pb * p1 * np.cos(theta1) # pb.px1
    T = Eb * E2 - pb * p2 * np.cos(theta2) # pb.px2
    U = E1 * E2 - p1 * p2 * cos_12         # px1.px2
    k2 = 2 * m_chi**2 + 2 * U              # k2 = (px1+px2)^2

    # intermediate proton momentum; this is on-shell! so Ep_pr is not equal to Eb - E1 - E2; instead  Ep_pr^2 = mp^2 + (pb-px1-px2)^2 
    p_pr = np.sqrt(pb**2 + p1**2 + p2**2 + 2*p1*p2*cos_12 - 2*pb*p1*np.cos(theta1) - 2*pb*p2*np.cos(theta2))
    Ep_pr = np.sqrt(m_p**2 + p_pr**2) # This is correct if momentum is conserved at the splitting vertex but not the energy!
    # Inelastic cross section for target (A)
    s_prime = 2 * m_p * (m_p + Ep_pr)
    P_prime2 = m_p**2 + 2 * m_chi**2 + 2 * (U - S - T) # This is only relavant for the denominator 1/(p-k)^2

    # Dot products for splitting function; p' = p - p1 - p2
    S_pr = Ep_pr * E1 - (pb * p1 * np.cos(theta1) - p1**2 - p1 * p2 * cos_12)           # p'.px1
    T_pr = Ep_pr * E2 - (pb * p2 * np.cos(theta2) - p2**2 - p1 * p2 * cos_12)           # p'.px2
    V    = Ep_pr * Eb - (pb**2 - pb * p1 * np.cos(theta1) - pb * p2 * np.cos(theta2))   # p'.pb

    # The kinematic consistency conditions for the quasi-real approximation
    DelE_abr = Ep_pr + (Eb - (E1 + E2))
    DelE_emt = Ep_pr - (Eb - (E1 + E2))
    qT = np.sqrt(p1T**2 + p2T**2 + 2*p1T*p2T*np.cos(phi))
    Kin_condition = (DelE_emt < 0.1 * DelE_abr) and (qT < (E1 + E2) ) and (m_p < E1 + E2 < Eb)
    Kin_filter = 1.0 if Kin_condition and (s_prime>s_valid) else 0.0

    # Matrix element squared (splitting-like)
    Amplitude = 16 * (4 * np.pi * alpha_em)**2 * eps**2 * (S_pr * T + T_pr * S + m_chi**2 * V - m_p**2 * U - 2 * m_p**2 * m_chi**2) / k2**2
    # Phase space and normalization factor
    XS_factor = (Ep_pr / Eb) * p1**2 * p2**2 * np.sin(theta1) * np.sin(theta2) * 2 * np.pi / (4 * E1 * E2 * (2 * np.pi)**6) / (P_prime2 - m_p**2)**2
    XS_factor = XS_factor * E1/p1 * E2/p2 # to conver into dE1dE2

    FormFactors = F_ppD(P_prime2, Lambda_p) * F1_proton_VMD(k2, mode)

    # Dimension mb to pb
    dim_facttor = 1.0e9

    # Final differential cross-section in picobarns
    CS = Amplitude * (FormFactors**2) * XS_factor * sigma_pA_inel(s_prime, A_fe) * Kin_filter * dim_facttor  
    return CS

# === Integration bounds ===
def bounds_phi(theta2, E2, theta1, E1, s, eps, m_chi, det_angle): return [0.0, 2 * np.pi]
def bounds_theta1(E1, s, eps, m_chi, det_angle): return [0, det_angle]
def bounds_theta2(E2, theta1, E1, s, eps, m_chi, det_angle): return [0.0, 100 * det_angle]
    
def bounds_E1(s, eps, m_chi, det_angle):
    Eb = s / (2 * m_p) - m_p
    pb = np.sqrt(Eb**2 - m_p**2)
    zmin, zmax = 1.0e-3, 0.999
    p_min = m_chi # zmin * pb
    p_max = zmax * pb
    return [np.sqrt(p_min**2 + m_chi**2), np.sqrt(p_max**2 + m_chi**2)]

def bounds_E2(theta1, E1, s, eps, m_chi, det_angle):
    Eb = s / (2 * m_p) - m_p
    pb = np.sqrt(Eb**2 - m_p**2)
    p1 = np.sqrt(E1**2 - m_chi**2)
    zmin, zmax = 1.0e-3, 0.999
    p_min = m_chi # zmin * pb
    p_max = (m_chi + zmax * pb)- p1 # (zmin + zmax) * pb - p1
    return [np.sqrt(p_min**2 + m_chi**2), np.sqrt(p_max**2 + m_chi**2)]

def bounds_vegas_E1(s, eps, m_chi, det_angle):
    Eb = s / (2 * m_p) - m_p
    pb = np.sqrt(Eb**2 - m_p**2)
    zmin, zmax = 0.01, 0.99
    p_min, p_max = zmin * pb, zmax * pb
    return [np.sqrt(p_min**2 + m_chi**2), np.sqrt(p_max**2 + m_chi**2)]

def bounds_vegas_E2(theta1, E1, s, eps, m_chi, det_angle):
    Eb = s / (2 * m_p) - m_p
    pb = np.sqrt(Eb**2 - m_p**2)
    p1 = np.sqrt(E1**2 - m_chi**2)
    zmin, zmax = 0.01, 0.99
    p_min = zmin * pb
    p_max = (zmin + zmax) * pb - p1
    return [np.sqrt(p_min**2 + m_chi**2), np.sqrt(p_max**2 + m_chi**2)]

def map_log(x, xmin, xmax):
    """Map x∈[0,1] → y∈[xmin,xmax] logarithmically; return y and dy/dx."""
    L = np.log(xmax/xmin)
    y = xmin * np.exp(L * x)
    jac = L * y          # dy/dx
    return y, jac

def map_linear(x, ymin, ymax):
    y = ymin + x*(ymax - ymin)
    jac = (ymax - ymin)
    return y, jac

# --- Total cross section with geometry cases ---
def compute_total_cross_section(s, eps, m_chi, det_angle, Lambda_p,
                                geometry="both",
                                nitn=10, neval=1000, alpha=False,
                                mode=None):
    """
    VEGAS integration over (phi, theta2, E2, theta1, E1).
    geometry: "both", "one", "none"
    """

    Eb = s/(2*m_p) - m_p
    pb = np.sqrt(Eb**2 - m_p**2)

    # Define angular ranges
    theta_min = 1e-5           # avoid log(0)
    theta_max = theta_max_vegas   # or >= det_angle

    def integrand(x):
        # x = [x_phi, x_th2, x_E2, x_th1, x_E1]

        # φ
        phi,  jac_phi  = map_linear(x[0], 0.0, 2*np.pi)

        # θ2 (log sampling)
        theta2, jac_th2 = map_log(x[1], theta_min, theta_max)

        # θ1 (log sampling)
        theta1, jac_th1 = map_log(x[3], theta_min, theta_max)

        # Energies (linear, but with dynamic bounds)
        E1_min, E1_max = bounds_vegas_E1(s, eps, m_chi, det_angle)
        E1, jac_E1 = map_linear(x[4], E1_min, E1_max)

        E2_min, E2_max = bounds_vegas_E2(theta1, E1, s, eps, m_chi, det_angle)
        E2, jac_E2 = map_linear(x[2], E2_min, E2_max)

        # Geometry cuts
        in1 = (theta1 < det_angle)
        in2 = (theta2 < det_angle)
        if geometry == "both" and not (in1 and in2): return 0.0
        if geometry == "one"  and not (in1 or  in2): return 0.0
        if geometry == "none": return 0.0

        # Differential CS at physical vars
        dcs = Brem_MCP_Differential_CrossSection_NSD_Lab(
            phi, theta2, E2, theta1, E1, s, eps, m_chi, det_angle, Lambda_p, mode
        )

        # Full Jacobian = product of each 1D mapping
        J = jac_phi * jac_th2 * jac_E2 * jac_th1 * jac_E1

        return dcs * J

    integ = vegas.Integrator([[0,1]]*5)
    integ(integrand, nitn=nitn, neval=neval)            # warm-up
    result = integ(integrand, nitn=nitn, neval=neval, alpha=alpha)

    return result.mean, result.sdev, result.summary()

def accept_event(theta1, theta2, phi, z_det=z_det, bar_size=bar_size, n_bar=n_bar):
    """
    Accepts an event if:
      - one particle hits inside and the other misses, OR
      - both hit the same detector bar.
    Rejects if both miss or both hit different bars.
    """
    half_size = (n_bar * bar_size) / 2.0

    def project(theta, phi):
        # spherical → cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        if z <= 0:
            return None
        scale = z_det / z
        x, y = x * scale, y * scale
        if -half_size <= x <= half_size and -half_size <= y <= half_size:
            return int((y + half_size) // bar_size), int((x + half_size) // bar_size)
        return None

    bar1, bar2 = project(theta1, 0), project(theta2, phi)

    # Single: one inside, one outside (accepted) or Double: both inside but in the same bar (accepted)
    # Determine Number_mCP
    if bar1 is None and bar2 is None:
        Number_mCP = 0  # rejected
    elif bar1 is not None and bar1 == bar2:
        Number_mCP = 2  # both inside same bar
    else:
        Number_mCP = 1  # exactly one inside

    return Number_mCP

def diff_dsigma_dE1dtheta1(E1, theta1, s, eps, m_chi, det_angle, Lambda_p,
                           nitn=10, neval=2000, alpha=False,
                           mode=None):
    """
    Compute dσ/(dE1 dtheta1) [pb / (GeV·rad)] for mCP pair production via ISR
    by Monte Carlo VEGAS integration over (phi, θ2, E2).
    single track scenario θ2> det_angle 
    double track scenario θ2< det_angle 
    Alternative scenario: one inside, one outside (accepted) or Double: both inside but in the same bar (accepted)
    
    Parameters
    ----------
    E1 : float
        Energy of chi_1 in the lab [GeV].
    theta1 : float
        Polar angle of chi_1 in the lab [rad].
    s : float
        Center-of-mass energy squared [GeV²].
    eps : float
        mCP charge fraction (Q = eps * e).
    m_chi : float
        mCP mass [GeV].
    nitn : int
        Number of VEGAS warm-up iterations.
    neval : int
        Number of integrand evaluations per iteration.
    alpha : float
        VEGAS adaptation parameter.

    Returns
    -------
    mean : float
        Estimated value of dσ/dE₁ dθ₁ in picobarns per GeV per rad.
    std  : float
        VEGAS uncertainty on the mean.
    """
    # 1) Precompute beam energy & set up E2 bounds
    Eb = s/(2*m_p) - m_p
    E2_min, E2_max = bounds_E2(theta1, E1, s, eps, m_chi, det_angle)

    # Single track or shared bar
    theta_min = 1e-5    
    theta_max = theta_max_vegas
    
    # Define angular ranges
    # single track scenario
    # theta_min = det_angle    
    # theta_max = theta_max_vegas
    
    # double track scenario
    # theta_min = 1e-5     
    # theta_max = det_angle  

    # 2) Define VEGAS integrand on the unit hypercube [0,1]^3 -> (phi,theta2,E2)
    def vegas_integrand(x):
        # map to physical variables
        phi, jac_phi = map_linear(x[0], 0.0, 2*np.pi)            # x[0] -> phi
        theta2, jac_th2 = map_log(x[1], theta_min, theta_max)    # x[1] -> theta2 (log sampling)
        E2, jac_E2 = map_linear(x[2], E2_min, E2_max)            # x[2] -> E2 (linear)

        # evaluate the 5D differential cross section at fixed (E1,theta1);
        # Accept the event if exactly one particle hits or both hit the same bar. Reject otherwise.
        accepted_N_mcp = accept_event(theta1, theta2, phi)
        dcs = Brem_MCP_Differential_CrossSection_NSD_Lab(phi, theta2, E2, theta1, E1, s, eps, m_chi, det_angle, Lambda_p, mode) * accepted_N_mcp

        # Full Jacobian of the transform:
        #   dphi/dx0 = 2π,
        #   dtheta2/dx1 = theta_max,
        #   dE2/dx2 = E2_max - E2_min
        J = jac_phi * jac_th2 * jac_E2

        return dcs * J

    # 3) Run VEGAS
    integ = vegas.Integrator([[0,1],[0,1],[0,1]])
    integ(vegas_integrand, nitn=nitn, neval=neval)  # warm-up
    result = integ(vegas_integrand, nitn=nitn, neval=neval, alpha=alpha)

    return result.mean, result.sdev

# --- Wrapper for differential distribution calculation ---
def compute_diff_brem_distribution_theta_p(theta, p, s, eps, m_chi, det_angle, Lambda_p, mode=None,
                                           nitn=DEFAULT_NITN, neval=DEFAULT_NEVAL):
    En = np.sqrt(p**2 + m_chi**2)
    mean, _ = diff_dsigma_dE1dtheta1(En, theta, s, eps, m_chi, det_angle, Lambda_p, nitn=nitn, neval=neval, mode=mode)
    return mean * (p/En)

def Distribution_production(m_chi, mode=None, lambda_values=None, nitn=DEFAULT_NITN, neval=DEFAULT_NEVAL,
                            output_dir=None, force=False, grid_size=DEFAULT_GRID_SIZE,
                            progress_every=DEFAULT_PROGRESS_EVERY):
    """
    Compute differential mCP bremsstrahlung distribution using MPI.
    """
    mode = resolve_form_factor_mode(mode)
    if lambda_values is None:
        lambda_values = DEFAULT_LAMBDA_VALUES
    lambda_values = [float(value) for value in lambda_values]
    output_dir = output_dir_for_mode(mode) if output_dir is None else output_dir
    outfile = os.path.join(output_dir, f"Brem_120GeV_{m_chi:g}.txt")

    should_skip = (not force) and os.path.exists(outfile)
    should_skip = comm.bcast(should_skip, root=0)
    if should_skip:
        if rank == 0:
            print(f"Skipping existing {outfile}", flush=True)
        return

    if rank == 0:
        print(
            f"Starting m_chi={m_chi:.5g} GeV with form_factor_mode={mode}; "
            f"lambdas={lambda_values}; nitn={nitn}; neval={neval}; grid={grid_size}x{grid_size}",
            flush=True,
        )
    pbeam = np.sqrt(Ebeam**2 - m_p**2)
    pmax = 0.999 * pbeam
    
    # --- Energy and angle grid ---
    p_edges = np.logspace(np.log10(m_chi), np.log10(pmax), grid_size + 1)
    th_edges = np.logspace(-5.0, np.log10(theta_max_vegas), grid_size + 1)
    p_mid = 0.5 * (p_edges[:-1] + p_edges[1:])
    th_mid = 0.5 * (th_edges[:-1] + th_edges[1:])
    
    # --- Prepare all (theta, p) tasks ---
    tasks = [(th, p) for th in th_mid for p in p_mid] 
    chunk_size = (len(tasks) + size - 1) // size
    # Divide tasks among ranks
    local_tasks = tasks[rank*chunk_size : min((rank+1)*chunk_size, len(tasks))]
    
    # --- Compute local results: return list of vectors [len(lambda_values)] ---
    local_results = []
    for task_index, (th, p) in enumerate(local_tasks, start=1):
        vals = []
        for Lam in lambda_values:
            val = compute_diff_brem_distribution_theta_p(th, p, s, epsilon, m_chi, det_angle, Lam, mode, nitn=nitn, neval=neval)
            vals.append(val)
        local_results.append(vals)
        if rank == 0 and progress_every and task_index % progress_every == 0:
            print(f"rank0 progress for m_chi={m_chi:g}: {task_index}/{len(local_tasks)} local bins", flush=True)

    # --- Gather results on rank 0 ---
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Flatten and reshape: (n_tasks, n_Lambda)
        results = np.array([val for sublist in all_results for val in sublist])
        results = results.reshape(len(th_mid), len(p_mid), len(lambda_values))  # (n_th, n_p, n_Lambda)

        # --- Compute weights using midpoint approximation ---
        # --- Vectorized weights ---
        dth = np.diff(th_edges)    # (n_th,)
        dp  = np.diff(p_edges)     # (n_p,)
        # Broadcasting: (n_th,1,1) × (1,n_p,1) → (n_th,n_p,1)
        weights = results * dth[:, None, None] * dp[None, :, None]
    
        # --- Prepare and save output ---
        PP, TH = np.meshgrid(p_mid, th_mid)   # (n_th, n_p)
        output_weights = np.full((len(th_mid), len(p_mid), len(OUTPUT_LAMBDA_VALUES)), np.nan)
        for source_index, lambda_value in enumerate(lambda_values):
            output_index = OUTPUT_LAMBDA_VALUES.index(lambda_value)
            output_weights[:, :, output_index] = weights[:, :, source_index]

        data = np.column_stack([
            np.log10(TH.ravel()),
            np.log10(PP.ravel()),
            output_weights[:, :, 0].ravel(),
            output_weights[:, :, 1].ravel(),
            output_weights[:, :, 2].ravel()
        ])
        
        header = (
            "This file contains the production cross section and number of mCPs from the proton bremsstrahlung channel with ε = 1, into the forward hemisphere only.\n"
            f"Form factor mode: {mode}\n"
            f"Computed Lambda_p values: {lambda_values}\n"
            f"Uncomputed Lambda_p columns are written as NaN.\n"
            f"Off-resonance veto: |sqrt(k2) - mV| <= {RESONANCE_VETO_N_WIDTHS} width for each VMD pole when mode = off_resonance.\n"
            "Luminosity inputs: nuclei/cm² = (ρ × L / A) × N_A, considering production only in the first interaction length.\n"
            "Single track or shared bar: Accept the event if exactly one particle hits or both hit the same bar. Reject otherwise.\n"
            "E_beam = 120 GeV   # beam energy\n"
            "N_POT   = 1.0e20   # protons on target\n"
            "L_target = 16.8 cm # one nuclear interaction length of protons in iron)\n"
            "ρ_Fe = 7.87 g/cm³  # density of iron\n"
            "A_Fe = 55.845 g/mol   # atomic mass of iron\n"
            "\n"
            f"{'log10(theta)':>14} {'log10(p/GeV)':>25}"
            f"{'cross-section [pb/bin]':>25} {'cross-section [pb/bin]':>25} {'cross-section [pb/bin]':>25}\n"
            f"{'':>40}{'Lambda_p=1.0 GeV':>25}{'Lambda_p=1.5 GeV':>26}{'Lambda_p=2.0 GeV':>26}"
        )
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
        np.savetxt(outfile, data,
                   fmt='%15.8E %25.8E %25.8E %25.8E %25.8E',
                   delimiter='\t\t',
                   header=header
                  )
        print(f"Saved {outfile}", flush=True)

def parse_lambda_values(value):
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in OUTPUT_LAMBDA_VALUES]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported Lambda_p values {invalid}; expected subset of {OUTPUT_LAMBDA_VALUES}")
    return values


def parse_mass_values(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def select_masses(args):
    if args.masses:
        masses = np.array(parse_mass_values(args.masses), dtype=float)
    else:
        mass_list = np.round(np.loadtxt(args.mass_file), 5)
        masses = mass_list[mass_list < args.mass_cutoff]

    if args.mass is not None:
        masses = np.array([args.mass], dtype=float)

    if args.mass_start is not None or args.mass_end is not None:
        start = 0 if args.mass_start is None else args.mass_start
        end = len(masses) if args.mass_end is None else args.mass_end
        masses = masses[start:end]

    if args.mass_index is not None:
        masses = np.array([masses[args.mass_index]], dtype=float)

    return masses


def parse_args():
    parser = argparse.ArgumentParser(description="Compute mCP proton-bremsstrahlung distributions.")
    parser.add_argument("--mass-file", default=MASS_FILE)
    parser.add_argument("--mass-cutoff", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=None, help="Run exactly one mass value.")
    parser.add_argument("--masses", default=None, help="Comma-separated mass values, e.g. 0.02,0.1,0.4.")
    parser.add_argument("--mass-index", type=int, default=None, help="Run one index from the selected mass list.")
    parser.add_argument("--mass-start", type=int, default=None, help="Start index for a mass-list slice.")
    parser.add_argument("--mass-end", type=int, default=None, help="End index for a mass-list slice.")
    parser.add_argument("--form-factor-mode", choices=FORM_FACTOR_MODES, default=form_factor_mode)
    parser.add_argument("--lambdas", type=parse_lambda_values, default=DEFAULT_LAMBDA_VALUES,
                        help="Comma-separated Lambda_p values. Default: 1.5. Use 1.0,1.5,2.0 for full band.")
    parser.add_argument("--nitn", type=int, default=DEFAULT_NITN)
    parser.add_argument("--neval", type=int, default=DEFAULT_NEVAL)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if the output file already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    masses = select_masses(args)
    if rank == 0:
        print(f"Selected {len(masses)} masses: {masses}", flush=True)

    for mass in masses:
        Distribution_production(
            mass,
            args.form_factor_mode,
            lambda_values=args.lambdas,
            nitn=args.nitn,
            neval=args.neval,
            output_dir=args.output_dir,
            force=args.force,
            grid_size=args.grid_size,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
