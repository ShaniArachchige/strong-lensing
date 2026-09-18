"""
Prior distributions for the strong lensing SBI project.

theta = [theta_E, e1, e2, gamma1, gamma2, xs, ys, Rs, kappa]

Set INCLUDE_KAPPA = False to reproduce the simplified 8-parameter model.
"""

import numpy as np

INCLUDE_KAPPA = True  # flip to False for the ablation without kappa

PARAM_NAMES_FULL = ["theta_E", "e1", "e2", "gamma1", "gamma2", "xs", "ys", "Rs", "kappa"]
PARAM_NAMES_NO_KAPPA = ["theta_E", "e1", "e2", "gamma1", "gamma2", "xs", "ys", "Rs"]

# Bounds chosen to keep images physically sensible for a 64x64, 0.05"/px grid
# (FOV = 3.2" x 3.2"), and to keep the lens close to the "typical" galaxy-galaxy
# lensing regime (Einstein radii ~0.7-2.2").
BOUNDS = {
    "theta_E":  (0.7, 2.2),     # arcsec
    "e_mod":    (0.0, 0.5),     # ellipticity modulus |e|, sampled then split into e1,e2
    "gamma_ext": (0.03, 0.15),   # external shear modulus
    "xs":       (-0.3, 0.3),    # arcsec, source offset from lens center
    "ys":       (-0.3, 0.3),
    "Rs":       (0.05, 0.35),   # source Sersic half-light radius, arcsec
    "kappa":    (-0.05, 0.18),  # external convergence
}


def sample_prior(n, include_kappa=INCLUDE_KAPPA, rng=None):
    """
    Draw n samples of theta from the prior.

    Ellipticity and shear are sampled in (modulus, angle) space and converted
    to (e1, e2) / (gamma1, gamma2) Cartesian components, which avoids
    over-weighting high-ellipticity / high-shear corners that a naive
    uniform-on-e1,e2 box would produce.

    Returns
    -------
    theta : (n, D) array, D = 9 if include_kappa else 8, order matches
            PARAM_NAMES_FULL / PARAM_NAMES_NO_KAPPA
    """
    rng = np.random.default_rng() if rng is None else rng

    theta_E = rng.uniform(*BOUNDS["theta_E"], size=n)

    e_mod = rng.uniform(*BOUNDS["e_mod"], size=n)
    phi_e = rng.uniform(0, np.pi, size=n)
    e1 = e_mod * np.cos(2 * phi_e)
    e2 = e_mod * np.sin(2 * phi_e)

    gamma_mod = rng.uniform(*BOUNDS["gamma_ext"], size=n)
    phi_g = rng.uniform(0, np.pi, size=n)
    gamma1 = gamma_mod * np.cos(2 * phi_g)
    gamma2 = gamma_mod * np.sin(2 * phi_g)

    # keep the source within roughly the Einstein radius so multiple imaging
    # actually happens for a decent fraction of draws
    xs = rng.uniform(*BOUNDS["xs"], size=n) * (theta_E / BOUNDS["theta_E"][1])
    ys = rng.uniform(*BOUNDS["ys"], size=n) * (theta_E / BOUNDS["theta_E"][1])

    Rs = rng.uniform(*BOUNDS["Rs"], size=n)

    cols = [theta_E, e1, e2, gamma1, gamma2, xs, ys, Rs]

    if include_kappa:
        kappa = rng.uniform(*BOUNDS["kappa"], size=n)
        cols.append(kappa)

    return np.stack(cols, axis=1)


def theta_to_kwargs(theta, include_kappa=INCLUDE_KAPPA):
    """
    Convert a single 1D theta vector into lenstronomy kwargs_lens / kwargs_source
    dicts, filling in the fixed nuisance parameters.
    """
    if include_kappa:
        theta_E, e1, e2, gamma1, gamma2, xs, ys, Rs, kappa = theta
    else:
        theta_E, e1, e2, gamma1, gamma2, xs, ys, Rs = theta
        kappa = 0.0

    kwargs_lens = [
        {"theta_E": theta_E, "e1": e1, "e2": e2, "center_x": 0.0, "center_y": 0.0},  # SIE
        {"gamma1": gamma1, "gamma2": gamma2, "ra_0": 0.0, "dec_0": 0.0},             # SHEAR
        {"kappa": kappa, "ra_0": 0.0, "dec_0": 0.0},                                # CONVERGENCE
    ]

    kwargs_source = [
        {
            "amp": FIXED_SOURCE["amp"],          # placeholder, rescaled per-draw for SNR
            "R_sersic": Rs,
            "n_sersic": FIXED_SOURCE["n_sersic"],
            "e1": FIXED_SOURCE["e1"],
            "e2": FIXED_SOURCE["e2"],
            "center_x": xs,
            "center_y": ys,
        }
    ]
    return kwargs_lens, kwargs_source


# Fixed (non-inferred) nuisance parameters
FIXED_SOURCE = {
    "amp": 10.0,      # starting point; simulator.py rescales this to hit target SNR
    "n_sersic": 1.5,
    "e1": 0.05,
    "e2": 0.02,
}