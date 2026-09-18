"""
Precompute a training set of (theta, image) pairs.

Usage:
    python generate_dataset.py --n 50000 --out ../data/train.h5 --include-kappa
"""

import argparse
import os
import time

import numpy as np
import h5py
from joblib import Parallel, delayed

from priors import sample_prior, PARAM_NAMES_FULL, PARAM_NAMES_NO_KAPPA
from simulator import LensSimulator


def _simulate_one(theta, sim_kwargs):
    sim = _get_simulator(sim_kwargs)
    return sim.simulate(theta, add_noise=True, return_snr=True)


# one simulator per worker process (ImageModel isn't trivially picklable/
# cheap to rebuild per-call, so cache it in each worker)
_SIM_CACHE = {}


def _get_simulator(sim_kwargs):
    key = tuple(sorted(sim_kwargs.items()))
    if key not in _SIM_CACHE:
        _SIM_CACHE[key] = LensSimulator(**sim_kwargs)
    return _SIM_CACHE[key]


def generate(n, include_kappa, sim_kwargs, n_jobs, seed, min_snr=5.0):
    rng = np.random.default_rng(seed)
    thetas = sample_prior(n, include_kappa=include_kappa, rng=rng)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=5)(
        delayed(_simulate_one)(theta, sim_kwargs) for theta in thetas
    )
    images, peak_snrs = zip(*results)
    images = np.stack(images, axis=0)
    peak_snrs = np.array(peak_snrs)
    print(f"Simulated {n} images in {time.time() - t0:.1f}s")

    # drop degenerate low-SNR images (e.g. source scattered fully outside
    # the caustic with negligible magnification). peak_snrs is the TRUE
    # SNR computed on the clean image inside simulate(), not recomputed
    # from noisy pixels.
    keep = peak_snrs >= min_snr
    print(f"Keeping {keep.sum()}/{n} images above min_snr={min_snr}")

    return thetas[keep], images[keep], peak_snrs[keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--out", type=str, default="../data/train.h5")
    p.add_argument("--include-kappa", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-pix", type=int, default=64)
    p.add_argument("--delta-pix", type=float, default=0.05)
    args = p.parse_args()

    sim_kwargs = dict(
        num_pix=args.num_pix,
        delta_pix=args.delta_pix,
        include_kappa=args.include_kappa,
    )

    thetas, images, snrs = generate(
        args.n, args.include_kappa, sim_kwargs, args.n_jobs, args.seed
    )

    param_names = PARAM_NAMES_FULL if args.include_kappa else PARAM_NAMES_NO_KAPPA

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("theta", data=thetas.astype(np.float32))
        f.create_dataset("images", data=images.astype(np.float32))
        f.create_dataset("snr", data=snrs.astype(np.float32))
        f.attrs["param_names"] = param_names
        f.attrs["num_pix"] = args.num_pix
        f.attrs["delta_pix"] = args.delta_pix

    print(f"Saved to {args.out}: theta {thetas.shape}, images {images.shape}")


if __name__ == "__main__":
    main()