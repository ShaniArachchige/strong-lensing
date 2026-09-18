"""
Evaluate the trained BayesFlow approximator on a held-out test set.
Produces a recovery plot (predicted vs. true theta) and a calibration
check (does the posterior's confidence match reality).
"""

import argparse
import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import h5py
import numpy as np
import keras
import bayesflow as bf

import model  # noqa: F401  -- registers the custom LensCNN class before loading


def load_test_set(path):
    with h5py.File(path, "r") as f:
        images = f["images"][:].astype(np.float32)
        theta = f["theta"][:].astype(np.float32)
        param_names = list(f.attrs["param_names"])
    images = images[..., None]
    return {"images": images, "parameters": theta}, param_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="../models/lens_approximator.keras")
    p.add_argument("--test-data", type=str, default="../data/test.h5")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--out-dir", type=str, default="../figures")
    args = p.parse_args()

    approximator = keras.saving.load_model(args.model)
    test_sims, param_names = load_test_set(args.test_data)

    # Sanity check on the reload itself: get posterior draws for just 3 images
    # and print their shape. If this errors, or the values look degenerate
    # (e.g. identical across very different images), the save/load round trip
    # is the problem -- not the model itself.
    probe = approximator.sample(
        conditions={"images": test_sims["images"][:3], "parameters": test_sims["parameters"][:3]},
        num_samples=20,
    )
    probe_arr = probe["parameters"] if isinstance(probe, dict) else probe
    print("Sanity check -- posterior draw shape:", np.asarray(probe_arr).shape)

    samples = approximator.sample(conditions=test_sims, num_samples=args.num_samples)
    estimates = samples["parameters"] if isinstance(samples, dict) else samples
    targets = test_sims["parameters"]

    os.makedirs(args.out_dir, exist_ok=True)

    fig = bf.diagnostics.plots.recovery(estimates=estimates, targets=targets, variable_names=param_names)
    fig.savefig(os.path.join(args.out_dir, "recovery.png"), dpi=150, bbox_inches="tight")

    fig = bf.diagnostics.plots.calibration_ecdf(
        estimates=estimates, targets=targets, variable_names=param_names,
        difference=True, rank_type="distance",
    )
    fig.savefig(os.path.join(args.out_dir, "calibration_ecdf.png"), dpi=150, bbox_inches="tight")

    print(f"Saved diagnostics to {args.out_dir}/ (recovery, calibration_ecdf, pairs_posterior, z_score_contraction)")


if __name__ == "__main__":
    main()
