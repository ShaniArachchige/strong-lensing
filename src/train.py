"""
Train the BayesFlow amortized posterior on a precomputed lens dataset.
"""

import argparse
import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import h5py
import numpy as np
import bayesflow as bf

from model import build_approximator


def load_dataset(path, val_fraction=0.1, seed=0):
    with h5py.File(path, "r") as f:
        images = f["images"][:].astype(np.float32)
        theta = f["theta"][:].astype(np.float32)
        param_names = list(f.attrs["param_names"])

    images = images[..., None]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    n_val = max(1, int(len(images) * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_sims = {"images": images[train_idx], "parameters": theta[train_idx]}
    val_sims = {"images": images[val_idx], "parameters": theta[val_idx]}
    return train_sims, val_sims, param_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="../data/smoke_test.h5")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--summary-dim", type=int, default=64)
    p.add_argument("--coupling-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--out", type=str, default="../models/lens_approximator.keras")
    args = p.parse_args()

    train_sims, val_sims, param_names = load_dataset(args.data)
    print(f"{len(train_sims['images'])} train / {len(val_sims['images'])} val images, "
          f"{len(param_names)} parameters: {param_names}")

    approximator = build_approximator(
        summary_dim=args.summary_dim,
        coupling_depth=args.coupling_depth,
        learning_rate=args.learning_rate,
    )
    adapter = approximator.adapter

    train_data = bf.OfflineDataset(train_sims, batch_size=args.batch_size, adapter=adapter)
    val_data = bf.OfflineDataset(val_sims, batch_size=args.batch_size, adapter=adapter)

    history = approximator.fit(
        dataset=train_data,
        validation_data=val_data,
        epochs=args.epochs,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    approximator.save(args.out)
    print(f"Saved trained approximator to {args.out}")


if __name__ == "__main__":
    main()
