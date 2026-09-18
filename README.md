# Amortized Bayesian Inference for Strong Gravitational Lensing

Simulation-based inference (SBI) pipeline that recovers the physical parameters of a galaxy-galaxy strong gravitational lens — Einstein radius, ellipticity, external shear, source position/size, and external convergence — from a **single noisy lensed image**, using a neural posterior estimator trained entirely on simulations.

Built as a course project for the Simulation-Based Inference course at TU Dortmund.

## Why this problem is hard

Classical strong-lens modeling fits one lens at a time with MCMC or nested sampling, which is accurate but slow — minutes to hours per lens. With upcoming surveys (Euclid, LSST) expected to discover on the order of 10^5 strong lenses, that doesn't scale. Amortized SBI flips the cost structure: pay the simulation + training cost once, then get a full posterior over lens parameters for any new image in milliseconds.

## Approach

1. **Simulator** ([`src/simulator.py`](src/simulator.py)) — wraps [`lenstronomy`](https://github.com/lenstronomy/lenstronomy) to render a galaxy-galaxy lens (SIE + external shear + convergence) lensing a Sérsic source, with realistic Poisson + background noise. Source brightness is automatically rescaled per-draw so the simulated peak SNR is spread uniformly across a target range (10-30) rather than collapsing to a single fixed value.
2. **Prior** ([`src/priors.py`](src/priors.py)) — samples the 9-parameter vector `theta = [theta_E, e1, e2, gamma1, gamma2, xs, ys, Rs, kappa]`. Ellipticity and shear are sampled in (modulus, angle) polar form and converted to Cartesian components, which avoids over-weighting high-ellipticity/high-shear corners that a naive uniform box prior would produce.
3. **Dataset generation** ([`src/generate_dataset.py`](src/generate_dataset.py)) — simulates (theta, image) pairs in parallel across CPU cores and discards degenerate, low-SNR draws. The final runs used 80k train / 10k val / 10k test images per model variant.
4. **Amortized posterior** ([`src/model.py`](src/model.py)) — a custom CNN summary network (6 conv layers → global average pooling → dense) compresses each 64x64 image to a summary vector, which conditions a [BayesFlow](https://github.com/bayesflow-org/bayesflow) `CouplingFlow` normalizing-flow inference network. Trained with a Keras 3 / PyTorch backend.
5. **Training** ([`src/train.py`](src/train.py)) — trains the approximator end-to-end on the offline simulated dataset; final runs used up to 120 epochs with early stopping on GPU (Colab/Kaggle T4).
6. **Diagnostics** ([`src/evaluate.py`](src/evaluate.py)) — on a held-out test set, produces recovery plots (posterior mean vs. true parameter) and simulation-based calibration (ECDF) plots to check the posterior isn't over- or under-confident.

An ablation compares the full 9-parameter model against a simplified 8-parameter model with external convergence fixed to zero (`kappa = 0`), motivated by the mass-sheet degeneracy between `theta_E` and `kappa`.

## Results

- **Well-recovered / well-calibrated:** `theta_E` (r = 0.97) and `Rs` (r = 0.98) are recovered with high accuracy and stay within the 95% calibration band in the full model.
- **Harder parameters:** source position (`xs`, `ys`) and shear/ellipticity components recover poorly in the full model and show posterior overconfidence — but recover dramatically better once `kappa` is fixed out (e.g. `xs`: r = 0.23 → 0.97), consistent with the mass-sheet degeneracy soaking up information that would otherwise pin down the source and shear.
- **The tradeoff:** removing `kappa` improves almost every parameter's point-estimate accuracy, but makes `theta_E` and `Rs` calibration *worse* (overconfident) despite the accuracy gain — accuracy and calibration don't move together here, which is the main empirical finding of the ablation.

See `figures/` for the full recovery and calibration diagnostics (`no_kappa/` holds the ablation's versions).

## Tech stack

`lenstronomy` (simulator) · `BayesFlow` (amortized SBI / normalizing flows) · `Keras 3` on a `PyTorch` backend · `NumPy` / `h5py` · `joblib` (parallel simulation) · trained on Google Colab / Kaggle (T4 GPU)

## Repo structure

```
src/          simulator, priors, model, train, evaluate, dataset generation
notebooks/    exploratory + final end-to-end training notebook
figures/      recovery and calibration diagnostic plots (full model + no-kappa ablation)
models/       trained approximators (.keras) -- gitignored, see note below
data/         simulated train/val/test sets (.h5) -- gitignored, see note below
```

`data/` and `models/` are gitignored since the .h5 datasets exceed GitHub's file size limits. Regenerate data with:

```bash
python src/generate_dataset.py --n 80000 --out data/train.h5 --include-kappa
```

or download the pre-generated data/trained models from: **[add your Google Drive link here]**

## Running it

```bash
pip install -r requirements.txt

# 1. generate a dataset
python src/generate_dataset.py --n 80000 --out data/train.h5 --include-kappa

# 2. train the approximator
python src/train.py --data data/train.h5 --epochs 100 --out models/lens_approximator.keras

# 3. evaluate on a held-out test set
python src/evaluate.py --model models/lens_approximator.keras --test-data data/test.h5
```

## Author

Shani Fernando — [add your LinkedIn / portfolio link here]
