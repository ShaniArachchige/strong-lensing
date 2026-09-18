from priors import sample_prior
from simulator import LensSimulator
import matplotlib.pyplot as plt

sim = LensSimulator(include_kappa=True)
thetas = sample_prior(6, include_kappa=True)
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for theta, ax in zip(thetas, axes.flat):
    img, snr = sim.simulate(theta, return_snr=True)
    ax.imshow(img, origin="lower")
    ax.set_title(f"θE={theta[0]:.2f}, snr≈{snr:.0f}")
    ax.axis("off")
plt.tight_layout()
plt.show()