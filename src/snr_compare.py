import h5py
import matplotlib.pyplot as plt

with h5py.File("../data/train.h5", "r") as f:
    snr_with = f["snr"][:]

with h5py.File("../data/train_no_kappa.h5", "r") as f:
    snr_without = f["snr"][:]

plt.figure(figsize=(7, 4.5))
plt.hist(snr_with, bins=30, alpha=0.6, label="with kappa", edgecolor="black")
plt.hist(snr_without, bins=30, alpha=0.6, label="without kappa", edgecolor="black")
plt.axvline(10, color="red", linestyle="--", label="target range (10-30)")
plt.axvline(30, color="red", linestyle="--")
plt.xlabel("Peak SNR")
plt.ylabel("Count")
plt.title("SNR distribution: with-kappa vs without-kappa datasets")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/snr_comparison.png", dpi=150)
plt.show()
