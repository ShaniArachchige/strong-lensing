import h5py
import matplotlib.pyplot as plt

with h5py.File("../data/train.h5", "r") as f:
    snr = f["snr"][:]

plt.figure(figsize=(6, 4))
plt.hist(snr, bins=30, edgecolor="black")
plt.axvline(10, color="red", linestyle="--", label="target range (10-30)")
plt.axvline(30, color="red", linestyle="--")
plt.xlabel("Peak SNR")
plt.ylabel("Count")
plt.title("SNR distribution across training images")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/snr_distribution.png", dpi=150)
plt.show()
