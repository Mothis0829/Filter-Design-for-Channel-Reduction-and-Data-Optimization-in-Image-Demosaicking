import matplotlib.pyplot as plt

labels = [
    "I0", "I45", "I90", "I135",
    "S0", "S1", "S2", "DoP", "Average"
]

psnr = [42.68, 41.15, 43.24, 41.32, 44.91, 48.58, 44.09, 32.46, 42.30]
psnr_std = [0.04, 0.05, 0.10, 0.02, 0.08, 0.05, 0.01, 0.03, 0.04]

plt.figure(figsize=(10,5))
plt.bar(labels, psnr, yerr=psnr_std, capsize=4)
plt.ylabel("CPSNR (dB)")
plt.title("Polarization Reconstruction Error Summary")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("error_summary_psnr.jpg", dpi=300)
plt.close()
