"""Studi Kasus 6 - Atribusi Penyebab Deforestasi dengan Citra Satelit & AI
(Dr. Bart Slachter, Wageningen University & Research)

Simulasi patch citra 16x16 dengan 4 kanal:
  Sentinel-2 : Red, NIR (optik -> NDVI)
  Sentinel-1 : VV, VH (radar, tembus awan)
Kelas penyebab: 0 pertanian, 1 jalan, 2 pertambangan, 3 gangguan alam/kebakaran.

Karena PyTorch tidak wajib, model "CNN" dibuat sederhana dengan NumPy-konvolusi
sebagai ekstraktor fitur + Random Forest. Pada GPU/PyTorch, ganti dengan CNN penuh.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

rng = np.random.default_rng(0)
KELAS = ["pertanian", "jalan", "tambang", "alam/kebakaran"]
S = 16


def buat_patch(kelas):
    """Hasilkan patch (4,S,S): red, nir, vv, vh dengan pola spasial khas tiap penyebab."""
    red = rng.normal(0.08, 0.02, (S, S)); nir = rng.normal(0.45, 0.05, (S, S))
    vv = rng.normal(-8, 1, (S, S)); vh = rng.normal(-14, 1, (S, S))
    mask = np.zeros((S, S), bool)
    if kelas == 0:      # petak persegi besar (lahan pertanian)
        a, b = rng.integers(0, 6, 2); mask[a:a + 9, b:b + 9] = True
        red[mask] += .12; nir[mask] -= .22; vv[mask] -= 2
    elif kelas == 1:    # garis tipis memanjang (jalan)
        r = rng.integers(3, 12); mask[r:r + 2, :] = True
        red[mask] += .15; nir[mask] -= .28; vv[mask] -= 4
    elif kelas == 2:    # bercak tidak beraturan sangat terang di radar (tambang)
        mask = rng.random((S, S)) < .35
        mask = convolve2d(mask, np.ones((3, 3)), mode="same") > 3
        red[mask] += .2; nir[mask] -= .3; vv[mask] += 3; vh[mask] += 3
    else:               # gangguan alam: area tersebar, NIR turun, radar relatif stabil
        mask = rng.random((S, S)) < .5
        red[mask] += .06; nir[mask] -= .15
    return np.stack([red, nir, vv, vh]), mask


def ekstrak_fitur(p):
    """Fitur ala lapisan konvolusi: NDVI, gradien (Sobel) dan statistik radar."""
    red, nir, vv, vh = p
    ndvi = (nir - red) / (nir + red + 1e-6)
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gx = convolve2d(ndvi, sobel, mode="valid"); gy = convolve2d(ndvi, sobel.T, mode="valid")
    hilang = ndvi < 0.5   # piksel kehilangan vegetasi
    baris, kolom = hilang.any(1).sum(), hilang.any(0).sum()
    return [ndvi.mean(), ndvi.std(), hilang.mean(), baris, kolom,
            np.abs(gx).mean() + np.abs(gy).mean(),
            vv[hilang].mean() if hilang.any() else vv.mean(),
            vh[hilang].mean() if hilang.any() else vh.mean(),
            vv.std(), max(baris, kolom) / (min(baris, kolom) + 1)]


N = 1600
y = rng.integers(0, 4, N)
patches = [buat_patch(k) for k in y]
X = np.array([ekstrak_fitur(p) for p, _ in patches])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
model = RandomForestClassifier(300, random_state=0).fit(Xtr, ytr)
pred = model.predict(Xte)
print(classification_report(yte, pred, target_names=KELAS))

# ---- Simulasi sistem peringatan: alert mingguan, atribusi penyebab bulanan ----
print("Simulasi 4 minggu: alert dikumpulkan tiap minggu, penyebab diperbarui bulanan")
alerts = rng.integers(0, 4, 40)
idx = [i for i in range(N) if y[i] in alerts][:40]
prediksi_bulanan = model.predict(X[idx])
for k, nama in enumerate(KELAS):
    print(f"  {nama:15s}: {(prediksi_bulanan == k).sum()} alert")

# ---- Visualisasi ----
fig, ax = plt.subplots(2, 5, figsize=(15, 6))
for k in range(4):
    j = int(np.where(y == k)[0][0]); p, mask = patches[j]
    ndvi = (p[1] - p[0]) / (p[1] + p[0])
    ax[0, k].imshow(ndvi, cmap="RdYlGn", vmin=0, vmax=1)
    ax[0, k].set_title(f"NDVI: {KELAS[k]}", fontsize=9)
    ax[1, k].imshow(p[2], cmap="gray"); ax[1, k].set_title("Radar VV", fontsize=9)
ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=KELAS).plot(
    ax=ax[0, 4], xticks_rotation=45, colorbar=False)
ax[0, 4].set_title("Confusion matrix", fontsize=9)
ax[1, 4].barh(["NDVI mean", "NDVI std", "frak. hilang", "baris", "kolom", "gradien",
               "VV", "VH", "VV std", "rasio bentuk"], model.feature_importances_)
ax[1, 4].set_title("Pentingnya fitur", fontsize=9)
for a in ax[:, :4].ravel():
    a.axis("off")
plt.tight_layout()
plt.savefig("hasil_deforestasi.png", dpi=110)
print("Grafik disimpan: hasil_deforestasi.png")
