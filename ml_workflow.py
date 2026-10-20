"""Studi Kasus 4 - Introduction to Machine Learning (Dr. Taniza Tajudin, UiTM)

Mencakup alur ML: data -> pembersihan -> split latih/uji -> model -> evaluasi,
supervised (regresi harga rumah, klasifikasi penipuan), unsupervised
(segmentasi pelanggan), dan demonstrasi overfitting/underfitting.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

rng = np.random.default_rng(42)

# ---------- 1. SUPERVISED: prediksi harga rumah (regresi) ----------
n = 500
house = pd.DataFrame({
    "luas_m2": rng.uniform(30, 250, n),
    "kamar": rng.integers(1, 6, n),
    "jarak_kota_km": rng.uniform(1, 30, n),
})
house["harga_juta"] = (8 * house.luas_m2 + 60 * house.kamar
                       - 15 * house.jarak_kota_km + 200 + rng.normal(0, 80, n))
house.loc[rng.choice(n, 15, replace=False), "luas_m2"] = np.nan  # data kotor
print("Data hilang sebelum dibersihkan:\n", house.isna().sum(), "\n")
house["luas_m2"] = house["luas_m2"].fillna(house["luas_m2"].median())  # cleaning

X, y = house.drop(columns="harga_juta"), house["harga_juta"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression().fit(Xtr, ytr)
pred_house, y_house = reg.predict(Xte), yte
print(f"[Regresi] MAE={mean_absolute_error(y_house, pred_house):.1f} juta | "
      f"R2={r2_score(y_house, pred_house):.3f}\n")

# ---------- 2. SUPERVISED: deteksi penipuan (klasifikasi) ----------
m = 3000
tx = pd.DataFrame({
    "nominal": rng.exponential(200, m),
    "jam": rng.integers(0, 24, m),
    "negara_beda": rng.integers(0, 2, m),
    "transaksi_24jam": rng.poisson(3, m),
})
risk = (0.004 * tx.nominal + 1.5 * tx.negara_beda + 0.3 * tx.transaksi_24jam
        + 0.8 * (tx.jam < 5) - 4.5)
tx["fraud"] = (rng.random(m) < 1 / (1 + np.exp(-risk))).astype(int)
Xtr, Xte, ytr, yte = train_test_split(tx.drop(columns="fraud"), tx["fraud"],
                                      test_size=0.25, stratify=tx["fraud"], random_state=42)
clf = RandomForestClassifier(200, class_weight="balanced", random_state=42).fit(Xtr, ytr)
print(f"[Klasifikasi] proporsi fraud = {tx.fraud.mean():.1%}")
print(classification_report(yte, clf.predict(Xte), target_names=["normal", "fraud"]))

# ---------- 3. UNSUPERVISED: segmentasi pelanggan ----------
cust = pd.DataFrame({
    "belanja_bulanan": np.r_[rng.normal(200, 30, 100), rng.normal(800, 80, 100), rng.normal(450, 50, 100)],
    "frekuensi": np.r_[rng.normal(2, .5, 100), rng.normal(12, 2, 100), rng.normal(6, 1, 100)],
})
Z = StandardScaler().fit_transform(cust)
cust["segmen"] = KMeans(3, n_init=10, random_state=42).fit_predict(Z)
print("[Clustering] rata-rata per segmen:\n", cust.groupby("segmen").mean().round(1), "\n")

# ---------- 4. OVERFITTING vs UNDERFITTING ----------
xs = np.sort(rng.uniform(0, 1, 40))[:, None]
ys = np.sin(2 * np.pi * xs).ravel() + rng.normal(0, .25, 40)
xtr, xte, ytr2, yte2 = train_test_split(xs, ys, test_size=.4, random_state=1)
degrees = list(range(1, 16))
tr_err, te_err = [], []
for d in degrees:
    mdl = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(xtr, ytr2)
    tr_err.append(np.mean((mdl.predict(xtr) - ytr2) ** 2))
    te_err.append(np.mean((mdl.predict(xte) - yte2) ** 2))
print(f"[Over/Underfitting] derajat polinomial terbaik (error uji minimum) = "
      f"{degrees[int(np.argmin(te_err))]}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(y_house, pred_house, s=10)
lim = [y_house.min(), y_house.max()]
ax[0].plot(lim, lim, "r--")
ax[0].set(title="Regresi: riil vs prediksi", xlabel="harga riil", ylabel="prediksi")
ax[1].scatter(cust.belanja_bulanan, cust.frekuensi, c=cust.segmen, s=12)
ax[1].set(title="Segmentasi pelanggan (K-Means)", xlabel="belanja", ylabel="frekuensi")
ax[2].plot(degrees, tr_err, label="latih")
ax[2].plot(degrees, te_err, label="uji")
ax[2].set(title="Kurva overfitting", xlabel="kompleksitas model", ylabel="MSE", yscale="log")
ax[2].legend()
plt.tight_layout()
plt.savefig("hasil_ml.png", dpi=120)
print("Grafik disimpan: hasil_ml.png")
