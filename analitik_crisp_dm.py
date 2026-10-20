"""Studi Kasus 1 - Make Sense of Data with Analytics and AI (Andrea Arisal, BRIN)

Alur CRISP-DM pada kasus churn pelanggan telekomunikasi (data sintetis),
mencakup 4 jenis analisis (deskriptif, diagnostik, prediktif, preskriptif)
dan perbandingan beberapa model seperti workflow KNIME, tetapi lewat kode.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(1)

# 1) Business understanding: kurangi churn dengan menawarkan retensi ke pelanggan berisiko
# 2) Data understanding
n = 2000
df = pd.DataFrame({
    "lama_langganan_bln": rng.integers(1, 72, n),
    "tagihan_bulanan": rng.normal(250, 60, n).round(0),
    "keluhan_3bln": rng.poisson(1.2, n),
    "kontrak_tahunan": rng.integers(0, 2, n),
})
logit = (-1.0 - 0.04 * df.lama_langganan_bln + 0.008 * df.tagihan_bulanan
         + 0.6 * df.keluhan_3bln - 1.2 * df.kontrak_tahunan)
df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df.loc[rng.choice(n, 60, replace=False), "tagihan_bulanan"] = np.nan   # data kotor
df.loc[rng.choice(n, 5, replace=False), "tagihan_bulanan"] = 9999      # outlier

# 3) Data preparation
df["tagihan_bulanan"] = df["tagihan_bulanan"].fillna(df["tagihan_bulanan"].median())
df = df[df.tagihan_bulanan < 1000].reset_index(drop=True)

print("== DESKRIPTIF: apa yang terjadi? ==")
print(f"Tingkat churn: {df.churn.mean():.1%}")
print(df.groupby("kontrak_tahunan").churn.mean().rename("churn per jenis kontrak").round(3), "\n")

print("== DIAGNOSTIK: mengapa? ==")
print(df.corr()["churn"].drop("churn").round(3).sort_values(), "\n")

# 4) Modeling + 5) Evaluation (perbandingan model ala KNIME)
X, y = df.drop(columns="churn"), df["churn"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=1)
modeles = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=1),
    "Random Forest": RandomForestClassifier(200, random_state=1),
}
print("== PREDIKTIF: perbandingan model ==")
skor = {}
for nama, m in modeles.items():
    cv = cross_val_score(m, Xtr, ytr, cv=5, scoring="roc_auc").mean()
    m.fit(Xtr, ytr)
    skor[nama] = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    print(f"{nama:20s} AUC-CV={cv:.3f} | AUC-uji={skor[nama]:.3f}")
terbaik = max(skor, key=skor.get)
print("Model terpilih:", terbaik, "\n")

# 6) Deployment / PRESKRIPTIF: rekomendasi tindakan berbasis risiko
prob = modeles[terbaik].predict_proba(Xte)[:, 1]
hasil = Xte.copy()
hasil["risiko"] = prob
hasil["rekomendasi"] = np.select(
    [prob >= .6, prob >= .35],
    ["Tawarkan diskon + telepon CS", "Kirim survei kepuasan"],
    "Pantau saja")
print("== PRESKRIPTIF: rekomendasi tindakan ==")
print(hasil.rekomendasi.value_counts(), "\n")
print("Catatan: hasil AI/model harus diverifikasi manusia; jaga privasi data pelanggan.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(skor.keys(), skor.values()); ax[0].set(ylim=(.5, .9), title="Perbandingan AUC model", ylabel="AUC")
ax[0].tick_params(axis="x", labelrotation=15)
hasil.rekomendasi.value_counts().plot.barh(ax=ax[1], title="Rekomendasi tindakan (preskriptif)")
plt.tight_layout(); plt.savefig("hasil_analitik.png", dpi=120)
print("Grafik disimpan: hasil_analitik.png")
