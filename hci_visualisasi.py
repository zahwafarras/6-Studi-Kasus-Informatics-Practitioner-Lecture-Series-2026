"""Studi Kasus 5 - HCI & Data Visualization (Arafat Febrian Dirza, BRIN)

Bagian A: evaluasi usability dengan skor SUS (System Usability Scale)
          untuk membandingkan dua prototipe antarmuka (iterasi desain).
Bagian B: pemilihan jenis grafik yang tepat (tren -> garis, perbandingan -> batang,
          komposisi -> batang bertumpuk) dan contoh grafik yang menyesatkan.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---------- A. Evaluasi usability (SUS) ----------
def sus_score(jawaban):
    """jawaban: 10 nilai skala 1-5. Item ganjil positif, genap negatif."""
    j = np.asarray(jawaban)
    skor = np.where(np.arange(10) % 2 == 0, j - 1, 5 - j)
    return skor.sum() * 2.5

def simulasi_responden(n, kualitas):
    pos = np.clip(rng.normal(kualitas, 0.7, (n, 5)).round(), 1, 5)      # item 1,3,5,7,9
    neg = np.clip(rng.normal(6 - kualitas, 0.7, (n, 5)).round(), 1, 5)  # item 2,4,6,8,10
    out = np.empty((n, 10))
    out[:, 0::2], out[:, 1::2] = pos, neg
    return out

hasil = {}
for nama, q in {"Prototipe v1": 3.0, "Prototipe v2 (setelah iterasi)": 4.2}.items():
    skor = [sus_score(r) for r in simulasi_responden(20, q)]
    hasil[nama] = skor
    print(f"{nama}: rata-rata SUS = {np.mean(skor):.1f} (n=20)")
print("Acuan: SUS >= 68 dianggap di atas rata-rata usability.\n")

# ---------- B. Visualisasi data ----------
bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun"]
data = pd.DataFrame({
    "bulan": bulan,
    "pengguna_web": [120, 135, 150, 170, 160, 190],
    "pengguna_mobile": [80, 95, 130, 150, 175, 210],
})
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].boxplot(hasil.values(), tick_labels=["v1", "v2"])
ax[0, 0].axhline(68, color="r", ls="--", label="batas 68")
ax[0, 0].set(title="A. Skor SUS: iterasi desain meningkatkan usability", ylabel="SUS")
ax[0, 0].legend()

ax[0, 1].plot(data.bulan, data.pengguna_web, marker="o", label="Web")
ax[0, 1].plot(data.bulan, data.pengguna_mobile, marker="o", label="Mobile")
ax[0, 1].set(title="B1. TREN -> grafik garis", ylabel="pengguna aktif")
ax[0, 1].legend()

# Perbandingan antar kategori yang jujur (sumbu mulai dari 0) vs menyesatkan
kat, nilai = ["A", "B", "C"], [96, 98, 99]
ax[1, 0].bar(kat, nilai, color="tab:red")
ax[1, 0].set_ylim(95, 100)
ax[1, 0].set(title="B2. MENYESATKAN: sumbu dipotong (selisih tampak besar)")

ax[1, 1].bar(kat, nilai, color="tab:green")
ax[1, 1].set_ylim(0, 100)
ax[1, 1].set(title="B3. BENAR: sumbu mulai dari 0 (selisih kecil)")

plt.tight_layout()
plt.savefig("hasil_hci.png", dpi=120)
print("Grafik disimpan: hasil_hci.png")

# ---------- Checklist heuristik Nielsen (ringkas) ----------
checklist = [
    "Konsistensi: istilah & tata letak sama di semua halaman",
    "Umpan balik: sistem merespons setiap aksi pengguna",
    "Pencegahan kesalahan: konfirmasi sebelum hapus data",
    "Keterbacaan: kontras warna & ukuran font memadai",
    "Navigasi: pengguna selalu tahu posisinya",
]
print("\nChecklist evaluasi heuristik:")
for c in checklist:
    print(" [ ]", c)
