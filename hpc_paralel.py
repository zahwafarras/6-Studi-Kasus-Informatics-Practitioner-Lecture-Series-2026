"""Studi Kasus 2 - Bridging Global Infrastructure to Address Local Challenges in HPC
(Dr. Warawan, Thammasat University)

Simulasi risiko banjir (Monte Carlo) dijalankan serial vs paralel (multiprocessing),
dihitung speedup & efisiensi, plus hukum Amdahl. Prinsip yang sama dipakai di
cluster HPC (MPI/SLURM); di sini disimulasikan dengan inti CPU lokal.
"""
import os
import time
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SIM = 24_000_000     # jumlah skenario hujan
N_KOTA = 8             # 8 sungai/kelurahan yang disimulasikan


def simulasi(args):
    """Monte Carlo: peluang debit sungai melebihi kapasitas tanggul."""
    seed, n = args
    r = np.random.default_rng(seed)
    hujan = r.gamma(2.0, 40.0, n)              # mm/hari
    tanah = r.uniform(0.2, 0.9, n)             # kejenuhan tanah
    debit = hujan * (0.3 + 0.7 * tanah)
    return float((debit > 150).mean())


def jalankan(n_proses):
    tugas = [(s, N_SIM // N_KOTA) for s in range(N_KOTA)]
    t = time.perf_counter()
    if n_proses == 1:
        hasil = [simulasi(x) for x in tugas]
    else:
        with Pool(n_proses) as p:
            hasil = p.map(simulasi, tugas)
    return time.perf_counter() - t, hasil


def amdahl(p, n):
    return 1 / ((1 - p) + p / n)


if __name__ == "__main__":
    cpu = os.cpu_count() or 1
    print(f"Inti CPU terdeteksi: {cpu}")
    t1, h1 = jalankan(1)
    print(f"Serial (1 proses): {t1:.2f} s")
    jumlah = sorted({1, 2, 4, min(8, cpu)})
    waktu = {1: t1}
    for k in jumlah[1:]:
        waktu[k], hk = jalankan(k)
        assert np.allclose(h1, hk), "hasil paralel harus sama dengan serial"
        print(f"{k} proses: {waktu[k]:.2f} s | speedup={t1 / waktu[k]:.2f}x | "
              f"efisiensi={t1 / waktu[k] / k:.0%}")
    print("\nPeluang banjir per wilayah:", np.round(h1, 4))

    ns = np.arange(1, 65)
    plt.figure(figsize=(6, 4))
    for p in (.5, .9, .99):
        plt.plot(ns, amdahl(p, ns), label=f"{int(p*100)}% paralel")
    plt.plot(list(waktu), [t1 / v for v in waktu.values()], "ko-", label="terukur")
    plt.xlabel("jumlah proses/inti"); plt.ylabel("speedup"); plt.title("Hukum Amdahl")
    plt.legend(); plt.tight_layout(); plt.savefig("hasil_hpc.png", dpi=120)
    print("Grafik disimpan: hasil_hpc.png")
