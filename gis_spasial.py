"""Studi Kasus 3 - Career and Competition in Industrial Era with GIS (Widi Prima Depa)

Analisis spasial ala GIS Analyst untuk pengelolaan aset kota (data sintetis):
 - jarak haversine, nearest-facility (KD-Tree), buffer/jangkauan layanan
 - agregasi per zona (spatial join sederhana) dan peta
Menggunakan Python + NumPy/SciPy (alternatif open source: QGIS, PostGIS).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

rng = np.random.default_rng(3)
LAT0, LON0, R = -0.5, 101.45, 6371.0     # area acuan (sekitar Pekanbaru), derajat/km


def xy_km(lat, lon):
    """Proyeksi datar sederhana (cukup untuk area kota kecil)."""
    return np.c_[(lon - LON0) * 111.32 * np.cos(np.radians(LAT0)), (lat - LAT0) * 110.57]


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# Data: 300 aset (lampu jalan/pipa) dan 6 depo perawatan
aset = pd.DataFrame({"lat": LAT0 + rng.uniform(-.06, .06, 300),
                     "lon": LON0 + rng.uniform(-.06, .06, 300),
                     "kondisi": rng.choice(["baik", "rusak"], 300, p=[.8, .2])})
depo = pd.DataFrame({"nama": [f"Depo-{i}" for i in range(1, 7)],
                     "lat": LAT0 + rng.uniform(-.05, .05, 6),
                     "lon": LON0 + rng.uniform(-.05, .05, 6)})

# 1) Nearest facility
tree = cKDTree(xy_km(depo.lat.values, depo.lon.values))
jarak, idx = tree.query(xy_km(aset.lat.values, aset.lon.values))
aset["depo_terdekat"] = depo.nama.values[idx]
aset["jarak_km"] = jarak
rusak = aset[aset.kondisi == "rusak"]
print(f"Aset rusak: {len(rusak)} dari {len(aset)}")
print(f"Rata-rata jarak aset rusak ke depo terdekat: {rusak.jarak_km.mean():.2f} km")

# 2) Buffer / jangkauan layanan 3 km
dalam = (aset.jarak_km <= 3).mean()
print(f"Aset dalam jangkauan 3 km depo: {dalam:.1%}")

# 3) Beban kerja per depo (agregasi / spatial join)
beban = rusak.groupby("depo_terdekat").size().reindex(depo.nama, fill_value=0)
print("\nBeban perbaikan per depo:\n", beban.to_string())

# 4) Verifikasi haversine vs proyeksi datar
d1 = haversine(depo.lat[0], depo.lon[0], depo.lat[1], depo.lon[1])
d2 = np.linalg.norm(xy_km(depo.lat.values[:1], depo.lon.values[:1]) - xy_km(depo.lat.values[1:2], depo.lon.values[1:2]))
print(f"\nJarak Depo-1 ke Depo-2: haversine={d1:.3f} km | proyeksi datar={d2:.3f} km")

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(aset.lon[aset.kondisi == "baik"], aset.lat[aset.kondisi == "baik"], s=8, c="lightgray", label="aset baik")
ax.scatter(rusak.lon, rusak.lat, s=14, c="red", label="aset rusak")
ax.scatter(depo.lon, depo.lat, s=120, marker="^", c="blue", label="depo")
for _, d in depo.iterrows():
    ax.add_patch(plt.Circle((d.lon, d.lat), 3 / 111.32, fill=False, ls="--", color="blue", alpha=.4))
ax.set(title="Peta aset, depo, dan jangkauan 3 km", xlabel="bujur", ylabel="lintang", aspect="equal")
ax.legend(loc="upper right", fontsize=8); plt.tight_layout(); plt.savefig("hasil_gis.png", dpi=120)
print("Grafik disimpan: hasil_gis.png")

# Catatan portofolio: ekspor GeoJSON agar bisa dibuka di QGIS
import json
geo = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
     "properties": {"kondisi": r.kondisi, "depo": r.depo_terdekat, "jarak_km": round(r.jarak_km, 2)}}
    for r in aset.itertuples()]}
json.dump(geo, open("aset.geojson", "w"))
print("GeoJSON disimpan: aset.geojson (buka di QGIS)")
