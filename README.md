# Studi Kasus Informatics Practitioner Lecture Series 2026

Implementasi Python dari materi para pemateri. Semua data **sintetis** (dibangkitkan lewat kode), jadi tidak perlu mengunduh dataset.

| Folder | Materi | Pemateri | Isi |
|---|---|---|---|
| `01_analitik_data_ai` | Make Sense of Data with Analytics and AI | Andrea Arisal (BRIN) | CRISP-DM pada kasus churn, 4 jenis analisis, perbandingan model ala KNIME |
| `02_hpc_paralel` | Bridging Global Infrastructure to Address Local Challenges in HPC | Dr. Warawan (Thammasat) | Simulasi risiko banjir serial vs paralel, speedup, hukum Amdahl, contoh job SLURM |
| `03_gis_spasial` | Career and Competition in Industrial Era with GIS | Widi Prima Depa | Nearest facility, buffer 3 km, beban kerja per depo, ekspor GeoJSON untuk QGIS |
| `04_machine_learning` | Introduction to Machine Learning | Dr. Taniza Tajudin (UiTM) | Regresi harga rumah, deteksi penipuan, K-Means, overfitting |
| `05_hci_visualisasi` | HCI & Data Visualization | Arafat Febrian Dirza (BRIN) | Skor SUS antar prototipe, pemilihan grafik, grafik menyesatkan vs benar |
| `06_deforestasi_satelit` | Atribusi penyebab deforestasi | Dr. Bart Slachter (WUR) | Klasifikasi penyebab dari patch Sentinel-1/2 simulasi |

## Menjalankan
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd 01_analitik_data_ai && python analitik_crisp_dm.py
cd ../02_hpc_paralel && python hpc_paralel.py
cd ../03_gis_spasial && python gis_spasial.py
cd ../04_machine_learning && python ml_workflow.py
cd ../05_hci_visualisasi && python hci_visualisasi.py
cd ../06_deforestasi_satelit && python deforestasi_drivers.py
```
Setiap skrip mencetak metrik ke terminal dan menyimpan grafik `hasil_*.png`.

## Catatan interpretasi
- **ML:** recall kelas *fraud* rendah (~0,3) karena kelasnya tidak seimbang. Ini sengaja dibiarkan sebagai bahan diskusi: akurasi 83% menyesatkan. Coba ubah threshold atau metrik (PR-AUC).
- **Deforestasi:** akurasi ~100% karena data sintetis terlalu mudah dipisahkan. Pada data nyata hasilnya jauh lebih rendah, terutama untuk kebakaran vs gangguan alam, sebagaimana disampaikan pemateri. Model di sini memakai fitur konvolusi + Random Forest, bukan CNN penuh. Untuk CNN, ganti dengan PyTorch/Keras.
- **HPC:** speedup tergantung jumlah inti CPU komputer Anda. Pada mesin 1 inti hasilnya tidak lebih cepat (itu wajar). Coba di laptop multi-core atau cluster dengan `slurm_contoh.sh`.
- **Analitik:** data churn sintetis; pada data nyata, lakukan CRISP-DM berulang dan validasi hasil model oleh manusia.
- **GIS:** buka `aset.geojson` di QGIS untuk latihan visualisasi dan portofolio.
- **HCI:** skor SUS disimulasikan; ganti dengan data kuesioner asli Anda.

## Upload ke GitHub
```bash
git init
git add .
git commit -m "Studi kasus Informatics Practitioner Lecture Series 2026"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```
Buat dulu repo kosong di github.com/new (tanpa README agar tidak konflik).
