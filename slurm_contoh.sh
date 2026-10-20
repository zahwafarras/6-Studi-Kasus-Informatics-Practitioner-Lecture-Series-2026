#!/bin/bash
# Contoh job script untuk cluster HPC (SLURM). Sesuaikan partisi/modul di cluster Anda.
#SBATCH --job-name=banjir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
module load python
python hpc_paralel.py
