# Oobleck: Resilient Distributed Training Framework

## Install

Using `micromamba` or `conda` on Ubuntu or Debian Linux
```bash
apt install build-essential ninja-build
DS_BUILD_FUSED_ADAM=1 micromamba env create -f environment.yml
micromamba activate oobleck
(oobleck) poetry install
Installing the current project: oobleck (0.1.0)
```