# 🧭 Battlefield Geospatial Environment Setup

This guide provides detailed steps to set up a Python virtual environment on **Ubuntu** for projects involving **geospatial processing** and **machine learning**.

---

## 📦 System Dependencies

Begin by updating your package index and installing the necessary system-level libraries:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
sudo apt install -y gdal-bin libgdal-dev
sudo apt install -y libspatialindex-dev
sudo apt install -y libproj-dev proj-data proj-bin
sudo apt install -y libgeos-dev
```

---

## 🧪 Python Virtual Environment

Set up an isolated virtual environment to manage project dependencies:

```bash
# Create a virtual environment
python3 -m venv battlefield-env

# Activate the virtual environment
source battlefield-env/bin/activate
```

---

## 🐍 Python Packages Installation

Install required Python libraries, grouped by functionality:

### General Utilities:

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib requests tqdm
```

### Geospatial Libraries:

Install GDAL with the exact version provided by your system installation:

```bash
pip install GDAL==$(gdal-config --version) --no-binary=gdal
pip install rasterio geopandas shapely pyproj
```

### Machine Learning:

```bash
pip install scikit-learn
pip install torch
```

---

## ✅ Verification

Test that the key libraries are properly installed:

```bash
python -c "import rasterio; import geopandas; import torch; print('All packages imported successfully!')"
```

---

## 🛠️ Troubleshooting GDAL and Rasterio

Should you encounter issues with GDAL or rasterio installations, you may opt to install them via `conda`:

```bash
# Optional: Create a minimal conda environment for geospatial packages
conda create -n geo_deps -c conda-forge gdal rasterio geopandas
```

You can then use these libraries from the conda environment while continuing development in your Python venv.

---

## 📍 Notes

* This setup is tailored for Ubuntu systems. Compatibility with other distributions may vary.
* The use of `--no-binary=gdal` ensures GDAL builds against system libraries for better integration and compatibility.

---

## ✨ Ready to Go

With this environment, you are now equipped to handle both **geospatial data processing** and **deep learning tasks** efficiently.