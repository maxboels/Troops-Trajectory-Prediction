# 🎯 Battlefield Target Movement Prediction

A geospatial analysis and machine learning system for predicting and visualizing target movements on a battlefield environment.

![Target Movement Animation](/target_movement_animation.mp4)

## 📋 Project Overview

This project combines geospatial data processing and machine learning to track and predict the movement of targets in a battlefield environment. It includes capabilities for:

- Training prediction models on historical movement data
- Evaluating prediction accuracy against ground truth
- Testing models with various prediction durations
- Visualizing both actual movements and predicted trajectories

---

## 🧭 Battlefield Geospatial Environment Setup

This guide provides detailed steps to set up a Python virtual environment on **Ubuntu** for projects involving **geospatial processing** and **machine learning**.

### 📦 System Dependencies

Begin by updating your package index and installing the necessary system-level libraries:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
sudo apt install -y gdal-bin libgdal-dev
sudo apt install -y libspatialindex-dev
sudo apt install -y libproj-dev proj-data proj-bin
sudo apt install -y libgeos-dev
```

### 🧪 Python Virtual Environment

Set up an isolated virtual environment to manage project dependencies:

```bash
# Create a virtual environment
python3 -m venv battlefield-env

# Activate the virtual environment
source battlefield-env/bin/activate
```

### 🐍 Python Packages Installation

Install required Python libraries, grouped by functionality:

#### General Utilities:

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib requests tqdm
```

#### Geospatial Libraries:

Install GDAL with the exact version provided by your system installation:

```bash
pip install GDAL==$(gdal-config --version) --no-binary=gdal
pip install rasterio geopandas shapely pyproj
```

#### Machine Learning:

```bash
pip install scikit-learn
pip install torch
```

### ✅ Verification

Test that the key libraries are properly installed:

```bash
python -c "import rasterio; import geopandas; import torch; print('All packages imported successfully!')"
```

### 🛠️ Troubleshooting GDAL and Rasterio

Should you encounter issues with GDAL or rasterio installations, you may opt to install them via `conda`:

```bash
# Optional: Create a minimal conda environment for geospatial packages
conda create -n geo_deps -c conda-forge gdal rasterio geopandas
```

You can then use these libraries from the conda environment while continuing development in your Python venv.

#### 📍 Notes

* This setup is tailored for Ubuntu systems. Compatibility with other distributions may vary.
* The use of `--no-binary=gdal` ensures GDAL builds against system libraries for better integration and compatibility.

---

## 🚀 Usage

The project offers several modes of operation to support the full workflow from model training to visualization.

### 🏋️ Training

Train a new prediction model using historical target movement data:

```bash
python target_movement_prediction.py --mode train --data_dir data --output_dir models/target_prediction
```

### 📊 Evaluation

Evaluate model performance against ground truth data:

```bash
python target_movement_prediction.py --mode predict --data_dir data --model_path models/target_prediction/target_predictor_model.pt --output_dir evaluation_results
```

### 🧪 Testing

Test the model with a specific prediction duration (e.g., 3600 seconds):

```bash
python target_movement_prediction.py --mode test --data_dir data --model_path models/target_prediction/target_predictor_model.pt --prediction_duration 3600 --output_dir test_results
```

### 🎨 Visualization

Generate static visualizations of target movements:

```bash
python target_movement_prediction.py --mode visualize --data_dir data --output_dir visualizations
```

### 🎬 Animation

Create animated visualizations of target movements and predictions:

```bash
# Generate animation of actual target movements
python target_movement_animation.py --data_dir data --output_file visualizations/target_animation.mp4 --duration 2

# Generate animation of predicted movements
python target_prediction_animation.py --model_path models/target_prediction/target_predictor_model.pt --data_dir data --output_file visualizations/target_predictions.mp4

# Generate animation with both actual and predicted movements
python target_prediction_animation_with_preds.py --model_path models/target_prediction/target_predictor_model.pt --data_dir data --output_file visualizations/target_predictions.mp4
```

---

## 📁 Project Structure

```
.
├── data/                       # Data directory containing target and terrain data
├── models/                     # Trained prediction models
│   └── target_prediction/      # Target movement prediction models
├── visualization/              # Generated visualizations and animations
├── target_movement_prediction.py  # Main script for training and prediction
├── target_movement_animation.py   # Script for generating movement animations
├── target_prediction_animation.py # Script for animating predictions
├── target_prediction_animation_with_preds.py  # Script for comparison animations
└── README.md                   # This file
```

---

## 💡 Features

- **Data Processing**: Handles various geospatial data formats including CSV, GeoJSON, and raster files
- **Machine Learning**: Employs deep learning models to predict future target positions
- **Time-Based Visualization**: Color-coded visualization of movement paths based on timestamp
- **Animation**: Dynamic visualization of target movements over time
- **Prediction Visualization**: Compare actual vs. predicted movement paths

---

## 📜 License

[Your License Information Here]

---

## 🤝 Contributing

[Your Contributing Guidelines Here]

---

## 🙏 Acknowledgments

[Your Acknowledgments Here]
