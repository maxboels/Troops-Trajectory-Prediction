# Create a new file: battlefield_prediction.py

from improved_battlefield_simulation import BattlefieldSimulation
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_decoder_only_transformer import CausalTransformerPredictor  # Your model

# Create and load simulation data
sim = BattlefieldSimulation()

# Load terrain data
sim.load_terrain_data(
    terrain_data_path="simulation_data/terrain_map.npy",
    elevation_data_path="simulation_data/elevation_map.npy"
)

# Load observation data
sim.load_observation_data(
    target_csv="synthetic_data/target_observations.csv",
    blue_force_csv="synthetic_data/blue_force_observations.csv"
)

# Generate datasets
datasets = sim.build_trajectory_datasets(
    test_ratio=0.2,
    window_size=5,
    prediction_horizons=[1, 3, 5, 10],
    include_terrain=True,
    include_blue_forces=True
)

# Initialize and train your transformer model
# [Your model training code here]

# Visualize predictions
sim.visualize_prediction(model, target_id="target-0", timestamp=None, prediction_horizon=5)
plt.show()