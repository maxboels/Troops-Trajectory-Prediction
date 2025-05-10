"""
Target Prediction Example - Shows how to use the visualization module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Import the target prediction module
from target_movement_prediction import TargetMovementPredictor, load_and_process_data

# Import our new visualization module
from target_visualization import TargetVisualizer

def run_visualization_examples():
    """Run several examples of target movement visualization."""
    
    # Set paths
    data_dir = "data"
    terrain_path = "adapted_data/terrain_map.npy"
    elevation_path = "adapted_data/elevation_map.npy"
    model_path = "models/target_prediction/target_predictor_model_100ep.pt" # "models/target_predictor_model.pt"
    output_dir = "output/visualizations"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_and_process_data(
        target_csv=os.path.join(data_dir, "red_sightings.csv"),
        blue_csv=os.path.join(data_dir, "blue_locations.csv"),
        terrain_path=terrain_path,
        elevation_path=elevation_path
    )
    
    target_data = data['target_data']
    blue_force_data = data['blue_force_data']
    terrain_data = data['terrain_data']
    elevation_data = data['elevation_data']
    
    # Initialize predictor and load the model
    predictor = TargetMovementPredictor(
        terrain_data_path=terrain_path,
        elevation_data_path=elevation_path
    )
    
    # Load model (or train one if not available)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        predictor.load_model(model_path)
    else:
        print(f"Model not found at {model_path}, training a new model...")
        train_loader, val_loader = predictor.prepare_data(target_data, blue_force_data)
        history = predictor.train(
            train_loader, 
            val_loader, 
            num_epochs=5  # Use fewer epochs for testing
        )
        predictor.save_model(model_path)
    
    # Create visualizer
    visualizer = TargetVisualizer(terrain_data, elevation_data)
    
    # Choose a timestamp for prediction
    timestamps = sorted(target_data['datetime'].unique())
    mid_idx = len(timestamps) // 2
    timestamp = timestamps[mid_idx]
    print(f"Using timestamp: {timestamp}")
    
    # EXAMPLE 1: Single Target Visualization
    # --------------------------------------
    # Select a target ID
    target_ids = target_data['target_id'].unique()
    selected_target = target_ids[0]
    
    # Make prediction
    prediction = predictor.predict_out_of_view(
        target_data, 
        selected_target, 
        timestamp,
        300  # 5-minute prediction
    )
    
    if prediction is not None:
        # Create visualization
        visualizer.visualize_single_target(
            target_data, 
            selected_target, 
            timestamp, 
            prediction,
            blue_force_data=blue_force_data,
            output_path=os.path.join(output_dir, f"single_target_{selected_target}.png")
        )
        print(f"Single target visualization saved for target {selected_target}")
    
    # EXAMPLE 2: All Targets Visualization
    # ------------------------------------
    # Make predictions for all targets
    all_predictions = {}
    
    for target_id in target_ids:
        # Skip if not enough history data at this timestamp
        target_df = target_data[target_data['target_id'] == target_id]
        if len(target_df[target_df['datetime'] <= timestamp]) < predictor.config['sequence_length']:
            continue
            
        try:
            # Generate prediction
            pred = predictor.predict_out_of_view(
                target_data, 
                target_id, 
                timestamp,
                300  # 5-minute prediction
            )
            
            if pred is not None:
                all_predictions[target_id] = pred
        except Exception as e:
            print(f"Error predicting for target {target_id}: {e}")
    
    # Create visualization for all targets
    if all_predictions:
        visualizer.visualize_all_targets(
            target_data,
            timestamp,
            all_predictions,
            blue_force_data=blue_force_data,
            output_path=os.path.join(output_dir, "all_targets.png")
        )
        print(f"All targets visualization saved with {len(all_predictions)} targets")
    
    # EXAMPLE 3: High-Quality Terrain Visualization
    # ---------------------------------------------
    if all_predictions:
        visualizer.visualize_predictions_on_terrain(
            target_data,
            all_predictions,
            timestamp,
            blue_force_data=blue_force_data,
            output_path=os.path.join(output_dir, "terrain_visualization.png")
        )
        print("High-quality terrain visualization saved")
    
    # EXAMPLE 4: Training History Visualization
    # -----------------------------------------
    # Create sample history data (or load from actual training)
    history = {
        'train_loss': np.linspace(-0.1, -0.5, 100) + np.random.normal(0, 0.03, 100),
        'val_loss': np.linspace(-0.25, -0.5, 100) + np.random.normal(0, 0.05, 100),
        'train_nll_loss': np.linspace(-1.5, -5.0, 100) + np.random.normal(0, 0.2, 100),
        'val_nll_loss': np.linspace(-2.5, -5.0, 100) + np.random.normal(0, 0.3, 100),
        'best_epoch': 92
    }
    
    visualizer.plot_training_history(
        history,
        output_path=os.path.join(output_dir, "training_history.png")
    )
    print("Training history visualization saved")
    
    print("\nAll visualizations completed!")
    
    # Optionally create animation (uncomment to run - this can take a while)
    # if len(all_predictions) > 0:
    #     visualizer.create_animation(
    #         predictor,
    #         target_data,
    #         blue_force_data,
    #         output_filename=os.path.join(output_dir, "target_animation.mp4")
    #     )
    #     print("Animation saved!")

if __name__ == "__main__":
    # Run the examples
    run_visualization_examples()
