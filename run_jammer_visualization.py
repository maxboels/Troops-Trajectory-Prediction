import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os

# Import visualization functions from our module
from jammer_prediction_visualization import (
    load_jammer_model_for_visualization,
    visualize_predictions_at_timestep,
    create_prediction_animation,
    load_and_process_jammer_data,
)
from battlefield_simulation import BattlefieldSimulation


def main():
    """
    Interactive CLI for jammer prediction visualization.
    """
    parser = argparse.ArgumentParser(description="Visualize jammer predictions")
    parser.add_argument(
        "--mode", 
        choices=["static", "animation", "both"], 
        default="both",
        help="Visualization mode (static, animation, or both)"
    )
    parser.add_argument(
        "--timestamp", 
        type=str,
        help="Timestamp for static visualization (format: YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--start_time", 
        type=str,
        help="Start time for animation (format: YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=6.0,
        help="Duration of animation in hours"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=15,
        help="Time interval between frames in minutes"
    )
    parser.add_argument(
        "--horizon", 
        type=int, 
        default=5,
        help="Prediction horizon (number of steps to predict ahead)"
    )
    parser.add_argument(
        "--sequence_length", 
        type=int, 
        default=10,
        help="Input sequence length"
    )
    parser.add_argument(
        "--max_frames", 
        type=int, 
        default=200,
        help="Maximum number of frames in animation"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="visualizations",
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictor model
    print("Loading predictor model...")
    predictor = load_jammer_model_for_visualization()
    if predictor is None:
        print("Failed to load predictor model. Exiting.")
        return
    
    # Load jammer data
    print("Loading jammer data...")
    jammer_data = load_and_process_jammer_data("synthetic_data/jammer_observations.csv")
    if jammer_data is None:
        print("Failed to load jammer data. Exiting.")
        return
    
    # Load battlefield simulation for terrain
    print("Loading terrain data...")
    sim = BattlefieldSimulation()
    sim.load_terrain_data(
        terrain_data_path="simulation_data/terrain_map.npy",
        elevation_data_path="simulation_data/elevation_map.npy"
    )
    
    # Generate static visualization if requested
    if args.mode in ["static", "both"]:
        print("Generating static visualization...")
        
        # Use provided timestamp or pick one from the middle of the data
        if args.timestamp:
            timestamp = pd.to_datetime(args.timestamp)
        else:
            timestamps = sorted(jammer_data['timestamp'].unique())
            timestamp = timestamps[len(timestamps) // 2]
            print(f"Using timestamp from middle of data: {timestamp}")
        
        # Create the visualization
        visualize_predictions_at_timestep(
            predictor=predictor,
            jammer_data=jammer_data,
            timestamp=timestamp,
            prediction_horizon=args.horizon,
            sequence_length=args.sequence_length,
            sim=sim,
            output_dir=args.output_dir,
            filename=f"jammer_predictions_{timestamp.strftime('%Y%m%d_%H%M')}.png"
        )
    
    # Generate animation if requested
    if args.mode in ["animation", "both"]:
        print("Generating prediction animation...")
        
        # Use provided start time or determine automatically
        start_time = None
        if args.start_time:
            start_time = pd.to_datetime(args.start_time)
        
        # Create the animation
        create_prediction_animation(
            predictor=predictor,
            jammer_data=jammer_data,
            start_time=start_time,
            duration_hours=args.duration,
            interval_minutes=args.interval,
            prediction_horizon=args.horizon,
            sequence_length=args.sequence_length,
            max_frames=args.max_frames,
            sim=sim,
            output_filename=f"{args.output_dir}/jammer_prediction_animation.mp4",
            fps=10
        )
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
