# target_prediction_animation.py
# target_prediction_animation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import os
from tqdm import tqdm
import argparse
from datetime import timedelta

def create_prediction_animation(predictor, target_data_path, blue_force_path=None, 
                               terrain_path=None, elevation_path=None,
                               output_file="target_prediction_animation.mp4",
                               fps=15, dpi=150, duration_seconds=30,
                               prediction_duration=300):
    """
    Create an animation of target movement predictions over time.
    
    Args:
        predictor: Trained TargetMovementPredictor model
        target_data_path: Path to target data CSV
        blue_force_path: Path to blue force data CSV (optional)
        terrain_path: Path to terrain data NPY file (optional)
        elevation_path: Path to elevation data NPY file (optional)
        output_file: Path to save the animation
        fps: Frames per second
        dpi: Resolution of the animation
        duration_seconds: Duration of the animation in seconds
        prediction_duration: How far to predict in seconds for each target
    """
    # Load data
    # target_data = pd.read_csv(target_data_path)
    target_data = pd.read_csv("test_results/test_predictions.csv")
    print(f"Loaded target data with shape {target_data.shape}")
    print(f"Target data columns: {target_data.columns.tolist()}")
    
    # Target data columns: 
    # ['target_id', 'target_class', 'prediction_step', 'timestamp', 
    # 'predicted_longitude', 'predicted_latitude', 'longitude_lower_ci', 'longitude_upper_ci', 
    # 'latitude_lower_ci', 'latitude_upper_ci', 'longitude_ci_size', 'latitude_ci_size', 'speed_m_s', 
    # 'time_since_last_seen_s', 'last_seen_time']

    # Ensure datetime is properly parsed
    if 'datetime' in target_data.columns:
        target_data['datetime'] = pd.to_datetime(target_data['datetime'])
    elif 'timestamp' in target_data.columns:
        target_data['datetime'] = pd.to_datetime(target_data['timestamp'])
    else:
        raise ValueError("No datetime or timestamp column found in target data.")
    
    # Load blue force data if provided
    blue_force_data = None
    if blue_force_path and os.path.exists(blue_force_path):
        blue_force_data = pd.read_csv(blue_force_path)
    
    # Load terrain and elevation data if available
    terrain_data = None
    elevation_data = None
    
    if terrain_path and os.path.exists(terrain_path):
        terrain_data = np.load(terrain_path)
        print(f"Loaded terrain data with shape {terrain_data.shape}")
        
        # Flip terrain data for correct orientation
        terrain_data = np.flipud(terrain_data)
    else:
        print("Warning: Terrain data not found or not provided.")
    
    if elevation_path and os.path.exists(elevation_path):
        elevation_data = np.load(elevation_path)
        print(f"Loaded elevation data with shape {elevation_data.shape}")
        
        # Flip elevation data for correct orientation
        elevation_data = np.flipud(elevation_data)
    else:
        print("Warning: Elevation data not found or not provided.")
    
    # Get all unique timestamps
    timestamps = sorted(pd.unique(target_data['datetime']))
    
    # Determine how many timestamps to use based on desired duration
    total_frames = fps * duration_seconds
    frame_skip = max(1, len(timestamps) // total_frames)
    selected_timestamps = timestamps[::frame_skip]
    
    print(f"Creating animation with {len(selected_timestamps)} frames...")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get coordinate bounds
    if "longitude" in target_data.columns and "latitude" in target_data.columns:
        lon_min, lon_max = target_data['longitude'].min(), target_data['longitude'].max()
        lat_min, lat_max = target_data['latitude'].min(), target_data['latitude'].max()
    elif "predicted_longitude" in target_data.columns and "predicted_latitude" in target_data.columns:
        lon_min, lon_max = target_data['predicted_longitude'].min(), target_data['predicted_longitude'].max()
        lat_min, lat_max = target_data['predicted_latitude'].min(), target_data['predicted_latitude'].max()
    else:
        raise ValueError("No longitude or latitude columns found in target data.")

    # lon_min, lon_max = target_data['longitude'].min(), target_data['longitude'].max()
    # lat_min, lat_max = target_data['latitude'].min(), target_data['latitude'].max()
    
    # Add padding
    padding = 0.05
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    lon_min -= lon_range * padding
    lon_max += lon_range * padding
    lat_min -= lat_range * padding
    lat_max += lat_range * padding
    
    # Define terrain colormap
    land_use_colors = [
        '#FFFFFF',  # 0: No data
        '#1A5BAB',  # 1: Broadleaf Forest
        '#358221',  # 2: Deciduous Forest
        '#2E8B57',  # 3: Evergreen Forest
        '#52A72D',  # 4: Mixed Forest
        '#76B349',  # 5: Tree Open
        '#D2B48C',  # 7: Shrub
        '#9ACD32',  # 8: Herbaceous
        '#F5DEB3',  # 10: Sparse vegetation
        '#FFD700',  # 11: Cropland
        '#F4A460',  # 12: Agricultural
        '#2F4F4F',  # 14: Wetland
        '#A0522D',  # 16: Bare area
        '#808080',  # 18: Urban
        '#0000FF',  # 20: Water
    ]
    
    terrain_cmap = ListedColormap(land_use_colors)
    
    # Plot terrain data if available
    if terrain_data is not None:
        from matplotlib.colors import Normalize
        terrain_norm = Normalize(0, 20)
        
        ax.imshow(terrain_data, cmap=terrain_cmap, norm=terrain_norm, alpha=0.7,
                 extent=[lon_min, lon_max, lat_min, lat_max],
                 aspect='auto', origin='lower', zorder=0)
        
        # Add elevation overlay if available
        if elevation_data is not None:
            elev_min = np.min(elevation_data)
            elev_max = np.max(elevation_data)
            elev_norm = Normalize(vmin=elev_min, vmax=elev_max)
            
            ax.imshow(elevation_data, cmap='terrain', norm=elev_norm, alpha=0.3,
                     extent=[lon_min, lon_max, lat_min, lat_max],
                     aspect='auto', origin='lower', zorder=1)
    
    # Define colors for target classes
    target_colors = {
        'tank': 'darkred',
        'armoured personnel carrier': 'orangered',
        'light vehicle': 'coral',
        'unknown': 'red'
    }
    
    # Set up plot elements
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Target Movement Prediction Animation', fontsize=14)
    
    # Add terrain legend
    terrain_legend = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#0000FF', markersize=10, label='Water'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#808080', markersize=10, label='Urban'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#358221', markersize=10, label='Forest'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#9ACD32', markersize=10, label='Herbaceous'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFD700', markersize=10, label='Cropland')
    ]
    
    # Target legend
    target_legend = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=10, label='Tank'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orangered', markersize=10, label='APC'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', markersize=10, label='Light Vehicle'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Blue Forces')
    ]
    
    # Add legend
    first_legend = ax.legend(handles=terrain_legend, loc='upper left', title='Terrain')
    ax.add_artist(first_legend)
    ax.legend(handles=target_legend, loc='upper right', title='Forces')
    
    # Blue force scatter plot
    blue_scatter = ax.scatter([], [], c='blue', s=120, marker='^', 
                            zorder=10, edgecolor='black')
    
    # Create a scatter plot for each target class
    target_scatters = {}
    for target_class, color in target_colors.items():
        target_scatters[target_class] = ax.scatter([], [], c=color, s=80, marker='o', 
                                                 zorder=8, edgecolor='black')
    
    # Track target trails, predictions, and ellipses
    target_trails = {}
    target_predictions = {}
    prediction_ellipses = {}
    
    # Time display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                      bbox=dict(facecolor='white', alpha=0.7), zorder=20)
    
    # Initialize function
    def init():
        blue_scatter.set_offsets(np.empty((0, 2)))
        for scatter in target_scatters.values():
            scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return [blue_scatter] + list(target_scatters.values()) + [time_text]
    
    # Update function for animation
    def update(frame):
        # Get current timestamp
        timestamp = selected_timestamps[frame]
        
        # Filter data for this timestamp
        current_data = target_data[target_data['datetime'] <= timestamp]
        
        # Get the most recent positions for each target
        if not current_data.empty:
            latest_positions = current_data.groupby('target_id').apply(
                lambda x: x.loc[x['datetime'].idxmax()]
            ).reset_index(drop=True)
        else:
            latest_positions = pd.DataFrame(columns=current_data.columns)
        
        # Update blue forces
        if blue_force_data is not None:
            blue_scatter.set_offsets(blue_force_data[['longitude', 'latitude']].values)
        
        # Update target positions by class
        for target_class, color in target_colors.items():
            # Get targets of this class
            if 'target_class' in latest_positions.columns:
                class_targets = latest_positions[
                    latest_positions['target_class'].str.lower() == target_class
                ]
            else:
                # If no class info, treat all as unknown
                class_targets = latest_positions if target_class == 'unknown' else pd.DataFrame()
            
            # Update scatter plot
            if len(class_targets) > 0:
                target_scatters[target_class].set_offsets(
                    class_targets[['longitude', 'latitude']].values
                )
            else:
                target_scatters[target_class].set_offsets(np.empty((0, 2)))
        
        # Remove old trails, predictions, and ellipses
        for target_id in list(target_trails.keys()):
            line = target_trails[target_id]
            if line in ax.lines:
                line.remove()
            del target_trails[target_id]
        
        for target_id in list(target_predictions.keys()):
            line = target_predictions[target_id]
            if line in ax.lines:
                line.remove()
            del target_predictions[target_id]
            
        for target_id in list(prediction_ellipses.keys()):
            ellipse = prediction_ellipses[target_id]
            if ellipse in ax.patches:
                ellipse.remove()
            del prediction_ellipses[target_id]
        
        # Add trails and generate predictions for each target
        if not latest_positions.empty:
            for target_id, target in latest_positions.iterrows():
                # Get target history
                target_history = current_data[current_data['target_id'] == target['target_id']]
                
                if len(target_history) >= 2:
                    # Get target class and color
                    target_class = target['target_class'].lower() if 'target_class' in target else 'unknown'
                    color = target_colors.get(target_class, 'red')
                    
                    # Create trail (past trajectory)
                    trail_points = target_history.sort_values('datetime')[['longitude', 'latitude']].values
                    line, = ax.plot(trail_points[:, 0], trail_points[:, 1], '-', 
                                  color=color, alpha=0.6, linewidth=1.5, zorder=5)
                    target_trails[target['target_id']] = line
                    
                    # Generate prediction if enough history
                    if len(target_history) >= predictor.config['sequence_length']:
                        try:
                            prediction = predictor.predict_out_of_view(
                                target_data,
                                target['target_id'],
                                timestamp,
                                prediction_duration
                            )
                            
                            if prediction is not None and len(prediction['mean']) > 0:
                                # Plot prediction line
                                pred_mean = prediction['mean']
                                pred_line, = ax.plot(pred_mean[:, 0], pred_mean[:, 1], '--', 
                                                   color=color, alpha=0.8, linewidth=1.5, zorder=6)
                                target_predictions[target['target_id']] = pred_line
                                
                                # Add confidence ellipse for final position
                                ellipse = Ellipse(
                                    (pred_mean[-1, 0], pred_mean[-1, 1]),
                                    width=prediction['upper_ci'][-1, 0] - prediction['lower_ci'][-1, 0],
                                    height=prediction['upper_ci'][-1, 1] - prediction['lower_ci'][-1, 1],
                                    color=color, alpha=0.2, zorder=4
                                )
                                ax.add_patch(ellipse)
                                prediction_ellipses[target['target_id']] = ellipse
                                
                                # Add end time annotation (only for some targets to avoid clutter)
                                if int(target_id) % 3 == 0:  # Show for every 3rd target
                                    end_time = prediction['time_points'][-1].strftime('%H:%M:%S')
                                    ax.annotate(end_time,
                                              (pred_mean[-1, 0], pred_mean[-1, 1]),
                                              fontsize=8, color='black',
                                              bbox=dict(facecolor='white', alpha=0.6),
                                              xytext=(5, 5), textcoords='offset points')
                        except Exception as e:
                            print(f"Error generating prediction for target {target['target_id']}: {e}")
        
        # Update time display
        time_text.set_text(f'Time: {pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Return updated artists
        artists = [blue_scatter] + list(target_scatters.values()) + [time_text]
        artists.extend(list(target_trails.values()))
        artists.extend(list(target_predictions.values()))
        artists.extend(list(prediction_ellipses.values()))
        return artists
    
    # Create animation
    animation = FuncAnimation(
        fig, update, frames=len(selected_timestamps),
        init_func=init, blit=True, interval=1000/fps
    )
    
    # Save animation
    print("Saving animation...")
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Target Movement Prediction'), bitrate=3600)
    
    with tqdm(total=100, desc="Encoding video") as pbar:
        animation.save(output_file, writer=writer, dpi=dpi,
                    progress_callback=lambda i, n: pbar.update(100/n))
    
    print(f"Animation saved to {output_file}")
    return animation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create target prediction animation')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing data files')
    parser.add_argument('--terrain_path', type=str, default='adapted_data/terrain_map.npy',
                      help='Path to terrain data')
    parser.add_argument('--elevation_path', type=str, default='adapted_data/elevation_map.npy',
                      help='Path to elevation data')
    parser.add_argument('--output_file', type=str, default='target_prediction_animation.mp4',
                      help='Output video file path')
    parser.add_argument('--fps', type=int, default=15,
                      help='Frames per second')
    parser.add_argument('--duration', type=int, default=5,
                      help='Duration of animation in seconds')
    parser.add_argument('--dpi', type=int, default=150,
                      help='Resolution of the animation')
    parser.add_argument('--prediction_duration', type=int, default=300,
                      help='How far to predict in seconds for each target')
    
    args = parser.parse_args()
    
    # Import predictor
    from target_movement_prediction import TargetMovementPredictor
    
    # Load model
    predictor = TargetMovementPredictor()
    if not predictor.load_model(args.model_path):
        print(f"Error: Could not load model from {args.model_path}")
        exit(1)
    
    # Create animation
    create_prediction_animation(
        predictor,
        os.path.join(args.data_dir, "red_sightings.csv"),
        os.path.join(args.data_dir, "blue_locations.csv"),
        args.terrain_path,
        args.elevation_path,
        args.output_file,
        fps=args.fps,
        duration_seconds=args.duration,
        dpi=args.dpi,
        prediction_duration=args.prediction_duration
    )