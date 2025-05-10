import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Ellipse
from tqdm import tqdm
import os
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

# Import your model and simulation classes
from model_jammer_prediction import JammerPositionPredictor, load_and_process_jammer_data
from battlefield_simulation import BattlefieldSimulation

def visualize_predictions_at_timestep(
    predictor, 
    jammer_data, 
    timestamp,
    prediction_horizon=5, 
    sequence_length=10,
    sim=None,
    output_dir="visualizations",
    filename=None
):
    """
    Create a static visualization of jammer predictions at a specific timestamp.
    
    Args:
        predictor: Trained JammerPositionPredictor
        jammer_data: DataFrame with jammer observations
        timestamp: Timestamp for visualization (datetime)
        prediction_horizon: Number of steps to predict ahead
        sequence_length: Length of input sequence
        sim: BattlefieldSimulation instance for terrain background
        output_dir: Directory to save visualizations
        filename: Filename for saved visualization (if None, generate automatically)
    
    Returns:
        Path to saved visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize simulation if not provided
    if sim is None:
        sim = BattlefieldSimulation()
        sim.load_terrain_data(
            terrain_data_path="simulation_data/terrain_map.npy",
            elevation_data_path="simulation_data/elevation_map.npy"
        )
    
    # Convert timestamp to pandas datetime if it's a string
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    
    # Find all observations at and before the specified timestamp
    jammer_data['timestamp'] = pd.to_datetime(jammer_data['timestamp'])
    
    # Create figure with terrain background
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot terrain background
    terrain_colors = [sim.TERRAIN_TYPES[i]['color'] for i in range(len(sim.TERRAIN_TYPES))]
    terrain_cmap = ListedColormap(terrain_colors)
    ax.imshow(sim.terrain_map.T, origin='lower', cmap=terrain_cmap, 
              vmin=0, vmax=len(sim.TERRAIN_TYPES)-1, alpha=0.7)
    
    # Get unique jammer IDs
    jammer_ids = jammer_data['id'].unique()
    
    # Colors for different jammers
    jammer_colors = cm.rainbow(np.linspace(0, 1, len(jammer_ids)))
    jammer_color_map = dict(zip(jammer_ids, jammer_colors))
    
    # Process each jammer
    for i, jammer_id in enumerate(jammer_ids):
        # Get jammer observations
        jammer_obs = jammer_data[jammer_data['id'] == jammer_id].sort_values('timestamp')
        
        # Skip if not enough observations before timestamp
        filtered_obs = jammer_obs[jammer_obs['timestamp'] <= timestamp]
        if len(filtered_obs) < sequence_length:
            continue
        
        # Get the most recent sequence_length observations
        input_sequence = filtered_obs.iloc[-sequence_length:]
        
        # Prepare input features for the model
        features = []
        for _, row in input_sequence.iterrows():
            # Extract relevant features from the jammer observation
            feature_vector = [
                row['x_coord'],
                row['y_coord'],
                row['power'],
                row['range'],
                row['direction']
            ]
            
            # One-hot encode jammer type if available
            if 'jammer_type' in row:
                jammer_types = jammer_data['jammer_type'].unique()
                one_hot = [1 if row['jammer_type'] == jt else 0 for jt in jammer_types]
                feature_vector.extend(one_hot)
            
            features.append(feature_vector)
        
        # Convert to tensor
        input_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Make prediction
        predictions = predictor.predict(input_tensor)
        
        # Extract predictions
        mean_predictions = predictions['mean']
        lower_ci = predictions['lower_ci']
        upper_ci = predictions['upper_ci']
        
        # Get actual future positions if available
        future_obs = jammer_obs[jammer_obs['timestamp'] > timestamp]
        actual_positions = []
        
        for step in range(1, prediction_horizon + 1):
            target_time = timestamp + timedelta(minutes=15 * step)  # Assuming 15-minute intervals
            future_at_time = future_obs[future_obs['timestamp'] == target_time]
            
            if not future_at_time.empty:
                x = future_at_time.iloc[0]['x_coord']
                y = future_at_time.iloc[0]['y_coord']
                actual_positions.append((x, y))
        
        # Convert current position to grid coordinates for plotting
        latest_pos = (input_sequence.iloc[-1]['x_coord'], input_sequence.iloc[-1]['y_coord'])
        current_grid_x, current_grid_y = sim.data_to_grid(*latest_pos)
        
        # Plot current position
        jammer_color = jammer_color_map[jammer_id]
        ax.scatter(current_grid_x, current_grid_y, color=jammer_color, s=100, marker='o', 
                   edgecolor='black', label=f"Jammer {jammer_id}")
        
        # Plot past trajectory
        past_positions = input_sequence[['x_coord', 'y_coord']].values
        past_grid_x, past_grid_y = zip(*[sim.data_to_grid(x, y) for x, y in past_positions])
        ax.plot(past_grid_x, past_grid_y, color=jammer_color, linestyle='-', alpha=0.5)
        
        # Plot predicted positions and confidence intervals
        for t in range(len(mean_predictions)):
            # Convert to grid coordinates
            pred_x, pred_y = mean_predictions[t]
            pred_grid_x, pred_grid_y = sim.data_to_grid(pred_x, pred_y)
            
            # Plot prediction
            alpha = 1.0 - 0.15 * t  # Fade alpha for later predictions
            ax.scatter(pred_grid_x, pred_grid_y, color=jammer_color, s=80, 
                       marker='x', alpha=alpha)
            
            # Plot confidence ellipse
            lower_x, lower_y = lower_ci[t]
            upper_x, upper_y = upper_ci[t]
            
            width = upper_x - lower_x
            height = upper_y - lower_y
            
            # Convert to grid coordinates
            lower_grid_x, lower_grid_y = sim.data_to_grid(lower_x, lower_y)
            upper_grid_x, upper_grid_y = sim.data_to_grid(upper_x, upper_y)
            width_grid = upper_grid_x - lower_grid_x
            height_grid = upper_grid_y - lower_grid_y
            
            ellipse = Ellipse(
                (pred_grid_x, pred_grid_y),
                width=width_grid, height=height_grid,
                alpha=0.2, color=jammer_color
            )
            ax.add_patch(ellipse)
            
            # Connect predictions with a line
            if t > 0:
                prev_x, prev_y = mean_predictions[t-1]
                prev_grid_x, prev_grid_y = sim.data_to_grid(prev_x, prev_y)
                ax.plot([prev_grid_x, pred_grid_x], [prev_grid_y, pred_grid_y], 
                        color=jammer_color, linestyle='--', alpha=0.7)
        
        # Plot actual future positions if available
        if actual_positions:
            actual_grid_positions = [sim.data_to_grid(x, y) for x, y in actual_positions]
            actual_grid_x, actual_grid_y = zip(*actual_grid_positions)
            ax.plot(actual_grid_x, actual_grid_y, color=jammer_color, linestyle='-', 
                    marker='*', markersize=10, alpha=0.7)
    
    # Add legend and title
    plt.title(f"Jammer Position Predictions at {timestamp}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Add legend with a descriptive indicator
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    # Add generic legend items
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Current Position'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markersize=10, label='Predicted Position'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=10, label='Actual Future Position')
    ]
    
    # Combine with jammer-specific legend items
    ax.legend(handles=unique_handles + legend_elements, loc='upper right')
    
    # Save figure
    if filename is None:
        filename = f"jammer_predictions_{timestamp.strftime('%Y%m%d_%H%M')}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return output_path

def create_prediction_animation(
    predictor,
    jammer_data, 
    start_time=None,
    duration_hours=6,
    interval_minutes=15,
    prediction_horizon=5,
    sequence_length=10,
    max_frames=200,
    sim=None,
    output_filename="jammer_prediction_animation.mp4",
    fps=10
):
    """
    Create an animation of jammer predictions over time.
    
    Args:
        predictor: Trained JammerPositionPredictor
        jammer_data: DataFrame with jammer observations
        start_time: Starting time for animation (defaults to earliest timestamp)
        duration_hours: Duration of animation in hours
        interval_minutes: Time between frames in minutes
        prediction_horizon: Number of time steps to predict ahead
        sequence_length: Number of past time steps to use as input
        max_frames: Maximum number of frames in animation
        sim: BattlefieldSimulation instance
        output_filename: Filename for output video
        fps: Frames per second for animation
        
    Returns:
        Animation object
    """
    # Initialize simulation if not provided
    if sim is None:
        sim = BattlefieldSimulation()
        sim.load_terrain_data(
            terrain_data_path="simulation_data/terrain_map.npy",
            elevation_data_path="simulation_data/elevation_map.npy"
        )
    
    # Ensure timestamp is datetime
    jammer_data['timestamp'] = pd.to_datetime(jammer_data['timestamp'])
    
    # Determine start time if not provided
    if start_time is None:
        start_time = jammer_data['timestamp'].min() + timedelta(hours=sequence_length * interval_minutes / 60)
    elif isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    
    # Determine end time
    end_time = start_time + timedelta(hours=duration_hours)
    
    # Generate timestamps for animation frames
    timestamps = []
    current_time = start_time
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += timedelta(minutes=interval_minutes)
    
    # Limit number of frames if necessary
    if len(timestamps) > max_frames:
        step = len(timestamps) // max_frames + 1
        timestamps = timestamps[::step]
    
    # Get unique jammer IDs
    jammer_ids = jammer_data['id'].unique()
    
    # Colors for different jammers
    jammer_colors = cm.rainbow(np.linspace(0, 1, len(jammer_ids)))
    jammer_color_map = dict(zip(jammer_ids, jammer_colors))
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot terrain background
    terrain_colors = [sim.TERRAIN_TYPES[i]['color'] for i in range(len(sim.TERRAIN_TYPES))]
    terrain_cmap = ListedColormap(terrain_colors)
    ax.imshow(sim.terrain_map.T, origin='lower', cmap=terrain_cmap, 
              vmin=0, vmax=len(sim.TERRAIN_TYPES)-1, alpha=0.7)
    
    # Initialize empty scatter plots and collections
    current_positions = {jammer_id: ax.scatter([], [], color=jammer_color_map[jammer_id], 
                                               s=100, marker='o', edgecolor='black') 
                         for jammer_id in jammer_ids}
    
    predicted_positions = {jammer_id: ax.scatter([], [], color=jammer_color_map[jammer_id], 
                                                s=80, marker='x') 
                          for jammer_id in jammer_ids}
    
    past_trajectories = {jammer_id: ax.plot([], [], color=jammer_color_map[jammer_id], 
                                           linestyle='-', alpha=0.5)[0]
                        for jammer_id in jammer_ids}
    
    prediction_lines = {jammer_id: ax.plot([], [], color=jammer_color_map[jammer_id], 
                                          linestyle='--', alpha=0.7)[0]
                       for jammer_id in jammer_ids}
    
    actual_trajectories = {jammer_id: ax.plot([], [], color=jammer_color_map[jammer_id], 
                                             linestyle='-', marker='*', markersize=10, alpha=0.7)[0]
                          for jammer_id in jammer_ids}
    
    # Text display for time
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Keep track of patches for confidence ellipses
    ellipse_patches = []
    
    # Update function for animation
    def update(frame):
        # Clear existing ellipses
        for patch in ellipse_patches:
            if patch in ax.patches:
                patch.remove()
        ellipse_patches.clear()
        
        # Get current timestamp
        timestamp = timestamps[frame]
        time_text.set_text(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        for jammer_id in jammer_ids:
            # Get jammer observations up to current timestamp
            jammer_obs = jammer_data[(jammer_data['id'] == jammer_id) & 
                                     (jammer_data['timestamp'] <= timestamp)]
            jammer_obs = jammer_obs.sort_values('timestamp')
            
            # Skip if not enough observations
            if len(jammer_obs) < sequence_length:
                # Hide visuals for this jammer
                current_positions[jammer_id].set_offsets(np.empty((0, 2)))
                predicted_positions[jammer_id].set_offsets(np.empty((0, 2)))
                past_trajectories[jammer_id].set_data([], [])
                prediction_lines[jammer_id].set_data([], [])
                actual_trajectories[jammer_id].set_data([], [])
                continue
            
            # Get the most recent sequence_length observations
            input_sequence = jammer_obs.iloc[-sequence_length:]
            
            # Prepare input features for the model
            features = []
            for _, row in input_sequence.iterrows():
                # Extract relevant features from the jammer observation
                feature_vector = [
                    row['x_coord'],
                    row['y_coord'],
                    row['power'],
                    row['range'],
                    row['direction']
                ]
                
                # One-hot encode jammer type if available
                if 'jammer_type' in row:
                    jammer_types = jammer_data['jammer_type'].unique()
                    one_hot = [1 if row['jammer_type'] == jt else 0 for jt in jammer_types]
                    feature_vector.extend(one_hot)
                
                features.append(feature_vector)
            
            # Convert to tensor
            input_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Make prediction
            predictions = predictor.predict(input_tensor)
            
            # Extract predictions
            mean_predictions = predictions['mean']
            lower_ci = predictions['lower_ci']
            upper_ci = predictions['upper_ci']
            
            # Get actual future positions if available
            future_obs = jammer_data[(jammer_data['id'] == jammer_id) & 
                                    (jammer_data['timestamp'] > timestamp)]
            actual_positions = []
            
            for step in range(1, prediction_horizon + 1):
                target_time = timestamp + timedelta(minutes=interval_minutes * step)
                future_at_time = future_obs[future_obs['timestamp'] == target_time]
                
                if not future_at_time.empty:
                    x = future_at_time.iloc[0]['x_coord']
                    y = future_at_time.iloc[0]['y_coord']
                    actual_positions.append((x, y))
            
            # Convert to grid coordinates for plotting
            
            # Current position
            latest_pos = (input_sequence.iloc[-1]['x_coord'], input_sequence.iloc[-1]['y_coord'])
            current_grid_x, current_grid_y = sim.data_to_grid(*latest_pos)
            current_positions[jammer_id].set_offsets(np.array([[current_grid_x, current_grid_y]]))
            
            # Past trajectory
            past_positions = input_sequence[['x_coord', 'y_coord']].values
            past_grid_x, past_grid_y = zip(*[sim.data_to_grid(x, y) for x, y in past_positions])
            past_trajectories[jammer_id].set_data(past_grid_x, past_grid_y)
            
            # Predicted positions
            if len(mean_predictions.shape) == 1:  # Single prediction
                pred_positions = [mean_predictions[:2]]  # Just take x, y coordinates
            else:
                pred_positions = mean_predictions
            
            pred_grid_positions = [sim.data_to_grid(x, y) for x, y in pred_positions]
            
            # Set predicted positions
            predicted_positions[jammer_id].set_offsets(np.array(pred_grid_positions))
            
            # Set prediction line
            if len(pred_grid_positions) > 1:
                pred_grid_x, pred_grid_y = zip(*pred_grid_positions)
                prediction_lines[jammer_id].set_data(
                    [current_grid_x] + list(pred_grid_x),
                    [current_grid_y] + list(pred_grid_y)
                )
            else:
                prediction_lines[jammer_id].set_data([], [])
            
            # Plot confidence ellipses
            for t in range(len(pred_positions)):
                # Lower and upper confidence bounds
                if len(lower_ci.shape) == 1:
                    lower_x, lower_y = lower_ci[:2]
                    upper_x, upper_y = upper_ci[:2]
                else:
                    lower_x, lower_y = lower_ci[t]
                    upper_x, upper_y = upper_ci[t]
                
                # Convert to grid coordinates
                lower_grid_x, lower_grid_y = sim.data_to_grid(lower_x, lower_y)
                upper_grid_x, upper_grid_y = sim.data_to_grid(upper_x, upper_y)
                width_grid = upper_grid_x - lower_grid_x
                height_grid = upper_grid_y - lower_grid_y
                
                # Create ellipse
                ellipse = Ellipse(
                    pred_grid_positions[t],
                    width=width_grid, height=height_grid,
                    alpha=0.2, color=jammer_color_map[jammer_id]
                )
                ax.add_patch(ellipse)
                ellipse_patches.append(ellipse)
            
            # Set actual future trajectory
            if actual_positions:
                actual_grid_positions = [sim.data_to_grid(x, y) for x, y in actual_positions]
                actual_grid_x, actual_grid_y = zip(*actual_grid_positions)
                actual_trajectories[jammer_id].set_data(actual_grid_x, actual_grid_y)
            else:
                actual_trajectories[jammer_id].set_data([], [])
        
        # Determine what to return
        artists = [time_text]
        artists.extend(current_positions.values())
        artists.extend(predicted_positions.values())
        artists.extend(past_trajectories.values())
        artists.extend(prediction_lines.values())
        artists.extend(actual_trajectories.values())
        artists.extend(ellipse_patches)
        
        return artists
    
    # Add title and labels
    plt.title("Jammer Position Predictions Over Time")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Current Position'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markersize=10, label='Predicted Position'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=10, label='Actual Position'),
        plt.Line2D([0], [0], linestyle='-', color='gray', label='Past Trajectory'),
        plt.Line2D([0], [0], linestyle='--', color='gray', label='Predicted Trajectory')
    ]
    
    # Add jammer IDs to legend
    for jammer_id, color in jammer_color_map.items():
        legend_elements.append(
            plt.Line2D([0], [0], linestyle='-', color=color, label=f"Jammer {jammer_id}")
        )
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(timestamps), interval=200, blit=True)
    
    # Save animation
    print(f"Creating animation with {len(timestamps)} frames...")
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=3600)
    ani.save(output_filename, writer=writer, dpi=150)
    print(f"Animation saved to {output_filename}")
    
    return ani

def load_jammer_model_for_visualization(model_path="models/jammer_predictor_model.pt"):
    """
    Load a trained jammer prediction model for visualization.
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        Loaded JammerPositionPredictor
    """
    from model_jammer_prediction import JammerPositionPredictor
    
    # Initialize the predictor
    predictor = JammerPositionPredictor(
        terrain_data_path="simulation_data/terrain_map.npy",
        elevation_data_path="simulation_data/elevation_map.npy"
    )
    
    try:
        import torch.serialization
        from sklearn.preprocessing import StandardScaler
        
        # For PyTorch 2.6+
        try:
            torch.serialization.add_safe_globals([StandardScaler])
            print("Added StandardScaler to safe globals")
        except (ImportError, AttributeError):
            pass
        
        # Try different loading methods
        try:
            loaded = predictor.load_model(model_path)
        except Exception as e:
            print(f"First load attempt failed: {e}")
            print("Trying to load with weights_only=False...")
            try:
                loaded = torch.load(model_path, map_location=predictor.config['device'], weights_only=False)
                
                # Manually load model components
                predictor.config.update(loaded['config'])
                predictor.input_scaler = loaded['input_scaler']
                predictor.output_scaler = loaded['output_scaler']
                
                # Initialize models if not already initialized
                if predictor.transformer_model is None:
                    input_dim = next(iter(loaded['transformer_state'].values())).shape[1]
                    predictor.build_models(input_dim)
                
                # Load weights
                predictor.transformer_model.load_state_dict(loaded['transformer_state'])
                
                if 'terrain_state' in loaded and predictor.terrain_model is not None:
                    predictor.terrain_model.load_state_dict(loaded['terrain_state'])
                
                loaded = True
            except Exception as e2:
                print(f"Second load attempt failed: {e2}")
                loaded = False
    except:
        # Fallback for critical failures
        print("Warning: Model loading failed. Initializing with default parameters.")
        loaded = False
    
    if not loaded:
        print("Failed to load model from", model_path)
        return None
    
    print("Model loaded successfully from", model_path)
    return predictor

# Main function to run the visualizations
def run_jammer_visualization():
    """
    Run the jammer prediction visualizations.
    """
    # Load predictor model
    predictor = load_jammer_model_for_visualization()
    if predictor is None:
        print("Failed to load predictor model. Exiting.")
        return
    
    # Load jammer data
    jammer_data = load_and_process_jammer_data("synthetic_data/jammer_observations.csv")
    if jammer_data is None:
        print("Failed to load jammer data. Exiting.")
        return
    
    # Load battlefield simulation for terrain
    sim = BattlefieldSimulation()
    sim.load_terrain_data(
        terrain_data_path="simulation_data/terrain_map.npy",
        elevation_data_path="simulation_data/elevation_map.npy"
    )
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Generate static visualizations for a few timestamps
    print("Generating static visualizations...")
    timestamps = sorted(jammer_data['timestamp'].unique())
    
    # Choose a few timestamps for visualization (25%, 50%, 75% through the data)
    timestamp_indices = [
        len(timestamps) // 4,
        len(timestamps) // 2,
        3 * len(timestamps) // 4
    ]
    
    for i, idx in enumerate(timestamp_indices):
        if idx < len(timestamps):
            timestamp = timestamps[idx]
            print(f"Creating visualization for timestamp {timestamp}")
            visualize_predictions_at_timestep(
                predictor=predictor,
                jammer_data=jammer_data,
                timestamp=timestamp,
                prediction_horizon=5,
                sequence_length=10,
                sim=sim,
                output_dir="visualizations",
                filename=f"jammer_predictions_{i+1}.png"
            )
    
    # Create animation
    print("Creating animation...")
    create_prediction_animation(
        predictor=predictor,
        jammer_data=jammer_data,
        start_time=None,  # Start from the beginning
        duration_hours=12,  # Show 12 hours of data
        interval_minutes=15,  # 15-minute intervals
        prediction_horizon=5,
        sequence_length=10,
        max_frames=200,  # Limit to 200 frames for performance
        sim=sim,
        output_filename="visualizations/jammer_prediction_animation.mp4",
        fps=10
    )
    
    print("Visualization complete!")

if __name__ == "__main__":
    run_jammer_visualization()
