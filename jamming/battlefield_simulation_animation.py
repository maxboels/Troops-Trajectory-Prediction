import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from battlefield_simulation import BattlefieldSimulation
from tqdm import tqdm
import time
from scipy.interpolate import interp1d
import matplotlib as mpl
from matplotlib.animation import FFMpegWriter
from datetime import timedelta

def create_trajectory_animation(output_filename='battlefield_animation.mp4', 
                               interpolation_steps=5,  # Number of frames between actual data points
                               max_frames=300):        # Maximum number of frames to render
    print("Starting animation creation process...")
    
    # Create progress bar for data loading
    loading_pbar = tqdm(total=4, desc="Loading data")
    
    # Load simulation data
    sim = BattlefieldSimulation()
    loading_pbar.update(1)
    
    sim.load_terrain_data(
        terrain_data_path="simulation_data/terrain_map.npy",
        elevation_data_path="simulation_data/elevation_map.npy"
    )
    loading_pbar.update(1)
    
    sim.load_observation_data(
        target_csv="synthetic_data/target_observations.csv",
        blue_force_csv="synthetic_data/blue_force_observations.csv"
    )
    loading_pbar.update(1)
    
    # Get all unique timestamps
    target_df = pd.read_csv("synthetic_data/target_observations.csv")
    target_df['timestamp'] = pd.to_datetime(target_df['timestamp'])
    
    # Load blue force data
    blue_force_df = pd.read_csv("synthetic_data/blue_force_observations.csv")
    blue_force_df['timestamp'] = pd.to_datetime(blue_force_df['timestamp'])
    
    # Get original timestamps (limit to reduce complexity)
    original_timestamps = sorted(target_df['timestamp'].unique())
    
    # Limit number of original timestamps to keep animation manageable
    if len(original_timestamps) > (max_frames // interpolation_steps):
        original_timestamps = original_timestamps[:(max_frames // interpolation_steps)]
    
    # Create interpolated timestamps
    interpolated_timestamps = []
    for i in range(len(original_timestamps) - 1):
        start_time = original_timestamps[i]
        end_time = original_timestamps[i + 1]
        
        # Calculate time difference and step size
        time_diff = (end_time - start_time).total_seconds()
        step_seconds = time_diff / interpolation_steps
        
        # Add interpolated timestamps
        for j in range(interpolation_steps):
            interp_time = start_time + timedelta(seconds=j * step_seconds)
            interpolated_timestamps.append(interp_time)
    
    # Add the last original timestamp
    interpolated_timestamps.append(original_timestamps[-1])
    
    # Limit total number of frames if needed
    if len(interpolated_timestamps) > max_frames:
        interpolated_timestamps = interpolated_timestamps[:max_frames]
    
    timestamps = interpolated_timestamps
    frame_count = len(timestamps)
    
    print(f"Original data has {len(original_timestamps)} timestamps (every 15 minutes)")
    print(f"After interpolation: {frame_count} frames (approximately every {15/interpolation_steps:.1f} minutes)")
    
    loading_pbar.update(1)
    loading_pbar.close()
    
    # Create figure and background
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot terrain background once
    terrain_colors = [sim.TERRAIN_TYPES[i]['color'] for i in range(len(sim.TERRAIN_TYPES))]
    from matplotlib.colors import ListedColormap
    terrain_cmap = ListedColormap(terrain_colors)
    ax.imshow(sim.terrain_map.T, origin='lower', cmap=terrain_cmap, 
              vmin=0, vmax=len(sim.TERRAIN_TYPES)-1, alpha=0.7)
    
    # Create empty scatter plots for different target types
    infantry_scatter = ax.scatter([], [], color='red', s=50, marker='s', label='Infantry')
    light_vehicle_scatter = ax.scatter([], [], color='orange', s=50, marker='^', label='Light Vehicle')
    heavy_vehicle_scatter = ax.scatter([], [], color='darkred', s=50, marker='D', label='Heavy Vehicle')
    uav_scatter = ax.scatter([], [], color='purple', s=50, marker='*', label='UAV')
    civilian_scatter = ax.scatter([], [], color='pink', s=50, marker='o', label='Civilian')
    
    # Create empty scatter plots for blue forces
    blue_infantry_scatter = ax.scatter([], [], color='blue', s=50, marker='s', label='Blue Infantry')
    blue_mechanized_scatter = ax.scatter([], [], color='blue', s=50, marker='^', label='Blue Mechanized')
    blue_recon_scatter = ax.scatter([], [], color='blue', s=50, marker='o', label='Blue Recon')
    blue_command_scatter = ax.scatter([], [], color='blue', s=50, marker='p', label='Blue Command')
    blue_uav_scatter = ax.scatter([], [], color='blue', s=50, marker='*', label='Blue UAV')
    
    # Create dictionaries to store interpolators for each entity
    target_interpolators = {}
    blue_force_interpolators = {}
    
    # Progress bar for preprocessing
    print("Preprocessing entity trajectories for interpolation...")
    
    # Process targets for interpolation
    target_ids = target_df['id'].unique()
    for target_id in tqdm(target_ids, desc="Creating target interpolators"):
        # Get all observations for this target
        target_observations = target_df[target_df['id'] == target_id].sort_values('timestamp')
        target_class = target_observations['target_class'].iloc[0]
        
        # Skip if fewer than 2 observations
        if len(target_observations) < 2:
            continue
            
        # Extract timestamps and positions
        obs_times = target_observations['timestamp'].values
        obs_times_numeric = [(t - original_timestamps[0]).total_seconds() for t in obs_times]
        x_coords = target_observations['x_coord'].values
        y_coords = target_observations['y_coord'].values
        
        # Create interpolators
        if len(obs_times) >= 4:
            # Use cubic interpolation if enough points
            x_interp = interp1d(obs_times_numeric, x_coords, kind='cubic', bounds_error=False, fill_value="extrapolate")
            y_interp = interp1d(obs_times_numeric, y_coords, kind='cubic', bounds_error=False, fill_value="extrapolate")
        else:
            # Fall back to linear interpolation
            x_interp = interp1d(obs_times_numeric, x_coords, kind='linear', bounds_error=False, fill_value="extrapolate")
            y_interp = interp1d(obs_times_numeric, y_coords, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # Store interpolators
        target_interpolators[target_id] = {
            'x_interp': x_interp,
            'y_interp': y_interp,
            'class': target_class,
            'start_time': obs_times[0],
            'end_time': obs_times[-1]
        }
    
    # Process blue forces for interpolation
    blue_force_ids = blue_force_df['id'].unique()
    for force_id in tqdm(blue_force_ids, desc="Creating blue force interpolators"):
        # Get all observations for this blue force
        force_observations = blue_force_df[blue_force_df['id'] == force_id].sort_values('timestamp')
        force_class = force_observations['force_class'].iloc[0]
        
        # Skip if fewer than 2 observations
        if len(force_observations) < 2:
            continue
            
        # Extract timestamps and positions
        obs_times = force_observations['timestamp'].values
        obs_times_numeric = [(t - original_timestamps[0]).total_seconds() for t in obs_times]
        x_coords = force_observations['x_coord'].values
        y_coords = force_observations['y_coord'].values
        
        # Create interpolators
        if len(obs_times) >= 4:
            # Use cubic interpolation if enough points
            x_interp = interp1d(obs_times_numeric, x_coords, kind='cubic', bounds_error=False, fill_value="extrapolate")
            y_interp = interp1d(obs_times_numeric, y_coords, kind='cubic', bounds_error=False, fill_value="extrapolate")
        else:
            # Fall back to linear interpolation
            x_interp = interp1d(obs_times_numeric, x_coords, kind='linear', bounds_error=False, fill_value="extrapolate")
            y_interp = interp1d(obs_times_numeric, y_coords, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # Store interpolators
        blue_force_interpolators[force_id] = {
            'x_interp': x_interp,
            'y_interp': y_interp,
            'class': force_class,
            'start_time': obs_times[0],
            'end_time': obs_times[-1]
        }
    
    # Add trails for selected targets (optional)
    trail_lines = {}  # Store trail lines for tracking movement history
    trail_length = 10  # Number of previous positions to show
    
    # Time indicator text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add terrain label in the corner
    terrain_text = ax.text(0.02, 0.02, 'Terrain speed factors:\nWater: 0.0 (impassable)\nForest: 0.7\nUrban: 0.8\nGrassland: 1.0',
                         transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title('Battlefield Simulation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    
    # Initialize the progress bar for frame updates
    progress_bar = tqdm(total=frame_count, desc="Generating frames")
    
    # Dictionary to store trail history
    trail_history = {}
    
    def update(frame):
        # Get current interpolated timestamp
        current_time = timestamps[frame]
        current_time_numeric = (current_time - original_timestamps[0]).total_seconds()
        
        # Prepare storage for positions of each entity type
        infantry_positions = []
        light_vehicle_positions = []
        heavy_vehicle_positions = []
        uav_positions = []
        civilian_positions = []
        
        blue_infantry_positions = []
        blue_mechanized_positions = []
        blue_recon_positions = []
        blue_command_positions = []
        blue_uav_positions = []
        
        # Update target positions using interpolation
        for target_id, interp_data in target_interpolators.items():
            # Skip if target not active at this time
            if current_time < interp_data['start_time'] or current_time > interp_data['end_time']:
                continue
                
            # Get interpolated position
            x = interp_data['x_interp'](current_time_numeric)
            y = interp_data['y_interp'](current_time_numeric)
            
            # Convert to grid coordinates
            grid_x, grid_y = sim.data_to_grid(x, y)
            
            # Add to appropriate list based on class
            if interp_data['class'] == 'infantry':
                infantry_positions.append((grid_x, grid_y))
            elif interp_data['class'] == 'light_vehicle':
                light_vehicle_positions.append((grid_x, grid_y))
            elif interp_data['class'] == 'heavy_vehicle':
                heavy_vehicle_positions.append((grid_x, grid_y))
            elif interp_data['class'] == 'uav':
                uav_positions.append((grid_x, grid_y))
            elif interp_data['class'] == 'civilian':
                civilian_positions.append((grid_x, grid_y))
            
            # Update trail history for this target
            if target_id not in trail_history:
                trail_history[target_id] = []
            
            # Add current position to trail
            trail_history[target_id].append((grid_x, grid_y))
            
            # Keep only the most recent positions for the trail
            if len(trail_history[target_id]) > trail_length:
                trail_history[target_id] = trail_history[target_id][-trail_length:]
        
        # Update blue force positions using interpolation
        for force_id, interp_data in blue_force_interpolators.items():
            # Skip if force not active at this time
            if current_time < interp_data['start_time'] or current_time > interp_data['end_time']:
                continue
                
            # Get interpolated position
            x = interp_data['x_interp'](current_time_numeric)
            y = interp_data['y_interp'](current_time_numeric)
            
            # Convert to grid coordinates
            grid_x, grid_y = sim.data_to_grid(x, y)
            
            # Add to appropriate list based on class
            if interp_data['class'] == 'infantry_squad':
                blue_infantry_positions.append((grid_x, grid_y))
            elif interp_data['class'] == 'mechanized_patrol':
                blue_mechanized_positions.append((grid_x, grid_y))
            elif interp_data['class'] == 'recon_team':
                blue_recon_positions.append((grid_x, grid_y))
            elif interp_data['class'] == 'command_post':
                blue_command_positions.append((grid_x, grid_y))
            elif interp_data['class'] == 'uav_surveillance':
                blue_uav_positions.append((grid_x, grid_y))
        
        # Update scatter plots
        def update_scatter(scatter, positions):
            if positions:
                x, y = zip(*positions)
                scatter.set_offsets(np.column_stack([x, y]))
            else:
                scatter.set_offsets(np.empty((0, 2)))
        
        update_scatter(infantry_scatter, infantry_positions)
        update_scatter(light_vehicle_scatter, light_vehicle_positions)
        update_scatter(heavy_vehicle_scatter, heavy_vehicle_positions)
        update_scatter(uav_scatter, uav_positions)
        update_scatter(civilian_scatter, civilian_positions)
        
        update_scatter(blue_infantry_scatter, blue_infantry_positions)
        update_scatter(blue_mechanized_scatter, blue_mechanized_positions)
        update_scatter(blue_recon_scatter, blue_recon_positions)
        update_scatter(blue_command_scatter, blue_command_positions)
        update_scatter(blue_uav_scatter, blue_uav_positions)
        
        # Update time text
        time_text.set_text(f'Time: {current_time.strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Draw trails (optional)
        for line_id in list(trail_lines.keys()):
            if line_id in ax.lines:
                ax.lines.remove(trail_lines[line_id])
                del trail_lines[line_id]
        
        # Add trails for a subset of entities (for visual clarity)
        for target_id, trail in list(trail_history.items())[:10]:  # Limit number of trails
            if len(trail) >= 2:
                x, y = zip(*trail)
                line, = ax.plot(x, y, '-', alpha=0.5, linewidth=1, color='grey')
                trail_lines[target_id] = line
        
        # Update progress bar
        progress_bar.update(1)
        
        return [infantry_scatter, light_vehicle_scatter, heavy_vehicle_scatter, 
                uav_scatter, civilian_scatter, blue_infantry_scatter, 
                blue_mechanized_scatter, blue_recon_scatter, blue_command_scatter, 
                blue_uav_scatter, time_text]
    
    # Create animation with a faster frame rate for smoother appearance
    ani = FuncAnimation(fig, update, frames=frame_count, 
                       blit=True, interval=100, repeat=False)
    
    # Create a custom writer
    print("\nSaving animation...")
    
    # Use higher fps for smoother animation
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=3600)
    
    # Save the animation with progress tracking
    with tqdm(total=100, desc="Encoding video") as pbar:
        # Start the saving process
        ani.save(output_filename, writer=writer, dpi=150, 
                 progress_callback=lambda i, n: pbar.update(100/n))
    
    progress_bar.close()
    print(f"Animation saved to {output_filename}")
    
    return ani

if __name__ == "__main__":
    # Call the function to create the animation
    animation = create_trajectory_animation(
        output_filename='battlefield_animation_smooth.mp4',
        interpolation_steps=18,     # Create 6 frames between each real data point (2.5 minute intervals)
        max_frames=300              # Limit total animation length
    )
    plt.show()