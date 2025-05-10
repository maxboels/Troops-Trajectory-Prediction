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
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection
import os
import math

def create_trajectory_animation_with_jammers(
    output_filename='battlefield_animation_with_jammers.mp4', 
    interpolation_steps=5,         # Number of frames between actual data points
    max_frames=300,                # Maximum number of frames to render
    show_jammers=True,             # Whether to display jammers
    show_jamming_effects=True,     # Whether to display jamming effects
    jammer_opacity=0.3,            # Opacity for jammer coverage visualization
    highlight_jammed_entities=True # Whether to highlight jammed entities
):
    """
    Create an animation of battlefield entities and jammers.
    
    Args:
        output_filename: Path to save the animation
        interpolation_steps: Number of frames to interpolate between actual data points
        max_frames: Maximum number of frames to render
        show_jammers: Whether to display jammers
        show_jamming_effects: Whether to display jamming effects
        jammer_opacity: Opacity for jammer coverage visualization
        highlight_jammed_entities: Whether to highlight jammed entities
        
    Returns:
        Animation object
    """
    print("Starting animation creation process with jammers...")
    
    # Create progress bar for data loading
    loading_pbar = tqdm(total=6, desc="Loading data")
    
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
    
    # Load jammer data if requested
    jammer_data = None
    jamming_effects = None
    
    if show_jammers:
        jammer_csv = "synthetic_data/jammer_observations.csv"
        if os.path.exists(jammer_csv):
            jammer_data = pd.read_csv(jammer_csv)
            jammer_data['timestamp'] = pd.to_datetime(jammer_data['timestamp'])
            print(f"Loaded {len(jammer_data)} jammer observations")
        else:
            print(f"Warning: Jammer data file not found at {jammer_csv}")
            show_jammers = False
    
    if show_jamming_effects:
        effects_csv = "synthetic_data/jamming_effects.csv"
        if os.path.exists(effects_csv):
            jamming_effects = pd.read_csv(effects_csv)
            jamming_effects['timestamp'] = pd.to_datetime(jamming_effects['timestamp'])
            print(f"Loaded {len(jamming_effects)} jamming effects records")
        else:
            print(f"Warning: Jamming effects file not found at {effects_csv}")
            show_jamming_effects = False
    
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
    fig, ax = plt.subplots(figsize=(14, 12))
    
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
    
    # Create empty scatter plots for jammers
    if show_jammers:
        jammer_static_scatter = ax.scatter([], [], color='darkorange', s=80, marker='X', 
                                        label='Static Jammer', zorder=15)
        jammer_mobile_scatter = ax.scatter([], [], color='magenta', s=80, marker='X', 
                                        label='Mobile Jammer', zorder=15)
    
    # Create dictionaries to store interpolators for each entity
    target_interpolators = {}
    blue_force_interpolators = {}
    jammer_interpolators = {}
    
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
    
    # Process jammers for interpolation if available
    if show_jammers and jammer_data is not None:
        jammer_ids = jammer_data['id'].unique()
        for jammer_id in tqdm(jammer_ids, desc="Creating jammer interpolators"):
            # Get all observations for this jammer
            jammer_observations = jammer_data[jammer_data['id'] == jammer_id].sort_values('timestamp')
            jammer_type = jammer_observations['jammer_type'].iloc[0]
            jammer_mobility = jammer_observations['mobility'].iloc[0]
            
            # Skip if fewer than 2 observations
            if len(jammer_observations) < 2:
                continue
                
            # Extract timestamps and positions
            obs_times = jammer_observations['timestamp'].values
            obs_times_numeric = [(t - original_timestamps[0]).total_seconds() for t in obs_times]
            x_coords = jammer_observations['x_coord'].values
            y_coords = jammer_observations['y_coord'].values
            
            # Extract other properties for interpolation
            ranges = jammer_observations['range'].values
            directions = jammer_observations['direction'].values
            powers = jammer_observations['power'].values
            
            # Create interpolators
            if len(obs_times) >= 4:
                # Use cubic interpolation if enough points
                x_interp = interp1d(obs_times_numeric, x_coords, kind='cubic', bounds_error=False, fill_value="extrapolate")
                y_interp = interp1d(obs_times_numeric, y_coords, kind='cubic', bounds_error=False, fill_value="extrapolate")
                range_interp = interp1d(obs_times_numeric, ranges, kind='linear', bounds_error=False, fill_value="extrapolate")
                dir_interp = interp1d(obs_times_numeric, directions, kind='linear', bounds_error=False, fill_value="extrapolate")
                power_interp = interp1d(obs_times_numeric, powers, kind='linear', bounds_error=False, fill_value="extrapolate")
            else:
                # Fall back to linear interpolation
                x_interp = interp1d(obs_times_numeric, x_coords, kind='linear', bounds_error=False, fill_value="extrapolate")
                y_interp = interp1d(obs_times_numeric, y_coords, kind='linear', bounds_error=False, fill_value="extrapolate")
                range_interp = interp1d(obs_times_numeric, ranges, kind='linear', bounds_error=False, fill_value="extrapolate")
                dir_interp = interp1d(obs_times_numeric, directions, kind='linear', bounds_error=False, fill_value="extrapolate")
                power_interp = interp1d(obs_times_numeric, powers, kind='linear', bounds_error=False, fill_value="extrapolate")
            
            # Get additional properties
            angle = jammer_observations['angle'].iloc[0]  # Assuming angle doesn't change
            freq_low = jammer_observations['freq_low'].iloc[0]
            freq_high = jammer_observations['freq_high'].iloc[0]
            
            # Store interpolators
            jammer_interpolators[jammer_id] = {
                'x_interp': x_interp,
                'y_interp': y_interp,
                'range_interp': range_interp,
                'direction_interp': dir_interp,
                'power_interp': power_interp,
                'type': jammer_type,
                'mobility': jammer_mobility,
                'angle': angle,
                'freq_low': freq_low,
                'freq_high': freq_high,
                'start_time': obs_times[0],
                'end_time': obs_times[-1]
            }
    
    # Map colors for jammer visualization
    jammer_colors = {
        'static_barrage': 'darkorange',
        'static_directional': 'crimson',
        'vehicle_tactical': 'purple',
        'portable_reactive': 'magenta',
        'drone_jammer': 'darkviolet'
    }
    
    # Add trails for selected targets (optional)
    trail_lines = {}  # Store trail lines for tracking movement history
    trail_length = 10  # Number of previous positions to show
    
    # Time indicator text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Jammer info text
    if show_jammers:
        jammer_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add terrain label in the corner
    terrain_text = ax.text(0.02, 0.02, 'Terrain speed factors:\nWater: 0.0 (impassable)\nForest: 0.7\nUrban: 0.8\nGrassland: 1.0',
                         transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title('Battlefield Simulation with Jammers')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    
    # Initialize the progress bar for frame updates
    progress_bar = tqdm(total=frame_count, desc="Generating frames")
    
    # Dictionary to store trail history
    trail_history = {}
    
    # Store jammer patches for each frame
    jammer_patches = []
    
    def update(frame):
        # Clear previous jammer patches
        for patch in jammer_patches:
            if patch in ax.patches:
                patch.remove()
        jammer_patches.clear()
        
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
        
        jammer_static_positions = []
        jammer_mobile_positions = []
        
        # Store jammed entities for highlighting
        jammed_entities = set()
        if show_jamming_effects and jamming_effects is not None and highlight_jammed_entities:
            # Find entities being jammed at this time
            effects_at_time = jamming_effects[jamming_effects['timestamp'] == current_time]
            for _, effect in effects_at_time.iterrows():
                if effect['jamming_effect'] > 0.1:  # Only count significant jamming
                    jammed_entities.add(effect['affected_id'])
        
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
            
            # Check if this entity is jammed
            is_jammed = target_id in jammed_entities
            
            # Add marker size increase if jammed
            marker_size = 80 if is_jammed else 50
            
            # Add to appropriate list based on class
            if interp_data['class'] == 'infantry':
                infantry_positions.append((grid_x, grid_y, marker_size))
            elif interp_data['class'] == 'light_vehicle':
                light_vehicle_positions.append((grid_x, grid_y, marker_size))
            elif interp_data['class'] == 'heavy_vehicle':
                heavy_vehicle_positions.append((grid_x, grid_y, marker_size))
            elif interp_data['class'] == 'uav':
                uav_positions.append((grid_x, grid_y, marker_size))
            elif interp_data['class'] == 'civilian':
                civilian_positions.append((grid_x, grid_y, marker_size))
            
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
            
            # Check if this entity is jammed
            is_jammed = force_id in jammed_entities
            
            # Add marker size increase if jammed
            marker_size = 80 if is_jammed else 50
            
            # Add to appropriate list based on class
            if interp_data['class'] == 'infantry_squad':
                blue_infantry_positions.append((grid_x, grid_y, marker_size))
            elif interp_data['class'] == 'mechanized_patrol':
                blue_mechanized_positions.append((grid_x, grid_y, marker_size))
            elif interp_data['class'] == 'recon_team':
                blue_recon_positions.append((grid_x, grid_y, marker_size))
            elif interp_data['class'] == 'command_post':
                blue_command_positions.append((grid_x, grid_y, marker_size))
            elif interp_data['class'] == 'uav_surveillance':
                blue_uav_positions.append((grid_x, grid_y, marker_size))
        
        # Update jammer positions if showing jammers
        active_jammers_count = 0
        if show_jammers and jammer_interpolators:
            for jammer_id, interp_data in jammer_interpolators.items():
                # Skip if jammer not active at this time
                if current_time < interp_data['start_time'] or current_time > interp_data['end_time']:
                    continue
                
                active_jammers_count += 1
                
                # Get interpolated position and properties
                x = interp_data['x_interp'](current_time_numeric)
                y = interp_data['y_interp'](current_time_numeric)
                jammer_range = interp_data['range_interp'](current_time_numeric)
                direction = interp_data['direction_interp'](current_time_numeric)
                power = interp_data['power_interp'](current_time_numeric)
                
                # Convert to grid coordinates
                grid_x, grid_y = sim.data_to_grid(x, y)
                grid_range = jammer_range * sim.x_scale  # Convert range to grid units
                
                # Add to appropriate list based on mobility
                if interp_data['mobility'] == 'static':
                    jammer_static_positions.append((grid_x, grid_y))
                else:
                    jammer_mobile_positions.append((grid_x, grid_y))
                
                # Draw jammer coverage area
                jammer_color = jammer_colors.get(interp_data['type'], 'red')
                
                # If directional (angle < 360), create a wedge
                if interp_data['angle'] < 360:
                    # Calculate start and end angles
                    theta1 = direction - interp_data['angle'] / 2
                    theta2 = direction + interp_data['angle'] / 2
                    
                    # Create wedge
                    wedge = Wedge((grid_x, grid_y), grid_range, theta1, theta2,
                                  alpha=jammer_opacity, color=jammer_color)
                    ax.add_patch(wedge)
                    jammer_patches.append(wedge)
                else:
                    # Create circle for omnidirectional
                    circle = Circle((grid_x, grid_y), grid_range,
                                   alpha=jammer_opacity, color=jammer_color)
                    ax.add_patch(circle)
                    jammer_patches.append(circle)
        
        # Update scatter plots
        def update_scatter_with_sizes(scatter, positions):
            if positions:
                x, y, sizes = zip(*positions)
                scatter.set_offsets(np.column_stack([x, y]))
                scatter.set_sizes(sizes)
            else:
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_sizes([])
        
        def update_scatter(scatter, positions):
            if positions:
                x, y = zip(*positions)
                scatter.set_offsets(np.column_stack([x, y]))
            else:
                scatter.set_offsets(np.empty((0, 2)))
        
        # Update with variable sizes for highlighting jammed entities
        update_scatter_with_sizes(infantry_scatter, infantry_positions)
        update_scatter_with_sizes(light_vehicle_scatter, light_vehicle_positions)
        update_scatter_with_sizes(heavy_vehicle_scatter, heavy_vehicle_positions)
        update_scatter_with_sizes(uav_scatter, uav_positions)
        update_scatter_with_sizes(civilian_scatter, civilian_positions)
        
        update_scatter_with_sizes(blue_infantry_scatter, blue_infantry_positions)
        update_scatter_with_sizes(blue_mechanized_scatter, blue_mechanized_positions)
        update_scatter_with_sizes(blue_recon_scatter, blue_recon_positions)
        update_scatter_with_sizes(blue_command_scatter, blue_command_positions)
        update_scatter_with_sizes(blue_uav_scatter, blue_uav_positions)
        
        if show_jammers:
            update_scatter(jammer_static_scatter, jammer_static_positions)
            update_scatter(jammer_mobile_scatter, jammer_mobile_positions)
            
            # Update jammer info text
            jammer_text.set_text(f"Active Jammers: {active_jammers_count}\nJammed Entities: {len(jammed_entities)}")
        
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
        
        # Return all artists that need to be redrawn
        artists = [
            infantry_scatter, light_vehicle_scatter, heavy_vehicle_scatter, 
            uav_scatter, civilian_scatter, blue_infantry_scatter, 
            blue_mechanized_scatter, blue_recon_scatter, blue_command_scatter, 
            blue_uav_scatter, time_text
        ]
        
        if show_jammers:
            artists.extend([jammer_static_scatter, jammer_mobile_scatter, jammer_text])
        
        return artists
    
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

def create_jamming_impact_visualization(
    output_dir="visualizations",
    jammer_csv="synthetic_data/jammer_observations.csv",
    effects_csv="synthetic_data/jamming_effects.csv"
):
    """
    Create visualizations showing the impact of jamming over time.
    
    Args:
        output_dir: Directory to save visualizations
        jammer_csv: Path to jammer observations CSV
        effects_csv: Path to jamming effects CSV
        
    Returns:
        Dictionary of file paths to saved visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    # Check if required files exist
    if not os.path.exists(effects_csv):
        print(f"Jamming effects file not found: {effects_csv}")
        return saved_files
    
    # Load data
    effects_df = pd.read_csv(effects_csv)
    effects_df['timestamp'] = pd.to_datetime(effects_df['timestamp'])
    
    # 1. Plot jamming effect over time
    plt.figure(figsize=(12, 6))
    
    # Group by timestamp and calculate mean effect
    time_effects = effects_df.groupby(pd.Grouper(key='timestamp', freq='h'))['jamming_effect'].mean()
    
    plt.plot(time_effects.index, time_effects.values, '-o', markersize=4)
    plt.title('Mean Jamming Effect Over Time')
    plt.xlabel('Time')
    plt.ylabel('Mean Jamming Effect')
    plt.grid(True)
    
    # Save figure
    time_plot_path = os.path.join(output_dir, 'jamming_effect_over_time.png')
    plt.savefig(time_plot_path, dpi=300)
    plt.close()
    saved_files['time_plot'] = time_plot_path
    
    # 2. Plot effect by entity type
    plt.figure(figsize=(12, 6))
    
    # Group by entity type
    entity_effects = effects_df.groupby('affected_type')['jamming_effect'].mean()
    
    plt.bar(entity_effects.index, entity_effects.values, color=['red', 'blue'])
    plt.title('Mean Jamming Effect by Entity Type')
    plt.xlabel('Entity Type')
    plt.ylabel('Mean Jamming Effect')
    plt.ylim(0, 1)
    
    # Save figure
    entity_plot_path = os.path.join(output_dir, 'jamming_effect_by_entity.png')
    plt.savefig(entity_plot_path, dpi=300)
    plt.close()
    saved_files['entity_plot'] = entity_plot_path
    
    # 3. Plot detection confidence degradation (for targets only)
    target_effects = effects_df[effects_df['affected_type'] == 'target']
    
    if 'original_detection_confidence' in target_effects.columns and 'degraded_detection_confidence' in target_effects.columns:
        plt.figure(figsize=(12, 6))
        
        # Calculate degradation
        target_effects['degradation'] = target_effects['original_detection_confidence'] - target_effects['degraded_detection_confidence']
        
        # Group by timestamp
        time_degradation = target_effects.groupby(pd.Grouper(key='timestamp', freq='h'))['degradation'].mean()
        
        plt.plot(time_degradation.index, time_degradation.values, '-o', markersize=4, color='orange')
        plt.title('Detection Confidence Degradation Over Time')
        plt.xlabel('Time')
        plt.ylabel('Mean Confidence Degradation')
        plt.grid(True)
        
        # Save figure
        degradation_plot_path = os.path.join(output_dir, 'detection_degradation_over_time.png')
        plt.savefig(degradation_plot_path, dpi=300)
        plt.close()
        saved_files['degradation_plot'] = degradation_plot_path
    
    # 4. Create a heatmap of jamming effect vs. distance
    if 'distance' in effects_df.columns:
        plt.figure(figsize=(12, 6))
        
        # Create distance bins
        max_distance = effects_df['distance'].max()
        bins = 20
        hist, xedges, yedges = np.histogram2d(
            effects_df['distance'], 
            effects_df['jamming_effect'], 
            bins=[bins, bins],
            range=[[0, max_distance], [0, 1]]
        )
        
        # Plot heatmap
        plt.imshow(
            hist.T, 
            origin='lower', 
            aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap='hot_r'
        )
        
        plt.colorbar(label='Number of Observations')
        plt.title('Jamming Effect vs. Distance')
        plt.xlabel('Distance from Jammer (meters)')
        plt.ylabel('Jamming Effect')
        
        # Add trend line
        try:
            # Use polynomial fit
            z = np.polyfit(effects_df['distance'], effects_df['jamming_effect'], 2)
            p = np.poly1d(z)
            
            # Create x values for the line
            x_line = np.linspace(0, max_distance, 100)
            y_line = p(x_line)
            
            # Plot the line
            plt.plot(x_line, y_line, 'b-', linewidth=2)
        except:
            # Skip trend line if fitting fails
            pass
        
        # Save figure
        distance_plot_path = os.path.join(output_dir, 'jamming_effect_vs_distance.png')
        plt.savefig(distance_plot_path, dpi=300)
        plt.close()
        saved_files['distance_plot'] = distance_plot_path
    
    print(f"Created {len(saved_files)} jamming impact visualizations in {output_dir}")
    return saved_files

if __name__ == "__main__":
    # Call the function to create the animation with jammers
    animation = create_trajectory_animation_with_jammers(
        output_filename='battlefield_animation_with_jammers.mp4',
        interpolation_steps=5,
        max_frames=300,
        show_jammers=True,
        show_jamming_effects=True
    )
    
    # Create additional impact visualizations
    create_jamming_impact_visualization()
    
    plt.close('all')  # Close all figures
