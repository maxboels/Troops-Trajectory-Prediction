"""
Enhanced battlefield visualization script with progress bars
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LightSource
import os
import time
from tqdm import tqdm
import warnings

# Import your battlefield simulation class
from battlefield_simulation import BattlefieldSimulation

# Suppress specific matplotlib warnings that might clutter output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def visualize_terrain_with_progress(sim, show_elevation=True, figsize=(15, 12), save_path="terrain_visualization.png"):
    """
    Visualize terrain with elevation contours and progress indicators
    """
    print("Generating terrain visualization...")
    
    # Start progress
    pbar = tqdm(total=5, desc="Terrain visualization steps")
    
    plt.figure(figsize=figsize)
    pbar.update(1)  # Step 1: Figure created
    
    # Create custom colormap for terrain
    terrain_colors = [sim.TERRAIN_TYPES[i]['color'] for i in range(len(sim.TERRAIN_TYPES))]
    terrain_cmap = mcolors.ListedColormap(terrain_colors)
    pbar.update(1)  # Step 2: Colormap created
    
    # Plot terrain
    plt.imshow(sim.terrain_map.T, origin='lower', cmap=terrain_cmap, 
              vmin=0, vmax=len(sim.TERRAIN_TYPES)-1)
    pbar.update(1)  # Step 3: Terrain plotted
    
    # Add elevation contours if requested
    if show_elevation and sim.elevation_map is not None:
        pbar.set_description("Generating elevation contours (may take time for large maps)")
        
        try:
            # Sample elevation map if it's very large to speed up contour generation
            elevation = sim.elevation_map
            if elevation.shape[0] > 500:  # If larger than 500x500, sample it
                sample_factor = max(1, elevation.shape[0] // 500)
                elevation_sampled = elevation[::sample_factor, ::sample_factor]
                print(f"Sampling elevation map from {elevation.shape} to {elevation_sampled.shape} for contours")
                
                # Calculate contour levels based on the full data range
                min_elev = np.min(elevation)
                max_elev = np.max(elevation)
                levels = np.linspace(min_elev, max_elev, 10)
                
                # Plot contours using the sampled data
                contour = plt.contour(
                    np.arange(0, elevation.shape[1], sample_factor),
                    np.arange(0, elevation.shape[0], sample_factor),
                    elevation_sampled.T, 
                    levels=levels, colors='black', alpha=0.5, linewidths=0.5
                )
                
                # Add contour labels but limit the number to avoid clutter
                plt.clabel(contour, inline=True, fontsize=8, fmt='%1.0f', 
                          use_clabeltext=True, levels=levels[::2])
            else:
                # For smaller maps, use the full resolution
                levels = np.linspace(np.min(elevation), np.max(elevation), 10)
                contour = plt.contour(elevation.T, levels=levels, colors='black', alpha=0.5, linewidths=0.5)
                plt.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')
                
        except Exception as e:
            print(f"Warning: Could not generate elevation contours: {e}")
            print("Continuing without contours...")
    
    pbar.update(1)  # Step 4: Contours added (or skipped)
    
    # Create legend for terrain types
    legend_patches = []
    for i, info in sim.TERRAIN_TYPES.items():
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=info['color'])
        legend_patches.append(patch)
    
    plt.legend(legend_patches, [info['name'] for info in sim.TERRAIN_TYPES.values()], 
              loc='upper right', title='Terrain Types')
    
    # Add elevation range information
    if sim.elevation_map is not None:
        min_elev = np.min(sim.elevation_map)
        max_elev = np.max(sim.elevation_map)
        plt.title(f'Battlefield Terrain Map (Elevation range: {min_elev:.0f}m - {max_elev:.0f}m)')
    else:
        plt.title('Battlefield Terrain Map')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    pbar.update(1)  # Step 5: Legend and labels added

    # Save figure
    plt.savefig(save_path, dpi=300)
    print(f"Terrain visualization saved to {save_path}")
    pbar.close()
    plt.close()

def visualize_elevation_separately(sim, figsize=(15, 12), save_path="elevation_visualization.png"):
    """
    Create a dedicated elevation map visualization with 3D effect
    """
    if sim.elevation_map is None:
        print("No elevation data available")
        return
    
    print("Generating 3D elevation visualization...")
    pbar = tqdm(total=5, desc="Elevation visualization steps")
    
    plt.figure(figsize=figsize)
    pbar.update(1)  # Step 1: Figure created
    
    # Create a light source for 3D effect
    ls = LightSource(azdeg=315, altdeg=45)
    pbar.update(1)  # Step 2: Light source created
    
    # Create a more interesting terrain colormap
    terrain_cmap = cm.terrain
    
    # Get elevation data
    elevation = sim.elevation_map
    
    # Apply hillshade effect for 3D appearance
    pbar.set_description("Rendering 3D hillshade (may take time for large maps)")
    
    try:
        # Sample if very large
        if elevation.shape[0] > 1000:
            sample_factor = max(1, elevation.shape[0] // 1000)
            elevation_sampled = elevation[::sample_factor, ::sample_factor]
            print(f"Sampling elevation map from {elevation.shape} to {elevation_sampled.shape} for hillshade")
            
            # Create hillshade with sampled data
            rgb = ls.shade(elevation_sampled, cmap=terrain_cmap, blend_mode='soft', vert_exag=0.3)
            
            # Plot the sampled data
            plt.imshow(rgb, extent=[0, elevation.shape[1], 0, elevation.shape[0]])
        else:
            # Use full resolution for smaller maps
            rgb = ls.shade(elevation, cmap=terrain_cmap, blend_mode='soft', vert_exag=0.3)
            plt.imshow(rgb)
    except Exception as e:
        print(f"Warning: Could not generate 3D hillshade: {e}")
        print("Falling back to simple elevation map...")
        plt.imshow(elevation.T, origin='lower', cmap='terrain')
    
    pbar.update(2)  # Steps 3-4: Hillshade created and plotted
    
    # Add colorbar
    cbar = plt.colorbar(label='Elevation (m)')
    
    plt.title('Battlefield Elevation Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    pbar.update(1)  # Step 5: Labels and colorbar added
    
    # Save figure
    plt.savefig(save_path, dpi=300)
    print(f"Elevation visualization saved to {save_path}")
    pbar.close()
    plt.close()

def visualize_entities_with_progress(sim, timestamp=None, show_terrain=True, figsize=(15, 12), save_path="entities_visualization.png"):
    """
    Visualize entities (targets and blue forces) on the terrain with progress indicators
    """
    print("Generating entity visualization...")
    
    # Initialize progress bar
    pbar = tqdm(total=7, desc="Entity visualization steps")
    
    plt.figure(figsize=figsize)
    pbar.update(1)  # Step 1: Figure created
    
    # Plot terrain if requested
    if show_terrain and sim.terrain_map is not None:
        # Create custom colormap for terrain
        terrain_colors = [sim.TERRAIN_TYPES[i]['color'] for i in range(len(sim.TERRAIN_TYPES))]
        terrain_cmap = mcolors.ListedColormap(terrain_colors)
        
        # Plot terrain
        plt.imshow(sim.terrain_map.T, origin='lower', cmap=terrain_cmap, 
                  vmin=0, vmax=len(sim.TERRAIN_TYPES)-1, alpha=0.7)
    
    pbar.update(1)  # Step 2: Terrain plotted
    
    # Find the relevant observations for the given timestamp
    pbar.set_description("Processing observation data")
    
    if timestamp is not None:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Filter observations by timestamp
        target_obs = [obs for obs in sim.target_observations if obs['timestamp'] == timestamp]
        blue_force_obs = [obs for obs in sim.blue_force_observations if obs['timestamp'] == timestamp]
    else:
        # If no timestamp provided, find the timestamp with the most observations
        timestamps = {}
        for obs in sim.target_observations:
            ts = obs['timestamp']
            if ts not in timestamps:
                timestamps[ts] = 0
            timestamps[ts] += 1
        
        # Get the timestamp with the most observations
        if timestamps:
            best_timestamp = max(timestamps.items(), key=lambda x: x[1])[0]
            print(f"Using timestamp with most observations: {best_timestamp}")
            
            # Filter observations
            target_obs = [obs for obs in sim.target_observations if obs['timestamp'] == best_timestamp]
            blue_force_obs = [obs for obs in sim.blue_force_observations if obs['timestamp'] == best_timestamp]
        else:
            # Use all observations
            target_obs = sim.target_observations
            blue_force_obs = sim.blue_force_observations
    
    pbar.update(1)  # Step 3: Observations filtered
    
    # Plot targets
    target_colors = {
        'infantry': 'red',
        'light_vehicle': 'orange',
        'heavy_vehicle': 'darkred',
        'uav': 'purple',
        'civilian': 'pink'
    }
    
    target_markers = {
        'infantry': 's',  # Square
        'light_vehicle': '^',  # Triangle
        'heavy_vehicle': 'D',  # Diamond
        'uav': '*',  # Star
        'civilian': 'o'  # Circle
    }
    
    pbar.set_description("Plotting targets")
    
    if target_obs:
        # If there are too many observations, sample them
        max_targets = 1000  # Maximum number of targets to plot
        if len(target_obs) > max_targets:
            print(f"Sampling {max_targets} targets from {len(target_obs)} total")
            import random
            target_obs = random.sample(target_obs, max_targets)
        
        for obs in target_obs:
            target_id = obs['id']
            x, y = obs['x_coord'], obs['y_coord']
            grid_x, grid_y = sim.data_to_grid(x, y)
            
            # Get target class
            target_class = 'unknown'
            if target_id in sim.targets:
                target_class = sim.targets[target_id]['class']
            
            # Set marker and color based on class
            marker = target_markers.get(target_class, 'o')
            color = target_colors.get(target_class, 'red')
            
            # Plot the target
            plt.scatter(grid_x, grid_y, marker=marker, color=color, s=80, edgecolors='black', zorder=10)
    
    pbar.update(1)  # Step 4: Targets plotted
    
    # Plot blue forces
    blue_force_markers = {
        'infantry_squad': 's',  # Square
        'mechanized_patrol': '^',  # Triangle
        'recon_team': 'o',  # Circle
        'command_post': 'p',  # Pentagon
        'uav_surveillance': '*'  # Star
    }
    
    pbar.set_description("Plotting blue forces")
    
    if blue_force_obs:
        # If there are too many observations, sample them
        max_blue_forces = 500  # Maximum number of blue forces to plot
        if len(blue_force_obs) > max_blue_forces:
            print(f"Sampling {max_blue_forces} blue forces from {len(blue_force_obs)} total")
            import random
            blue_force_obs = random.sample(blue_force_obs, max_blue_forces)
        
        for obs in blue_force_obs:
            force_id = obs['id']
            x, y = obs['x_coord'], obs['y_coord']
            grid_x, grid_y = sim.data_to_grid(x, y)
            
            # Get blue force class
            force_class = 'unknown'
            if force_id in sim.blue_forces:
                force_class = sim.blue_forces[force_id]['class']
            
            # Set marker based on class
            marker = blue_force_markers.get(force_class, 'o')
            
            # Plot the blue force
            plt.scatter(grid_x, grid_y, marker=marker, color='blue', s=80, edgecolors='black', zorder=10)
            
            # Draw detection range circle
            detection_range = sim.BLUE_FORCE_CLASSES.get(force_class, {}).get('detection_range', 3)
            detection_circle = plt.Circle((grid_x, grid_y), detection_range, color='blue', fill=False, 
                                          linestyle='--', alpha=0.5, zorder=5)
            plt.gca().add_patch(detection_circle)
    
    pbar.update(1)  # Step 5: Blue forces plotted
    
    pbar.set_description("Adding legends and labels")
    
    # Create legends
    target_legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Infantry'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10, label='Light Vehicle'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='darkred', markersize=10, label='Heavy Vehicle'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='purple', markersize=10, label='UAV'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label='Civilian')
    ]
    
    blue_legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Blue Infantry'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Blue Mechanized'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Blue Recon'),
        plt.Line2D([0], [0], marker='p', color='w', markerfacecolor='blue', markersize=10, label='Blue Command'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=10, label='Blue UAV')
    ]
    
    # Add legends
    plt.legend(handles=target_legend_elements, loc='upper right', title='Targets')
    plt.gca().add_artist(plt.legend(handles=blue_legend_elements, loc='upper left', title='Blue Forces'))
    
    # Set title with timestamp if provided
    if timestamp is not None:
        plt.title(f'Battlefield Entities at {timestamp}')
    else:
        plt.title('Battlefield Entities')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    
    pbar.update(1)  # Step 6: Legends and labels added
    
    # Save figure
    plt.savefig(save_path, dpi=300)
    print(f"Entity visualization saved to {save_path}")
    pbar.update(1)  # Step 7: Figure saved
    pbar.close()
    
    plt.close()
    return save_path

def visualize_trajectories_with_progress(sim, target_ids=None, max_targets=10, show_terrain=True, 
                                        figsize=(15, 12), save_path="trajectories_visualization.png"):
    """
    Visualize trajectories of targets with progress indicators
    """
    print("Generating trajectory visualization...")
    
    # Initialize progress bar
    pbar = tqdm(total=5, desc="Trajectory visualization steps")
    
    plt.figure(figsize=figsize)
    pbar.update(1)  # Step 1: Figure created
    
    # Plot terrain if requested
    if show_terrain:
        # Create custom colormap for terrain
        terrain_colors = [sim.TERRAIN_TYPES[i]['color'] for i in range(len(sim.TERRAIN_TYPES))]
        terrain_cmap = mcolors.ListedColormap(terrain_colors)
        
        # Plot terrain
        plt.imshow(sim.terrain_map.T, origin='lower', cmap=terrain_cmap, 
                  vmin=0, vmax=len(sim.TERRAIN_TYPES)-1, alpha=0.7)
    
    pbar.update(1)  # Step 2: Terrain plotted
    
    # If no target IDs provided, select a subset of targets with the most observations
    if target_ids is None:
        # Count observations per target
        target_counts = {}
        for obs in sim.target_observations:
            target_id = obs['id']
            if target_id not in target_counts:
                target_counts[target_id] = 0
            target_counts[target_id] += 1
        
        # Sort targets by observation count
        sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take the top N targets
        target_ids = [t[0] for t in sorted_targets[:max_targets]]
        print(f"Showing trajectories for top {len(target_ids)} targets with most observations")
    
    pbar.update(1)  # Step 3: Target selection complete
    
    # Target colors by class
    target_colors = {
        'infantry': 'red',
        'light_vehicle': 'orange',
        'heavy_vehicle': 'darkred',
        'uav': 'purple',
        'civilian': 'pink'
    }
    
    pbar.set_description("Plotting trajectories")
    
    # Plot each target's trajectory
    for target_id in target_ids:
        if target_id not in sim.targets:
            continue
            
        target = sim.targets[target_id]
        target_class = target['class']
        color = target_colors.get(target_class, 'red')
        
        # Get observations for this target
        observations = []
        for obs in sim.target_observations:
            if obs['id'] == target_id:
                observations.append(obs)
        
        # Sort by timestamp
        observations.sort(key=lambda x: x['timestamp'])
        
        if not observations:
            continue
            
        # Convert coordinates to grid indices
        grid_points = []
        for obs in observations:
            x, y = obs['x_coord'], obs['y_coord']
            grid_x, grid_y = sim.data_to_grid(x, y)
            grid_points.append((grid_x, grid_y))
        
        # Extract x and y coordinates
        grid_xs, grid_ys = zip(*grid_points)
        
        # Plot trajectory
        plt.plot(grid_xs, grid_ys, '-', color=color, linewidth=2, alpha=0.7, label=f"{target_class} ({target_id})")
        
        # Plot start and end points
        plt.scatter(grid_xs[0], grid_ys[0], marker='o', color=color, s=100, edgecolors='black', zorder=10)
        plt.scatter(grid_xs[-1], grid_ys[-1], marker='x', color=color, s=100, edgecolors='black', zorder=10)
    
    pbar.update(1)  # Step 4: Trajectories plotted
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='Infantry'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Light Vehicle'),
        plt.Line2D([0], [0], color='darkred', lw=2, label='Heavy Vehicle'),
        plt.Line2D([0], [0], color='purple', lw=2, label='UAV'),
        plt.Line2D([0], [0], color='pink', lw=2, label='Civilian'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Start'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=8, label='End')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('Target Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300)
    print(f"Trajectory visualization saved to {save_path}")
    pbar.update(1)  # Step 5: Figure saved
    pbar.close()
    
    plt.close()
    return save_path

def build_datasets_with_progress(sim, test_ratio=0.2, window_size=5, prediction_horizons=[1, 3, 5], 
                               include_terrain=True, include_blue_forces=True):
    """
    Build trajectory datasets with progress indicators
    """
    print("\nBuilding trajectory datasets...")
    
    # Initialize progress
    pbar = tqdm(total=len(prediction_horizons) + 3, desc="Dataset preparation")
    
    # Group observations by target
    pbar.set_description("Grouping observations by target")
    target_trajectories = {}
    
    for target_id, target in sim.targets.items():
        # Sort observations by timestamp
        if 'observations' in target:
            observations = sorted(target['observations'], key=lambda x: x['timestamp'])
            target_trajectories[target_id] = observations
    
    pbar.update(1)  # Step 1: Observations grouped
    
    # Split targets into training and testing sets
    pbar.set_description("Splitting targets into train/test sets")
    target_ids = list(target_trajectories.keys())
    if not target_ids:
        print("No trajectory data available")
        pbar.close()
        return None
        
    # Use sklearn for the split
    from sklearn.model_selection import train_test_split
    train_ids, test_ids = train_test_split(target_ids, test_size=test_ratio, random_state=42)
    
    print(f"Split {len(target_ids)} targets into {len(train_ids)} training and {len(test_ids)} testing")
    pbar.update(1)  # Step 2: Targets split
    
    # Prepare blue force data if included
    pbar.set_description("Preparing blue force data")
    blue_force_data = None
    if include_blue_forces and sim.blue_force_observations:
        # Map blue force observations by timestamp
        blue_force_data = {}
        for obs in sim.blue_force_observations:
            timestamp = obs['timestamp']
            if timestamp not in blue_force_data:
                blue_force_data[timestamp] = []
            blue_force_data[timestamp].append(obs)
    
    pbar.update(1)  # Step 3: Blue force data prepared
    
    # Build datasets for each prediction horizon
    datasets = {}
    
    for i, horizon in enumerate(prediction_horizons):
        pbar.set_description(f"Building datasets for {horizon}-step prediction ({i+1}/{len(prediction_horizons)})")
        
        # Create training dataset
        train_inputs = []
        train_outputs = []
        
        # Use tqdm for train targets
        train_pbar = tqdm(train_ids, desc=f"Processing {len(train_ids)} training targets", leave=False)
        for target_id in train_pbar:
            trajectory = target_trajectories[target_id]
            target_class = sim.targets[target_id]['class']
            
            # For each valid start position
            valid_positions = len(trajectory) - window_size - horizon + 1
            if valid_positions > 0:
                train_pbar.set_postfix({"valid_positions": valid_positions})
                
                for i in range(valid_positions):
                    # Extract input sequence
                    input_seq = trajectory[i:i+window_size]
                    
                    # Extract target (future position)
                    target_obs = trajectory[i+window_size+horizon-1]
                    
                    # Create input features
                    input_features = sim._create_features(
                        input_seq, target_id, target_class, 
                        include_terrain=include_terrain, 
                        include_blue_forces=include_blue_forces, 
                        blue_force_data=blue_force_data
                    )
                    
                    # Create output features (future position)
                    output_features = [target_obs['x_coord'], target_obs['y_coord']]
                    
                    train_inputs.append(input_features)
                    train_outputs.append(output_features)
        
        # Create testing dataset
        test_inputs = []
        test_outputs = []
        
        # Use tqdm for test targets
        test_pbar = tqdm(test_ids, desc=f"Processing {len(test_ids)} testing targets", leave=False)
        for target_id in test_pbar:
            trajectory = target_trajectories[target_id]
            target_class = sim.targets[target_id]['class']
            
            # For each valid start position
            valid_positions = len(trajectory) - window_size - horizon + 1
            if valid_positions > 0:
                test_pbar.set_postfix({"valid_positions": valid_positions})
                
                for i in range(valid_positions):
                    # Extract input sequence
                    input_seq = trajectory[i:i+window_size]
                    
                    # Extract target (future position)
                    target_obs = trajectory[i+window_size+horizon-1]
                    
                    # Create input features
                    input_features = sim._create_features(
                        input_seq, target_id, target_class, 
                        include_terrain=include_terrain, 
                        include_blue_forces=include_blue_forces, 
                        blue_force_data=blue_force_data
                    )
                    
                    # Create output features (future position)
                    output_features = [target_obs['x_coord'], target_obs['y_coord']]
                    
                    test_inputs.append(input_features)
                    test_outputs.append(output_features)
        
        # Store datasets
        datasets[f'horizon_{horizon}'] = {
            'X_train': train_inputs,
            'y_train': train_outputs,
            'X_test': test_inputs,
            'y_test': test_outputs
        }
        
        print(f"  - Horizon {horizon}: {len(train_inputs)} training samples, {len(test_inputs)} testing samples")
        pbar.update(1)  # Step 4+i: Dataset for horizon i built
    
    pbar.close()
    
    # Save datasets to disk for later use
    print("Saving datasets to disk...")
    import pickle
    os.makedirs('synthetic_data', exist_ok=True)
    with open('synthetic_data/datasets.pkl', 'wb') as f:
        pickle.dump(datasets, f)
    print("Datasets saved to synthetic_data/datasets.pkl")
    
    return datasets

def run_enhanced_visualization():
    """Run enhanced battlefield visualization with progress bars"""
    print("Starting enhanced battlefield visualization...")
    start_time = time.time()
    
    # Create simulation
    sim = BattlefieldSimulation(size=(100, 100))
    
    # Load terrain data with clear progress indication
    print("\nLoading terrain data...")
    try:
        terrain_loaded = sim.load_terrain_data(
            terrain_data_path="simulation_data/land_use_map.npy",
            elevation_data_path="simulation_data/elevation_map.npy"
        )
        if not terrain_loaded:
            print("Failed to load terrain data. Check file paths and formats.")
    except Exception as e:
        print(f"Error loading terrain data: {e}")
    
    # Load observation data with clear progress indication
    print("\nLoading observation data...")
    try:
        obs_loaded = sim.load_observation_data(
            target_csv="synthetic_data/target_observations.csv",
            blue_force_csv="synthetic_data/blue_force_observations.csv"
        )
        if not obs_loaded:
            print("Failed to load observation data. Check file paths and formats.")
    except Exception as e:
        print(f"Error loading observation data: {e}")
    
    # Create visualizations with progress indicators
    print("\nGenerating visualizations...")
    os.makedirs('visualizations', exist_ok=True)
    
    # Check if terrain data is available
    if sim.terrain_map is not None:
        visualize_terrain_with_progress(sim, show_elevation=True, 
                                     save_path="visualizations/terrain_visualization.png")
    
        # Create a dedicated elevation visualization if available
        if sim.elevation_map is not None:
            visualize_elevation_separately(sim, 
                                        save_path="visualizations/elevation_visualization.png")
    
    # Check if observation data is available
    if sim.target_observations:
        visualize_entities_with_progress(sim, timestamp=None, show_terrain=True,
                                      save_path="visualizations/entities_visualization.png")
        
        visualize_trajectories_with_progress(sim, target_ids=None, max_targets=10, show_terrain=True,
                                          save_path="visualizations/trajectories_visualization.png")
    
    # Build datasets with progress indicators if observations are available
    if sim.target_observations:
        datasets = build_datasets_with_progress(
            sim,
            test_ratio=0.2,
            window_size=5,
            prediction_horizons=[1, 3, 5, 10],  # Different horizons
            include_terrain=True,
            include_blue_forces=True
        )
        
        # Print dataset statistics
        if datasets:
            print("\nDataset Statistics:")
            for horizon, data in datasets.items():
                print(f"  {horizon}:")
                print(f"    Training: {len(data['X_train'])} samples")
                print(f"    Testing: {len(data['X_test'])} samples")
    
    end_time = time.time()
    print(f"\nVisualization completed in {end_time - start_time:.2f} seconds.")
    print(f"All visualizations saved to the 'visualizations' directory.")

if __name__ == "__main__":
    run_enhanced_visualization()
