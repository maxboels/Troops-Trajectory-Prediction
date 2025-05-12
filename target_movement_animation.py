# target_movement_animation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap
from tqdm import tqdm

import random
import string


def add_target_ids_to_red_data(target_data_path, mapping_file_path):
    """
    Add target IDs to the red data file by merging with the mapping file.
    Generate new target IDs for locations that don't have mappings.
    
    Args:
        target_data_path: Path to the red data CSV
        mapping_file_path: Path to the mapping IDs CSV
    
    Returns:
        DataFrame: Updated red data with target IDs
    """
    # Load the red data
    target_data = pd.read_csv(target_data_path)
    
    # Handle any truncated or incomplete lines by checking for NaN values
    target_data = target_data.dropna(how='all')
    
    # Load the mapping data
    mapping_data = pd.read_csv(mapping_file_path)
    
    # Create a dictionary for name to target_id mapping
    name_to_target_id = dict(zip(mapping_data['name'], mapping_data['target_id']))
    
    # Find locations missing from the mapping file
    missing_locations = [loc for loc in target_data['name'].unique() 
                        if loc not in name_to_target_id]
    
    # Generate new target IDs for missing locations
    def generate_target_id():
        chars = string.ascii_uppercase + string.digits
        return 'ID_' + ''.join(random.choice(chars) for _ in range(8))
    
    # Add the new mappings
    for location in missing_locations:
        if pd.notna(location) and location.strip():
            name_to_target_id[location] = generate_target_id()
    
    # Create an updated mapping file
    updated_mapping_data = pd.DataFrame({
        'name': list(name_to_target_id.keys()),
        'target_id': list(name_to_target_id.values())
    })
    
    # Save the updated mapping file
    updated_mapping_path = os.path.join(os.path.dirname(mapping_file_path), 'updated_mappings_ids.csv')
    updated_mapping_data.to_csv(updated_mapping_path, index=False)
    
    # Add target_id to the red data
    target_data['target_id'] = target_data['name'].map(name_to_target_id)
    
    # Save the updated red data
    updated_red_path = os.path.join(os.path.dirname(target_data_path), 'red_data_with_target_ids.csv')
    target_data.to_csv(updated_red_path, index=False)
    
    print(f"Updated red data saved to: {updated_red_path}")
    print(f"Updated mappings saved to: {updated_mapping_path}")
    
    return target_data


def create_target_movement_animation(target_data_path, blue_force_path, 
                                     terrain_path=None, elevation_path=None,
                                     output_file="target_movement_animation.mp4",
                                     fps=10, dpi=150, duration_seconds=30,
                                     use_geometry=False):
    """
    Create an animation of target movements over time with colored trails representing time progression.
    
    Args:
        target_data_path: Path to target data CSV
        blue_force_path: Path to blue force data CSV
        terrain_path: Path to terrain data NPY file (optional)
        elevation_path: Path to elevation data NPY file (optional)
        output_file: Path to save the animation
        fps: Frames per second
        dpi: Resolution of the animation
        duration_seconds: Duration of the animation in seconds
        use_geometry: Whether to use geometry column instead of longitude/latitude
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from tqdm import tqdm
    import os
    
    # Get directory of the target data file
    target_dir = os.path.dirname(target_data_path)
    
    # Define path to mapping file
    mapping_file_path = os.path.join(target_dir, "mappings_ids.csv")
    
    # First, add target IDs to the red data if needed
    if 'target_id' not in pd.read_csv(target_data_path).columns:
        print("Adding target IDs to red data...")
        add_target_ids_to_red_data(target_data_path, mapping_file_path)
        # Update the target_data_path to use the new file
        target_data_path = os.path.join(target_dir, "red_data_with_target_ids.csv")
    
    # Load data
    target_data = pd.read_csv(target_data_path)
    print(f"Loaded target data with columns: {target_data.columns.tolist()}")
    blue_force_data = pd.read_csv(blue_force_path)
    print(f"Loaded blue force data with columns: {blue_force_data.columns.tolist()}")
    
    # Ensure target_id column is properly formatted
    if 'target_id' in target_data.columns:
        target_data['target_id'] = target_data['target_id'].astype(str)
        
    # Ensure datetime is properly parsed
    if 'datetime' in target_data.columns:
        target_data['datetime'] = pd.to_datetime(target_data['datetime'])
    
    # Handle geometry column if use_geometry is true
    if use_geometry and 'geometry' in target_data.columns:
        print("Using geometry column for coordinates...")
        
        # Function to extract coordinates from WKT POINT format
        def extract_coordinates(geom_text):
            try:
                # Extract text within parentheses
                coords_text = geom_text.split('(')[1].split(')')[0].strip()
                # Split the coordinates and convert to float
                lon, lat = map(float, coords_text.split())
                return lon, lat
            except (IndexError, ValueError, AttributeError):
                # Return NaN if parsing fails
                return np.nan, np.nan
        
        # Extract coordinates from geometry column
        coords = target_data['geometry'].apply(extract_coordinates)
        
        # Create new longitude and latitude columns from the extracted coordinates
        if 'longitude_geom' not in target_data.columns:
            target_data['longitude_geom'] = [coord[0] for coord in coords]
        if 'latitude_geom' not in target_data.columns:
            target_data['latitude_geom'] = [coord[1] for coord in coords]
        
        # Use the geometry-derived coordinates for the visualization
        lon_col = 'longitude_geom'
        lat_col = 'latitude_geom'
        
        # Check for any parsing errors
        invalid_coords = target_data[target_data[lon_col].isna() | target_data[lat_col].isna()]
        if len(invalid_coords) > 0:
            print(f"Warning: {len(invalid_coords)} rows had invalid geometry formats and will be excluded")
            target_data = target_data.dropna(subset=[lon_col, lat_col])
    else:
        # Use standard longitude and latitude columns
        lon_col = 'longitude'
        lat_col = 'latitude'
    
    # Add a normalized time column for coloring by time progression
    time_min = target_data['datetime'].min()
    time_max = target_data['datetime'].max()
    time_range = (time_max - time_min).total_seconds()
    
    # Function to normalize time
    def normalize_time(dt):
        return (dt - time_min).total_seconds() / time_range
    
    target_data['normalized_time'] = target_data['datetime'].apply(normalize_time)
    
    # Load terrain and elevation data if available
    terrain_data = None
    elevation_data = None
    
    if terrain_path and os.path.exists(terrain_path):
        terrain_data = np.load(terrain_path)
        print(f"Loaded terrain data with shape {terrain_data.shape}")
        print(f"Terrain data min: {terrain_data.min()}, max: {terrain_data.max()}")
        print(f"Origin lower: {terrain_data.shape[0]}, upper: {terrain_data.shape[1]}")

        # Flip vertically (South ↔ North)
        terrain_data = np.flipud(terrain_data)
    
    if elevation_path and os.path.exists(elevation_path):
        elevation_data = np.load(elevation_path)
        print(f"Loaded elevation data with shape {elevation_data.shape}")
        print(f"Elevation data min: {elevation_data.min()}, max: {elevation_data.max()}")
        print(f"Origin lower: {elevation_data.shape[0]}, upper: {elevation_data.shape[1]}")

        # Flip vertically (South ↔ North)
        elevation_data = np.flipud(elevation_data)
    
    # Get all unique timestamps
    timestamps = sorted(pd.unique(target_data['datetime']))
    
    # Determine how many timestamps to use based on desired duration
    total_frames = fps * duration_seconds
    frame_skip = max(1, len(timestamps) // total_frames)
    selected_timestamps = timestamps[::frame_skip]
    
    print(f"Creating animation with {len(selected_timestamps)} frames...")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get coordinate bounds
    lon_min, lon_max = target_data[lon_col].min(), target_data[lon_col].max()
    lat_min, lat_max = target_data[lat_col].min(), target_data[lat_col].max()
    
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
        ax.imshow(terrain_data, cmap=terrain_cmap, alpha=0.7,
                 extent=[lon_min, lon_max, lat_min, lat_max],
                 aspect='auto', origin='lower', zorder=0)
        
        # Add elevation overlay if available
        if elevation_data is not None:
            from matplotlib.colors import Normalize
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
    
    # Define the colormap for the time-based trails (similar to your example)
    time_cmap = plt.cm.viridis
    time_norm = Normalize(vmin=0, vmax=1)
    
    # Set up plot elements
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Target Movement Animation')
    ax.grid(True, alpha=0.3)
    
    # Add time colorbar
    time_sm = ScalarMappable(cmap=time_cmap, norm=time_norm)
    time_sm.set_array([])
    cbar = plt.colorbar(time_sm, ax=ax, pad=0.01, shrink=0.6)
    cbar.set_label('Time Progression')
    
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
    
    # Initialize scatter plots for blue and red forces
    # For blue forces, we always use the standard longitude/latitude
    blue_scatter = ax.scatter([], [], c='blue', s=120, marker='^', 
                            label='Blue Forces', zorder=10, edgecolor='black')
    
    # Create a scatter plot for each target class
    target_scatters = {}
    for target_class, color in target_colors.items():
        target_scatters[target_class] = ax.scatter([], [], c=color, s=80, marker='o', 
                                                 zorder=8, edgecolor='black')
    
    # Track lines for each target's trail
    target_line_collections = {}
    
    # Text for timestamp display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                      bbox=dict(facecolor='white', alpha=0.7), zorder=20)
    
    # Add a notice if using geometry coordinates
    if use_geometry:
        geometry_text = ax.text(0.02, 0.90, 'Using geometry coordinates', 
                             transform=ax.transAxes, fontsize=10,
                             bbox=dict(facecolor='white', alpha=0.7), zorder=20)
    
    # Initialize function for animation
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
        latest_positions = current_data.groupby('target_id').apply(
            lambda x: x.loc[x['datetime'].idxmax()]
        ).reset_index(drop=True)
        
        # Update blue forces
        if blue_force_data is not None:
            blue_scatter.set_offsets(blue_force_data[['longitude', 'latitude']].values)
        
        # Update target positions by class
        for target_class, color in target_colors.items():
            # Get targets of this class
            if 'class' in latest_positions.columns:
                class_targets = latest_positions[
                    latest_positions['class'].str.lower() == target_class
                ]
            elif 'target_class' in latest_positions.columns:
                class_targets = latest_positions[
                    latest_positions['target_class'].str.lower() == target_class
                ]
            else:
                # If no class info, treat all as unknown
                class_targets = latest_positions if target_class == 'unknown' else pd.DataFrame()
            
            # Update scatter plot
            if len(class_targets) > 0:
                target_scatters[target_class].set_offsets(
                    class_targets[[lon_col, lat_col]].values
                )
            else:
                target_scatters[target_class].set_offsets(np.empty((0, 2)))
        
        # Remove existing line collections
        for target_id in list(target_line_collections.keys()):
            if target_id in target_line_collections:
                target_line_collections[target_id].remove()
        target_line_collections.clear()
        
        # Add time-colored trails for each target
        for target_id, group in current_data.groupby('target_id'):
            if len(group) >= 2:
                # Sort by time
                group = group.sort_values('datetime')
                
                # Get coordinates
                points = group[[lon_col, lat_col]].values
                
                # Create segments
                segments = np.array([np.column_stack([points[i:i+2, 0], points[i:i+2, 1]]) 
                                   for i in range(len(points)-1)])
                
                # Skip if no segments
                if len(segments) == 0:
                    continue
                
                # Get normalized times for coloring
                times = group['normalized_time'].values
                segment_times = [(times[i] + times[i+1])/2 for i in range(len(times)-1)]
                
                # Create line collection with time-based coloring
                lc = LineCollection(segments, cmap=time_cmap, norm=time_norm, 
                                   linewidth=3.5, zorder=5, alpha=0.8)
                
                # Set segment colors based on time
                lc.set_array(np.array(segment_times))
                
                # Add to axis
                line_collection = ax.add_collection(lc)
                target_line_collections[target_id] = line_collection
        
        # Update time display
        time_text.set_text(f'Time: {pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Return updated artists
        artists = [blue_scatter] + list(target_scatters.values()) + list(target_line_collections.values()) + [time_text]
        if use_geometry:
            artists.append(geometry_text)
        return artists
    
    # Create animation
    animation = FuncAnimation(
        fig, update, frames=len(selected_timestamps),
        init_func=init, blit=True, interval=1000/fps
    )
    
    # Save animation
    print("Saving animation...")
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Target Movement Animation'), bitrate=3600)
    
    with tqdm(total=100, desc="Encoding video") as pbar:
        animation.save(output_file, writer=writer, dpi=dpi,
                    progress_callback=lambda i, n: pbar.update(100/n))
    
    print(f"Animation saved to {output_file}")
    return animation


# Version that displays static trails similar to the example image
def create_target_movement_animation_static(target_data_path, blue_force_path, 
                                           terrain_path=None, elevation_path=None,
                                           output_file="target_movement_map.png",
                                           use_geometry=False):
    """
    Create a static image showing target movement trails colored by time, similar to the example image.
    
    Args:
        target_data_path: Path to target data CSV
        blue_force_path: Path to blue force data CSV
        terrain_path: Path to terrain data NPY file (optional)
        elevation_path: Path to elevation data NPY file (optional)
        output_file: Path to save the static image
        use_geometry: Whether to use geometry column instead of longitude/latitude
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from scipy.interpolate import interp1d
    import os
    
    # Get directory of the target data file
    target_dir = os.path.dirname(target_data_path)
    
    # Define path to mapping file
    mapping_file_path = os.path.join(target_dir, "mappings_ids.csv")
    
    # First, add target IDs to the red data if needed
    if 'target_id' not in pd.read_csv(target_data_path).columns:
        print("Adding target IDs to red data...")
        add_target_ids_to_red_data(target_data_path, mapping_file_path)
        # Update the target_data_path to use the new file
        target_data_path = os.path.join(target_dir, "red_data_with_target_ids.csv")
    
    # Load data
    target_data = pd.read_csv(target_data_path)
    print(f"Loaded target data with columns: {target_data.columns.tolist()}")
    blue_force_data = pd.read_csv(blue_force_path)
    print(f"Loaded blue force data with columns: {blue_force_data.columns.tolist()}")
    
    # Ensure target_id column is properly formatted
    if 'target_id' in target_data.columns:
        target_data['target_id'] = target_data['target_id'].astype(str)
        
    # Ensure datetime is properly parsed
    if 'datetime' in target_data.columns:
        target_data['datetime'] = pd.to_datetime(target_data['datetime'])
    
    # Handle geometry column if use_geometry is true
    if use_geometry and 'geometry' in target_data.columns:
        print("Using geometry column for coordinates...")
        
        # Function to extract coordinates from WKT POINT format
        def extract_coordinates(geom_text):
            try:
                # Extract text within parentheses
                coords_text = geom_text.split('(')[1].split(')')[0].strip()
                # Split the coordinates and convert to float
                lon, lat = map(float, coords_text.split())
                return lon, lat
            except (IndexError, ValueError, AttributeError):
                # Return NaN if parsing fails
                return np.nan, np.nan
        
        # Extract coordinates from geometry column
        coords = target_data['geometry'].apply(extract_coordinates)
        
        # Create new longitude and latitude columns from the extracted coordinates
        if 'longitude_geom' not in target_data.columns:
            target_data['longitude_geom'] = [coord[0] for coord in coords]
        if 'latitude_geom' not in target_data.columns:
            target_data['latitude_geom'] = [coord[1] for coord in coords]
        
        # Use the geometry-derived coordinates for the visualization
        lon_col = 'longitude_geom'
        lat_col = 'latitude_geom'
        
        # Check for any parsing errors
        invalid_coords = target_data[target_data[lon_col].isna() | target_data[lat_col].isna()]
        if len(invalid_coords) > 0:
            print(f"Warning: {len(invalid_coords)} rows had invalid geometry formats and will be excluded")
            target_data = target_data.dropna(subset=[lon_col, lat_col])
    else:
        # Use standard longitude and latitude columns
        lon_col = 'longitude'
        lat_col = 'latitude'
    
    # Add a normalized time column for coloring by time progression
    time_min = target_data['datetime'].min()
    time_max = target_data['datetime'].max()
    time_range = (time_max - time_min).total_seconds()
    
    # Function to normalize time
    def normalize_time(dt):
        return (dt - time_min).total_seconds() / time_range
    
    target_data['normalized_time'] = target_data['datetime'].apply(normalize_time)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get coordinate bounds
    lon_min, lon_max = target_data[lon_col].min(), target_data[lon_col].max()
    lat_min, lat_max = target_data[lat_col].min(), target_data[lat_col].max()
    
    # Add padding
    padding = 0.05
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    lon_min -= lon_range * padding
    lon_max += lon_range * padding
    lat_min -= lat_range * padding
    lat_max += lat_range * padding
    
    # Load terrain and elevation data if available
    if terrain_path and os.path.exists(terrain_path):
        terrain_data = np.load(terrain_path)
        print(f"Loaded terrain data with shape {terrain_data.shape}")
        
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
        
        # Flip vertically (South ↔ North)
        terrain_data = np.flipud(terrain_data)
        
        # Plot terrain
        ax.imshow(terrain_data, cmap=terrain_cmap, alpha=0.7,
                 extent=[lon_min, lon_max, lat_min, lat_max],
                 aspect='auto', origin='lower', zorder=0)
    
    # If elevation data is available
    if elevation_path and os.path.exists(elevation_path):
        elevation_data = np.load(elevation_path)
        print(f"Loaded elevation data with shape {elevation_data.shape}")
        
        # Flip vertically (South ↔ North)
        elevation_data = np.flipud(elevation_data)
        
        # Add elevation overlay
        from matplotlib.colors import Normalize
        elev_min = np.min(elevation_data)
        elev_max = np.max(elevation_data)
        elev_norm = Normalize(vmin=elev_min, vmax=elev_max)
        
        ax.imshow(elevation_data, cmap='terrain', norm=elev_norm, alpha=0.3,
                 extent=[lon_min, lon_max, lat_min, lat_max],
                 aspect='auto', origin='lower', zorder=1)
    
    # Set plot limits
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Target Movement Trails')
    ax.grid(True, alpha=0.3)
    
    # Define the colormap for time-based trails (similar to your example)
    time_cmap = plt.cm.viridis
    time_norm = Normalize(vmin=0, vmax=1)
    
    # Add time colorbar
    time_sm = ScalarMappable(cmap=time_cmap, norm=time_norm)
    time_sm.set_array([])
    cbar = plt.colorbar(time_sm, ax=ax, pad=0.01, shrink=0.6)
    cbar.set_label('Time [s]')
    
    # Add ticks with actual time intervals
    time_ticks = np.linspace(0, 1, 6)
    time_tick_labels = [f"{i:.1f}" for i in np.linspace(0, time_range, 6)]
    cbar.set_ticks(time_ticks)
    cbar.set_ticklabels(time_tick_labels)
    
    # Draw blue force positions
    if blue_force_data is not None:
        ax.scatter(blue_force_data['longitude'], blue_force_data['latitude'], 
                 c='blue', s=120, marker='^', zorder=10, 
                 edgecolor='black', label='Blue Forces')
    
    # Draw time-colored trails for each target
    for target_id, group in target_data.groupby('target_id'):
        if len(group) >= 2:
            # Sort by time
            group = group.sort_values('datetime')
            
            # Get target class if available
            if 'class' in group.columns:
                target_class = group['class'].iloc[0].lower()
            elif 'target_class' in group.columns:
                target_class = group['target_class'].iloc[0].lower()
            else:
                target_class = 'unknown'
            
            # Get coordinates and times
            x = group[lon_col].values
            y = group[lat_col].values
            times = group['normalized_time'].values
            
            # Create a denser path for smoother visualization (using interpolation)
            if len(x) > 2:
                # Create path parameter
                path_param = np.linspace(0, 1, len(x))
                # Create denser path parameter
                dense_param = np.linspace(0, 1, len(x) * 10)
                
                # Create interpolation functions
                interp_x = interp1d(path_param, x, kind='cubic')
                interp_y = interp1d(path_param, y, kind='cubic')
                interp_time = interp1d(path_param, times, kind='linear')
                
                # Generate denser path
                x_dense = interp_x(dense_param)
                y_dense = interp_y(dense_param)
                times_dense = interp_time(dense_param)
                
                # Use the interpolated values
                x, y, times = x_dense, y_dense, times_dense
            
            # Create points
            points = np.column_stack((x, y))
            
            # Draw colored line with varying width
            # (thicker for more recent times)
            for i in range(len(points) - 1):
                segment = np.array([points[i], points[i+1]])
                time_val = times[i]  # Use time value for color
                
                # Calculate line width (thicker for more recent)
                line_width = 1.5 + 3.5 * time_val
                
                # Draw line segment
                line = plt.Line2D(segment[:, 0], segment[:, 1], 
                                color=time_cmap(time_val), 
                                linewidth=line_width,
                                solid_capstyle='round',
                                zorder=5 + time_val)  # More recent paths on top
                
                # Add a slight glow effect
                line.set_path_effects([
                    PathEffects.Stroke(linewidth=line_width+1.5, foreground='white', alpha=0.3),
                    PathEffects.Normal()
                ])
                
                ax.add_line(line)
            
            # Mark end positions (most recent position) with a circle
            marker_color = {
                'tank': 'darkred',
                'armoured personnel carrier': 'orangered',
                'light vehicle': 'coral'
            }.get(target_class, 'red')
            
            ax.scatter(points[-1, 0], points[-1, 1], s=80, c=marker_color, 
                     zorder=10, edgecolor='black')
    
    # Add target legend
    target_legend = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=10, label='Tank'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orangered', markersize=10, label='APC'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', markersize=10, label='Light Vehicle'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Blue Forces')
    ]
    
    # Add legend
    ax.legend(handles=target_legend, loc='upper right', title='Forces')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Static visualization saved to {output_file}")
    
    return fig

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create target movement animation')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing data files')
    parser.add_argument('--terrain_path', type=str, default='adapted_data/terrain_map.npy',
                      help='Path to terrain data')
    parser.add_argument('--elevation_path', type=str, default='adapted_data/elevation_map.npy',
                      help='Path to elevation data')
    parser.add_argument('--output_file', type=str, default='target_movement_animation.mp4',
                      help='Output video file path')
    parser.add_argument('--fps', type=int, default=10,
                      help='Frames per second')
    parser.add_argument('--duration', type=int, default=10,
                      help='Duration of animation in seconds')
    parser.add_argument('--dpi', type=int, default=150,
                      help='Resolution of the animation')
    
    args = parser.parse_args()
    
    # Create animation
    create_target_movement_animation(
        os.path.join(args.data_dir, "red_positions_ground_truth.csv"),#"red_sightings.csv"),
        os.path.join(args.data_dir, "blue_locations.csv"),
        args.terrain_path,
        args.elevation_path,
        args.output_file,
        fps=args.fps,
        duration_seconds=args.duration,
        dpi=args.dpi,
        use_geometry=True
    )

    # Example usage
    # python target_movement_animation.py --data_dir data --terrain_path adapted_data/terrain_map.npy --elevation_path adapted_data/elevation_map.npy --output_file target_movement_animation.mp4 --fps 10 --duration 30 --dpi 150