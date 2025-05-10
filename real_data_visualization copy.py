"""
Adapter script to convert real dataset into a format compatible with existing visualization code.
Specifically tailored for the provided blue_locations.csv and red_sightings.csv formats.
"""
import os
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import warnings
import matplotlib.colors as mcolors
from matplotlib import cm
from pathlib import Path
from battlefield_simulation import BattlefieldSimulation
from helpers import LandUseCategory, get_raster_value_at_coords, get_altitude

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")

# Paths to the real data
DATA_DIR = "data"
BLUE_LOCATIONS_CSV = os.path.join(DATA_DIR, "blue_locations.csv")
RED_SIGHTINGS_CSV = os.path.join(DATA_DIR, "red_sightings.csv")
LAND_COVER_TIF = os.path.join(DATA_DIR, "gm_lc_v3_1_1.tif")
ELEVATION_TIF = os.path.join(DATA_DIR, "output_AW3D30.tif")

# Paths for adapted data
OUTPUT_DIR = "adapted_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_visualizations"), exist_ok=True)

# Mapping from the global land cover categories to our simulation categories
# Adjust this mapping based on your simulation's terrain definitions
LAND_USE_MAPPING = {
    # Water bodies -> Water (0)
    LandUseCategory.WATER_BODIES.value: 0,
    
    # Urban -> Urban (1)
    LandUseCategory.URBAN.value: 1,
    
    # Agricultural/cropland -> Agricultural (2)
    LandUseCategory.CROPLAND.value: 2,
    LandUseCategory.PADDY_FIELD.value: 2,
    LandUseCategory.CROPLAND_OTHER_VEGETATION_MOSAIC.value: 2,
    
    # Forest types -> Forest (3)
    LandUseCategory.BROADLEAF_EVERGREEN_FOREST.value: 3,
    LandUseCategory.BROADLEAF_DECIDUOUS_FOREST.value: 3,
    LandUseCategory.NEEDLELEAF_EVERGREEN_FOREST.value: 3,
    LandUseCategory.NEEDLELEAF_DECIDUOUS_FOREST.value: 3,
    LandUseCategory.MIXED_FOREST.value: 3,
    LandUseCategory.TREE_OPEN.value: 3,
    LandUseCategory.MANGROVE.value: 3,
    
    # Grassland/Herbaceous -> Grassland (4)
    LandUseCategory.HERBACEOUS.value: 4,
    LandUseCategory.HERBACEOUS_WITH_SPARSE_TREE_SHRUB.value: 4,
    LandUseCategory.SHRUB.value: 4,
    
    # Barren/Sparse vegetation -> Barren (5)
    LandUseCategory.SPARSE_VEGETATION.value: 5,
    LandUseCategory.BARE_AREA_CONSOLIDATED.value: 5,
    LandUseCategory.BARE_AREA_UNCONSOLIDATED.value: 5,
    
    # Wetland -> Wetland (6)
    LandUseCategory.WETLAND.value: 6,
    
    # Snow/Ice -> Snow/Ice (7)
    LandUseCategory.SNOW_ICE.value: 7
}

def print_data_info():
    """Print information about the new dataset to understand its structure"""
    # Check CSV files
    try:
        blue_df = pd.read_csv(BLUE_LOCATIONS_CSV)
        print(f"Blue locations file has {len(blue_df)} rows and columns: {blue_df.columns.tolist()}")
        print("First few rows of blue locations:")
        print(blue_df.head())
        print("\nUnique locations:", blue_df['name'].nunique())
        print()
    except Exception as e:
        print(f"Error reading blue locations: {e}")
    
    try:
        red_df = pd.read_csv(RED_SIGHTINGS_CSV)
        print(f"Red sightings file has {len(red_df)} rows and columns: {red_df.columns.tolist()}")
        print("First few rows of red sightings:")
        print(red_df.head())
        print("\nUnique targets:", red_df['target_id'].nunique())
        print("Target classes:", red_df['target_class'].unique())
        
        # Check time range
        red_df['datetime'] = pd.to_datetime(red_df['datetime'])
        print(f"Time range: {red_df['datetime'].min()} to {red_df['datetime'].max()}")
        print()
    except Exception as e:
        print(f"Error reading red sightings: {e}")
    
    # Check geospatial files
    try:
        with rasterio.open(LAND_COVER_TIF) as src:
            print(f"Land cover raster: {src.width}x{src.height} pixels, {src.count} bands")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Resolution: {src.res}")
            print()
    except Exception as e:
        print(f"Error reading land cover TIF: {e}")
    
    try:
        with rasterio.open(ELEVATION_TIF) as src:
            print(f"Elevation raster: {src.width}x{src.height} pixels, {src.count} bands")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Resolution: {src.res}")
            print()
    except Exception as e:
        print(f"Error reading elevation TIF: {e}")

def convert_rasters_to_numpy():
    """
    Convert GeoTIFF rasters to NumPy arrays for simulation.
    Returns bounding box information for coordinate conversion.
    """
    print("Converting rasters to NumPy arrays...")
    
    # Get raster bounds and transform
    with rasterio.open(LAND_COVER_TIF) as src:
        bounds = src.bounds
        transform = src.transform
        width = src.width
        height = src.height
        land_cover_data = src.read(1)
    
    # Convert land cover categories to simulation terrain types
    print("Mapping land cover categories to simulation terrain types...")
    terrain_map = np.zeros_like(land_cover_data, dtype=np.uint8)
    
    # Default to grassland (4) for any unmapped categories
    terrain_map.fill(4)
    
    # Apply mapping
    for lc_value, terrain_value in LAND_USE_MAPPING.items():
        terrain_map[land_cover_data == lc_value] = terrain_value
    
    # Load elevation data
    print("Loading elevation data...")
    with rasterio.open(ELEVATION_TIF) as src:
        elevation_data = src.read(1)
        
        # If elevation has different dimensions, resample to match land cover
        if elevation_data.shape != land_cover_data.shape:
            print("Elevation data has different dimensions, resampling...")
            # Simple resampling - this could be improved with proper resampling techniques
            from skimage.transform import resize
            elevation_data = resize(elevation_data, land_cover_data.shape, 
                                 preserve_range=True).astype(elevation_data.dtype)
    
    # Save as NumPy arrays
    terrain_path = os.path.join(OUTPUT_DIR, "terrain_map.npy")
    elevation_path = os.path.join(OUTPUT_DIR, "elevation_map.npy")
    
    np.save(terrain_path, terrain_map)
    np.save(elevation_path, elevation_data)
    
    print(f"Saved terrain map to {terrain_path}")
    print(f"Saved elevation map to {elevation_path}")
    
    return {
        'bounds': bounds,
        'transform': transform,
        'width': width,
        'height': height,
        'terrain_path': terrain_path,
        'elevation_path': elevation_path
    }

def adapt_location_data(raster_info):
    """
    Convert the location CSV files to the format expected by the visualization code.
    Uses the raster information for proper coordinate transformation.
    """
    print("Adapting location data...")
    
    # Load the data
    blue_df = pd.read_csv(BLUE_LOCATIONS_CSV)
    red_df = pd.read_csv(RED_SIGHTINGS_CSV)
    red_df['datetime'] = pd.to_datetime(red_df['datetime'])
    
    # Function to convert lat/lon to grid coordinates
    def latlon_to_grid(lat, lon, bounds, width, height):
        # Simple linear mapping from geographic coordinates to grid indices
        x_ratio = (lon - bounds.left) / (bounds.right - bounds.left)
        y_ratio = (bounds.top - lat) / (bounds.top - bounds.bottom)
        
        grid_x = int(x_ratio * width)
        grid_y = int(y_ratio * height)
        
        # Clamp to valid grid range
        grid_x = max(0, min(width - 1, grid_x))
        grid_y = max(0, min(height - 1, grid_y))
        
        return grid_x, grid_y
    
    # Adapt blue locations to match blue_force_observations.csv format
    blue_force_data = []
    
    # Blue locations don't have timestamps, so we'll assign them the earliest time from red sightings
    min_time = red_df['datetime'].min()
    max_time = red_df['datetime'].max()
    
    for _, row in blue_df.iterrows():
        # Convert coordinates
        grid_x, grid_y = latlon_to_grid(
            row['latitude'], row['longitude'], 
            raster_info['bounds'], raster_info['width'], raster_info['height']
        )
        
        # Map name to force class - based on position name
        name = row['name']
        
        # Determine force class based on name
        if 'Center' in name:
            force_class = 'command_post'
        elif 'North' in name:
            force_class = 'recon_team'
        elif 'South' in name:
            force_class = 'infantry_squad'
        elif 'East' in name:
            force_class = 'mechanized_patrol'
        elif 'West' in name:
            force_class = 'uav_surveillance'
        else:
            force_class = 'command_post'
        
        # Create multiple entries with different timestamps to make them appear throughout the animation
        num_entries = 10  # Number of entries per blue location
        for i in range(num_entries):
            # Calculate timestamp at regular intervals
            time_ratio = i / (num_entries - 1)
            timestamp = min_time + time_ratio * (max_time - min_time)
            
            # Add small random variation to position for each timestamp to create movement
            jitter = 2  # Number of grid cells to jitter
            jittered_x = grid_x + np.random.randint(-jitter, jitter + 1)
            jittered_y = grid_y + np.random.randint(-jitter, jitter + 1)
            
            # Clamp to valid grid range
            jittered_x = max(0, min(raster_info['width'] - 1, jittered_x))
            jittered_y = max(0, min(raster_info['height'] - 1, jittered_y))
            
            blue_force_data.append({
                'id': f"blue-{row['name'].replace(' ', '_')}",
                'timestamp': timestamp,
                'x_coord': jittered_x,
                'y_coord': jittered_y,
                'force_class': force_class
            })
    
    # Adapt red sightings to match target_observations.csv format
    target_data = []
    
    for _, row in red_df.iterrows():
        # Convert coordinates
        grid_x, grid_y = latlon_to_grid(
            row['latitude'], row['longitude'], 
            raster_info['bounds'], raster_info['width'], raster_info['height']
        )
        
        # Map target class to simulation target class
        target_class_mapping = {
            'tank': 'heavy_vehicle',
            'armoured personnel carrier': 'light_vehicle',
            # Add more mappings as needed
        }
        
        target_class = target_class_mapping.get(row['target_class'].lower(), 'infantry')
        
        target_data.append({
            'id': row['target_id'],
            'timestamp': row['datetime'],
            'x_coord': grid_x,
            'y_coord': grid_y,
            'target_class': target_class
        })
    
    # Save to CSV files
    blue_force_df = pd.DataFrame(blue_force_data)
    target_df = pd.DataFrame(target_data)
    
    # Convert timestamps to ISO format strings for consistent serialization
    blue_force_df['timestamp'] = blue_force_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    target_df['timestamp'] = target_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    blue_force_path = os.path.join(OUTPUT_DIR, "blue_force_observations.csv")
    target_path = os.path.join(OUTPUT_DIR, "target_observations.csv")
    
    blue_force_df.to_csv(blue_force_path, index=False)
    target_df.to_csv(target_path, index=False)
    
    print(f"Saved blue force observations to {blue_force_path}")
    print(f"Saved target observations to {target_path}")
    
    return {
        'blue_force_csv': blue_force_path,
        'target_csv': target_path
    }

def create_basic_visualizations():
    """
    Create basic visualizations directly from the raw data to understand it better.
    """
    print("Creating basic visualizations from raw data...")
    
    # Load data
    blue_df = pd.read_csv(BLUE_LOCATIONS_CSV)
    red_df = pd.read_csv(RED_SIGHTINGS_CSV)
    red_df['datetime'] = pd.to_datetime(red_df['datetime'])
    
    # Create a background map using land cover
    try:
        with rasterio.open(LAND_COVER_TIF) as src:
            land_cover = src.read(1)
            bounds = src.bounds
            
            plt.figure(figsize=(16, 12))
            plt.imshow(land_cover, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], 
                      alpha=0.7, cmap='terrain')
            
            # Plot blue locations
            plt.scatter(blue_df['longitude'], blue_df['latitude'], 
                       color='blue', marker='^', s=100, label='Blue Positions', edgecolor='black')
            
            # Plot red sightings (sample to avoid overcrowding)
            if len(red_df) > 100:
                # Group by target_id and take first and last sighting of each
                targets = red_df['target_id'].unique()
                sample_rows = []
                for target in targets:
                    target_rows = red_df[red_df['target_id'] == target]
                    sample_rows.append(target_rows.iloc[0])  # First sighting
                    sample_rows.append(target_rows.iloc[-1])  # Last sighting
                sample_df = pd.DataFrame(sample_rows)
            else:
                sample_df = red_df
            
            plt.scatter(sample_df['longitude'], sample_df['latitude'], 
                       color='red', marker='o', s=80, label='Red Sightings', edgecolor='black')
            
            plt.title('Overview Map with Blue and Red Positions')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the visualization
            plt.savefig(os.path.join(OUTPUT_DIR, "raw_visualizations", "overview_map.png"), dpi=300)
            plt.close()
            
    except Exception as e:
        print(f"Error creating overview map: {e}")
    
    # Visualize red target trajectories
    try:
        # Get unique targets
        targets = red_df['target_id'].unique()
        
        plt.figure(figsize=(16, 12))
        
        # Create a colormap for different targets
        colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))
        
        # Plot each target's trajectory
        for i, target_id in enumerate(targets):
            target_data = red_df[red_df['target_id'] == target_id].sort_values('datetime')
            
            # Plot trajectory line
            plt.plot(target_data['longitude'], target_data['latitude'], 
                    '-', color=colors[i], linewidth=2, alpha=0.7)
            
            # Plot start point
            plt.scatter(target_data['longitude'].iloc[0], target_data['latitude'].iloc[0], 
                       marker='o', color=colors[i], s=100, edgecolor='black', 
                       label=f"{target_id} ({target_data['target_class'].iloc[0]})")
            
            # Plot end point
            plt.scatter(target_data['longitude'].iloc[-1], target_data['latitude'].iloc[-1], 
                       marker='x', color=colors[i], s=100, edgecolor='black')
        
        plt.title('Red Target Trajectories')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the visualization
        plt.savefig(os.path.join(OUTPUT_DIR, "raw_visualizations", "red_trajectories.png"), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating trajectory visualization: {e}")
    
    # Visualize elevation data with positions
    try:
        with rasterio.open(ELEVATION_TIF) as src:
            elevation = src.read(1)
            bounds = src.bounds
            
            plt.figure(figsize=(16, 12))
            plt.imshow(elevation, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], 
                      cmap='terrain', alpha=0.8)
            plt.colorbar(label='Elevation (m)')
            
            # Plot blue locations
            plt.scatter(blue_df['longitude'], blue_df['latitude'], 
                       color='blue', marker='^', s=100, label='Blue Positions', edgecolor='black')
            
            # Plot random sample of red sightings
            if len(red_df) > 50:
                sample_df = red_df.sample(50)
            else:
                sample_df = red_df
                
            plt.scatter(sample_df['longitude'], sample_df['latitude'], 
                       color='red', marker='o', s=80, label='Red Sightings', edgecolor='black')
            
            plt.title('Elevation Map with Blue and Red Positions')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            
            # Save the visualization
            plt.savefig(os.path.join(OUTPUT_DIR, "raw_visualizations", "elevation_with_positions.png"), dpi=300)
            plt.close()
            
    except Exception as e:
        print(f"Error creating elevation visualization: {e}")
    
    print("Basic visualizations created in:", os.path.join(OUTPUT_DIR, "raw_visualizations"))

def create_animation_script():
    """Create a customized animation script for the real data."""
    script_path = os.path.join(OUTPUT_DIR, "real_data_animation.py")
    
    script_content = """
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
import os

# Path to adapted data
ADAPTED_DATA_DIR = "adapted_data"

def create_real_data_animation(output_filename='real_battlefield_animation.mp4', 
                              interpolation_steps=5,   # Number of frames between actual data points
                              max_frames=300):         # Maximum number of frames to render
    print("Starting animation creation for real data...")
    
    # Create progress bar for data loading
    loading_pbar = tqdm(total=4, desc="Loading data")
    
    # Load simulation data
    sim = BattlefieldSimulation()
    loading_pbar.update(1)
    
    sim.load_terrain_data(
        terrain_data_path=os.path.join(ADAPTED_DATA_DIR, "terrain_map.npy"),
        elevation_data_path=os.path.join(ADAPTED_DATA_DIR, "elevation_map.npy")
    )
    loading_pbar.update(1)
    
    sim.load_observation_data(
        target_csv=os.path.join(ADAPTED_DATA_DIR, "target_observations.csv"),
        blue_force_csv=os.path.join(ADAPTED_DATA_DIR, "blue_force_observations.csv")
    )
    loading_pbar.update(1)
    
    # Get all unique timestamps
    target_df = pd.read_csv(os.path.join(ADAPTED_DATA_DIR, "target_observations.csv"))
    target_df['timestamp'] = pd.to_datetime(target_df['timestamp'])
    
    # Load blue force data
    blue_force_df = pd.read_csv(os.path.join(ADAPTED_DATA_DIR, "blue_force_observations.csv"))
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
    
    print(f"Original data has {len(original_timestamps)} timestamps")
    print(f"After interpolation: {frame_count} frames")
    
    loading_pbar.update(1)
    loading_pbar.close()
    
    # Create figure and background
    fig, ax = plt.subplots(figsize=(16, 12))
    
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
        target_class = target_observations['target_class'].iloc[0] if 'target_class' in target_df.columns else 'unknown'
        
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
        force_class = force_observations['force_class'].iloc[0] if 'force_class' in blue_force_df.columns else 'unknown'
        
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
    
    # Add trails for selected targets
    trail_lines = {}  # Store trail lines for tracking movement history
    trail_length = 10  # Number of previous positions to show
    
    # Time indicator text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add terrain label in the corner
    terrain_text = ax.text(0.02, 0.02, 'Nova Scotia Battlefield Visualization',
                         transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title('Real Battlefield Data Visualization')
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
            
            # Add to appropriate list based on class
            if interp_data['class'] == 'infantry':
                infantry_positions.append((x, y))
            elif interp_data['class'] == 'light_vehicle':
                light_vehicle_positions.append((x, y))
            elif interp_data['class'] == 'heavy_vehicle':
                heavy_vehicle_positions.append((x, y))
            elif interp_data['class'] == 'uav':
                uav_positions.append((x, y))
            elif interp_data['class'] == 'civilian':
                civilian_positions.append((x, y))
            else:
                # Default to infantry if class is unknown
                infantry_positions.append((x, y))
            
            # Update trail history for this target
            if target_id not in trail_history:
                trail_history[target_id] = []
            
            # Add current position to trail
            trail_history[target_id].append((x, y))
            
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
            
            # Add to appropriate list based on class
            if interp_data['class'] == 'infantry_squad':
                blue_infantry_positions.append((x, y))
            elif interp_data['class'] == 'mechanized_patrol':
                blue_mechanized_positions.append((x, y))
            elif interp_data['class'] == 'recon_team':
                blue_recon_positions.append((x, y))
            elif interp_data['class'] == 'command_post':
                blue_command_positions.append((x, y))
            elif interp_data['class'] == 'uav_surveillance':
                blue_uav_positions.append((x, y))
            else:
                # Default to infantry if class is unknown
                blue_infantry_positions.append((x, y))
        
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
    print("\\nSaving animation...")
    
    # Use higher fps for smoother animation
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=3600)
    
    # Save the animation with progress tracking
    output_path = os.path.join(ADAPTED_DATA_DIR, output_filename)
    with tqdm(total=100, desc="Encoding video") as pbar:
        # Start the saving process
        ani.save(output_path, writer=writer, dpi=150, 
                 progress_callback=lambda i, n: pbar.update(100/n))
    
    progress_bar.close()
    print(f"Animation saved to {output_path}")
    
    return ani

if __name__ == "__main__":
    # Call the function to create the animation
    animation = create_real_data_animation(
        output_filename='real_battlefield_animation.mp4',
        interpolation_steps=5,
        max_frames=300
    )
    """
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created animation script at {script_path}")
    return script_path

def visualize_adapted_data():
    """
    Use the existing visualization code with the adapted data.
    """
    print("Visualizing adapted data...")
    
    # Initialize simulation with adapted data
    sim = BattlefieldSimulation()
    
    # Load terrain data
    sim.load_terrain_data(
        terrain_data_path=os.path.join(OUTPUT_DIR, "terrain_map.npy"),
        elevation_data_path=os.path.join(OUTPUT_DIR, "elevation_map.npy")
    )
    
    # Load observation data
    sim.load_observation_data(
        target_csv=os.path.join(OUTPUT_DIR, "target_observations.csv"),
        blue_force_csv=os.path.join(OUTPUT_DIR, "blue_force_observations.csv")
    )
    
    # Create visualizations
    # Visualize terrain
    terrain_fig = sim.visualize_terrain(show_elevation=True)
    terrain_fig.savefig(os.path.join(OUTPUT_DIR, "visualizations", "terrain_visualization.png"), dpi=300)
    plt.close(terrain_fig)
    
    # Visualize entities
    entities_fig = sim.visualize_entities(timestamp=None, show_terrain=True)
    entities_fig.savefig(os.path.join(OUTPUT_DIR, "visualizations", "entities_visualization.png"), dpi=300)
    plt.close(entities_fig)
    
    # Visualize trajectories
    trajectories_fig = sim.visualize_trajectories(target_ids=None, show_terrain=True)
    trajectories_fig.savefig(os.path.join(OUTPUT_DIR, "visualizations", "trajectories_visualization.png"), dpi=300)
    plt.close(trajectories_fig)
    
    print("Visualization complete. Files saved to:", os.path.join(OUTPUT_DIR, "visualizations"))

def main():
    """Main function to run the adaptation process"""
    # Print information about the dataset
    print_data_info()
    
    # Create basic visualizations of the raw data
    create_basic_visualizations()
    
    # Convert rasters to NumPy arrays
    raster_info = convert_rasters_to_numpy()
    
    # Adapt the location data
    location_paths = adapt_location_data(raster_info)
    
    # Create an animation script customized for the real data
    animation_script_path = create_animation_script()
    
    # Visualize the adapted data
    visualize_adapted_data()
    
    print("\nAdaptation complete! To visualize the data:")
    print(f"1. Check raw data visualizations in {os.path.join(OUTPUT_DIR, 'raw_visualizations')}")
    print(f"2. Check adapted data visualizations in {os.path.join(OUTPUT_DIR, 'visualizations')}")
    print(f"3. Run the animation script: python {animation_script_path}")
    print("\nTo run the animation:")
    print(f"python {animation_script_path}")

if __name__ == "__main__":
    main()