"""
Simplified adapter script for the real dataset that avoids memory issues.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from helpers import LandUseCategory

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

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
    
    # Check metadata of geospatial files without loading their full content
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

def visualize_point_data():
    """Create visualizations of point data only without loading full raster files"""
    print("Creating basic point data visualizations...")
    
    try:
        # Load CSVs
        blue_df = pd.read_csv(BLUE_LOCATIONS_CSV)
        red_df = pd.read_csv(RED_SIGHTINGS_CSV)
        red_df['datetime'] = pd.to_datetime(red_df['datetime'])
        
        # 1. Plot blue locations
        plt.figure(figsize=(10, 8))
        plt.scatter(blue_df['longitude'], blue_df['latitude'], 
                   s=100, c='blue', marker='^', edgecolor='black',
                   label='Blue Locations')
        
        # Add labels for blue locations
        for _, row in blue_df.iterrows():
            plt.text(row['longitude'], row['latitude'] + 0.01, 
                    row['name'], fontsize=8, ha='center')
        
        plt.title('Blue Force Locations')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "raw_visualizations", "blue_locations.png"), dpi=300)
        plt.close()
        
        # 2. Plot red target trajectories
        plt.figure(figsize=(12, 10))
        
        # Get unique targets and assign colors
        targets = red_df['target_id'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))
        
        # Create a coordinate bounding box to filter out outliers
        all_lats = red_df['latitude']
        all_lons = red_df['longitude']
        
        lat_min, lat_max = np.percentile(all_lats, [0.5, 99.5])  # Using percentiles to filter outliers
        lon_min, lon_max = np.percentile(all_lons, [0.5, 99.5])
        
        # Add some padding to the bounds
        lat_padding = (lat_max - lat_min) * 0.05
        lon_padding = (lon_max - lon_min) * 0.05
        
        lat_min -= lat_padding
        lat_max += lat_padding
        lon_min -= lon_padding
        lon_max += lon_padding
        
        # Plot each target's trajectory
        for i, target_id in enumerate(targets):
            target_data = red_df[red_df['target_id'] == target_id].sort_values('datetime')
            
            # Skip targets outside our bounds to focus on the main area
            if (target_data['latitude'].min() > lat_max or 
                target_data['latitude'].max() < lat_min or
                target_data['longitude'].min() > lon_max or
                target_data['longitude'].max() < lon_min):
                continue
            
            # Plot the trajectory line
            plt.plot(target_data['longitude'], target_data['latitude'], 
                    '-', color=colors[i % len(colors)], linewidth=1.5, alpha=0.7)
            
            # Plot start and end points
            plt.scatter(target_data['longitude'].iloc[0], target_data['latitude'].iloc[0], 
                       marker='o', color=colors[i % len(colors)], s=80, zorder=5,
                       label=f"{target_id} ({target_data['target_class'].iloc[0]})" if i < 10 else "")
            
            plt.scatter(target_data['longitude'].iloc[-1], target_data['latitude'].iloc[-1],
                       marker='x', color=colors[i % len(colors)], s=80, zorder=5)
        
        # Add blue locations on the same plot
        plt.scatter(blue_df['longitude'], blue_df['latitude'], 
                   s=120, c='blue', marker='^', edgecolor='black',
                   label='Blue Locations')
        
        # Set bounds to focus on the area of interest
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)
        
        plt.title('Red Targets Trajectories and Blue Locations')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        plt.savefig(os.path.join(OUTPUT_DIR, "raw_visualizations", "red_trajectories.png"), dpi=300)
        plt.close()
        
        # 3. Create heatmap of target activity
        plt.figure(figsize=(12, 10))
        
        # Use hexbin for a heatmap visualization
        hb = plt.hexbin(red_df['longitude'], red_df['latitude'], 
                      gridsize=50, cmap='inferno', alpha=0.7, bins='log')
                      
        # Add blue locations on the same plot
        plt.scatter(blue_df['longitude'], blue_df['latitude'], 
                   s=120, c='cyan', marker='^', edgecolor='black',
                   label='Blue Locations')
        
        # Set bounds to focus on the area of interest
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)
        
        plt.colorbar(hb, label='Log10(Count)')
        plt.title('Heat Map of Red Target Activity')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "raw_visualizations", "target_heatmap.png"), dpi=300)
        plt.close()
        
        # 4. Time-based visualization (show progression over time)
        # Divide the time range into several bins
        time_min = red_df['datetime'].min()
        time_max = red_df['datetime'].max()
        time_bins = 5
        
        time_edges = pd.date_range(start=time_min, end=time_max, periods=time_bins+1)
        
        plt.figure(figsize=(15, 12))
        
        # Create a subplot for each time bin
        for i in range(time_bins):
            ax = plt.subplot(2, 3, i+1)
            
            # Get data for this time bin
            bin_start = time_edges[i]
            bin_end = time_edges[i+1]
            
            bin_data = red_df[(red_df['datetime'] >= bin_start) & 
                              (red_df['datetime'] < bin_end)]
            
            # Plot all targets (faded)
            plt.scatter(red_df['longitude'], red_df['latitude'], 
                      s=10, c='gray', alpha=0.1)
            
            # Plot targets in this time bin
            plt.scatter(bin_data['longitude'], bin_data['latitude'], 
                      s=20, c='red', alpha=0.7)
            
            # Add blue locations
            plt.scatter(blue_df['longitude'], blue_df['latitude'], 
                       s=80, c='blue', marker='^', edgecolor='black')
            
            # Set bounds to focus on the area of interest
            plt.xlim(lon_min, lon_max)
            plt.ylim(lat_min, lat_max)
            
            plt.title(f'Time: {bin_start.strftime("%H:%M:%S")} - {bin_end.strftime("%H:%M:%S")}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "raw_visualizations", "time_progression.png"), dpi=300)
        plt.close()
        
        print("Basic point data visualizations created successfully")
        
    except Exception as e:
        print(f"Error creating point data visualizations: {e}")
        import traceback
        traceback.print_exc()

def extract_terrain_patch(lon_bounds, lat_bounds, output_path, downsample=10):
    """
    Extract and save a patch of the terrain data for the area of interest
    
    Args:
        lon_bounds: (min_lon, max_lon) bounds
        lat_bounds: (min_lat, max_lat) bounds
        output_path: Path to save the extracted numpy arrays
        downsample: Factor to downsample the raster data (to save memory)
    """
    print("Extracting terrain patch for area of interest...")
    
    try:
        # First create an elevation patch
        with rasterio.open(ELEVATION_TIF) as src:
            # Calculate pixel coordinates from lat/lon
            transformer = src.transform
            
            # Convert lat/lon to pixel coordinates (might be outside image bounds)
            min_row, min_col = ~transformer * (lon_bounds[0], lat_bounds[1])  # Upper left
            max_row, max_col = ~transformer * (lon_bounds[1], lat_bounds[0])  # Lower right
            
            # Ensure they're within bounds
            width = src.width
            height = src.height
            
            min_row = max(0, min(int(min_row), width-1))
            max_row = max(0, min(int(max_row), width-1))
            min_col = max(0, min(int(min_col), height-1))
            max_col = max(0, min(int(max_col), height-1))
            
            # Make sure min is less than max
            if min_row > max_row:
                min_row, max_row = max_row, min_row
            if min_col > max_col:
                min_col, max_col = max_col, min_col
            
            # Calculate window size
            win_width = max_row - min_row
            win_height = max_col - min_col
            
            # Skip if window is too small
            if win_width < 10 or win_height < 10:
                print("Window too small, using full extent...")
                min_row, min_col = 0, 0
                max_row, max_col = width, height
                win_width = max_row - min_row
                win_height = max_col - min_col
            
            # Apply downsampling by taking a slice
            sample_width = win_width // downsample
            sample_height = win_height // downsample
            
            # Create sampling windows - just take a few samples rather than the whole area
            window = Window(min_row, min_col, sample_width, sample_height)
            
            # Read data
            elevation_patch = src.read(1, window=window)
            
            # Save data
            np.save(os.path.join(output_path, "elevation_map.npy"), elevation_patch)
            print(f"Saved elevation patch with shape {elevation_patch.shape}")
        
        # Now create a land cover patch
        try:
            with rasterio.open(LAND_COVER_TIF) as src:
                # For land cover, we'll use a simplified approach - just take a portion of the image
                # This won't be geographically accurate but will give us something to work with
                width = src.width
                height = src.height
                
                # Create a small window - just a portion in the middle of the image
                window_size = 1000
                win_row = (width - window_size) // 2
                win_col = (height - window_size) // 2
                
                window = Window(win_row, win_col, window_size, window_size)
                
                # Read data
                land_cover = src.read(1, window=window)
                
                # Create a simplified terrain map based on land cover
                # Map global land cover categories to our simulation categories (0-7)
                terrain_map = np.zeros_like(land_cover, dtype=np.uint8)
                
                # Iterate over basic categories
                terrain_map[land_cover == LandUseCategory.WATER_BODIES.value] = 0  # Water
                terrain_map[land_cover == LandUseCategory.URBAN.value] = 1  # Urban
                
                # Agricultural
                terrain_map[land_cover == LandUseCategory.CROPLAND.value] = 2
                terrain_map[land_cover == LandUseCategory.PADDY_FIELD.value] = 2
                
                # Forest
                forest_classes = [
                    LandUseCategory.BROADLEAF_EVERGREEN_FOREST.value,
                    LandUseCategory.BROADLEAF_DECIDUOUS_FOREST.value,
                    LandUseCategory.NEEDLELEAF_EVERGREEN_FOREST.value,
                    LandUseCategory.NEEDLELEAF_DECIDUOUS_FOREST.value,
                    LandUseCategory.MIXED_FOREST.value
                ]
                for fc in forest_classes:
                    terrain_map[land_cover == fc] = 3
                
                # Default to grassland (4) for undefined areas
                grassland_mask = terrain_map == 0
                grassland_mask &= (land_cover != LandUseCategory.WATER_BODIES.value)
                terrain_map[grassland_mask] = 4
                
                # Save the simplified terrain map
                np.save(os.path.join(output_path, "terrain_map.npy"), terrain_map)
                print(f"Saved terrain map with shape {terrain_map.shape}")
                
        except Exception as e:
            print(f"Error processing land cover: {e}")
            # Create a simple dummy terrain map of the same size as the elevation patch
            terrain_map = np.ones_like(elevation_patch, dtype=np.uint8) * 4  # Default to grassland
            np.save(os.path.join(output_path, "terrain_map.npy"), terrain_map)
            print(f"Created dummy terrain map with shape {terrain_map.shape}")
    
    except Exception as e:
        print(f"Error extracting terrain patch: {e}")
        # Create simple dummy arrays as fallback
        dummy_size = 1000
        elevation_patch = np.zeros((dummy_size, dummy_size), dtype=np.float32)
        terrain_map = np.ones((dummy_size, dummy_size), dtype=np.uint8) * 4  # Default to grassland
        
        np.save(os.path.join(output_path, "elevation_map.npy"), elevation_patch)
        np.save(os.path.join(output_path, "terrain_map.npy"), terrain_map)
        print(f"Created dummy terrain and elevation maps with shape {(dummy_size, dummy_size)}")

def prepare_adapted_data():
    """Prepare adapted data files for visualization and modeling"""
    print("Preparing adapted data...")
    
    try:
        # Load CSV files
        blue_df = pd.read_csv(BLUE_LOCATIONS_CSV)
        red_df = pd.read_csv(RED_SIGHTINGS_CSV)
        red_df['datetime'] = pd.to_datetime(red_df['datetime'])
        
        # Get coordinate bounds to focus on the relevant area
        all_lats = np.concatenate([red_df['latitude'].values, blue_df['latitude'].values])
        all_lons = np.concatenate([red_df['longitude'].values, blue_df['longitude'].values])
        
        lat_min, lat_max = np.percentile(all_lats, [0.5, 99.5])
        lon_min, lon_max = np.percentile(all_lons, [0.5, 99.5])
        
        # Add some padding
        lat_padding = (lat_max - lat_min) * 0.05
        lon_padding = (lon_max - lon_min) * 0.05
        
        lat_min -= lat_padding
        lat_max += lat_padding
        lon_min -= lon_padding
        lon_max += lon_padding
        
        # Extract terrain patch for this area
        extract_terrain_patch(
            lon_bounds=(lon_min, lon_max),
            lat_bounds=(lat_min, lat_max),
            output_path=OUTPUT_DIR,
            downsample=10
        )
        
        # Create adapted blue force observations CSV
        blue_force_data = []
        
        # Since blue locations don't have timestamps, we'll add multiple entries 
        # spanning the time range of red observations
        time_min = red_df['datetime'].min()
        time_max = red_df['datetime'].max()
        
        # Create 10 time points evenly spaced
        time_points = pd.date_range(start=time_min, end=time_max, periods=10)
        
        # For each blue location, create entries at each time point
        for _, blue_row in blue_df.iterrows():
            # Map location name to a force type
            if 'North' in blue_row['name']:
                force_class = 'recon_team'
            elif 'Center' in blue_row['name']:
                force_class = 'command_post'
            elif 'South' in blue_row['name']:
                force_class = 'infantry_squad' 
            elif 'East' in blue_row['name']:
                force_class = 'mechanized_patrol'
            elif 'West' in blue_row['name']:
                force_class = 'uav_surveillance'
            else:
                force_class = 'infantry_squad'
            
            # Create an entry for each time point with slight position jitter
            for time_point in time_points:
                # Add slight random movement to make it look dynamic
                jitter = 0.0001  # Degree of random movement (about 10m)
                lat_jitter = blue_row['latitude'] + np.random.uniform(-jitter, jitter)
                lon_jitter = blue_row['longitude'] + np.random.uniform(-jitter, jitter)
                
                blue_force_data.append({
                    'id': f"blue_{blue_row['name'].replace(' ', '_')}",
                    'timestamp': time_point,
                    'x_coord': lon_jitter,  # Using lon for x
                    'y_coord': lat_jitter,  # Using lat for y
                    'force_class': force_class
                })
        
        # Create DataFrame and save
        blue_force_df = pd.DataFrame(blue_force_data)
        blue_force_df['timestamp'] = blue_force_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        blue_force_path = os.path.join(OUTPUT_DIR, "blue_force_observations.csv")
        blue_force_df.to_csv(blue_force_path, index=False)
        print(f"Saved adapted blue force observations to {blue_force_path}")
        
        # Create adapted red target observations CSV
        target_data = []
        
        for _, red_row in red_df.iterrows():
            # Map target class to simulation target class
            if red_row['target_class'].lower() == 'tank':
                target_class = 'heavy_vehicle'
            elif 'armoured' in red_row['target_class'].lower():
                target_class = 'light_vehicle'
            elif 'light' in red_row['target_class'].lower():
                target_class = 'light_vehicle'
            else:
                target_class = 'infantry'
            
            target_data.append({
                'id': red_row['target_id'],
                'timestamp': red_row['datetime'],
                'x_coord': red_row['longitude'],  # Using lon for x
                'y_coord': red_row['latitude'],   # Using lat for y
                'target_class': target_class
            })
        
        # Create DataFrame and save
        target_df = pd.DataFrame(target_data)
        target_df['timestamp'] = target_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        target_path = os.path.join(OUTPUT_DIR, "target_observations.csv")
        target_df.to_csv(target_path, index=False)
        print(f"Saved adapted target observations to {target_path}")
        
        print("Adapted data preparation complete!")
        
    except Exception as e:
        print(f"Error preparing adapted data: {e}")
        import traceback
        traceback.print_exc()

def create_animation_script():
    """Create a simple version of the animation script for the real data"""
    script_path = os.path.join(OUTPUT_DIR, "real_data_animation.py")
    
    script_content = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from tqdm import tqdm
from datetime import datetime, timedelta

# Output settings
OUTPUT_DIR = "adapted_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "battlefield_animation.mp4")

def create_animation():
    # Load data
    print("Loading data...")
    blue_df = pd.read_csv(os.path.join(OUTPUT_DIR, "blue_force_observations.csv"))
    red_df = pd.read_csv(os.path.join(OUTPUT_DIR, "target_observations.csv"))
    
    # Convert timestamps to datetime
    blue_df['timestamp'] = pd.to_datetime(blue_df['timestamp'])
    red_df['timestamp'] = pd.to_datetime(red_df['timestamp'])
    
    # Get all unique timestamps sorted
    all_timestamps = sorted(pd.unique(red_df['timestamp']))
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get coordinate bounds
    min_x = min(blue_df['x_coord'].min(), red_df['x_coord'].min())
    max_x = max(blue_df['x_coord'].max(), red_df['x_coord'].max())
    min_y = min(blue_df['y_coord'].min(), red_df['y_coord'].min())
    max_y = max(blue_df['y_coord'].max(), red_df['y_coord'].max())
    
    # Add some padding
    x_padding = (max_x - min_x) * 0.05
    y_padding = (max_y - min_y) * 0.05
    
    min_x -= x_padding
    max_x += x_padding
    min_y -= y_padding
    max_y += y_padding
    
    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Create empty scatter plots for different entity types
    blue_scatter = ax.scatter([], [], c='blue', s=100, marker='^', label='Blue Forces')
    red_scatter = ax.scatter([], [], c='red', s=80, marker='o', label='Red Forces')
    
    # Create time text display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))
    
    # Create paths for each red target (to show where they've been)
    paths = {}
    
    def init():
        blue_scatter.set_offsets(np.empty((0, 2)))
        red_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return blue_scatter, red_scatter, time_text
    
    def update(frame):
        # Get current timestamp
        current_time = all_timestamps[frame]
        
        # Filter data for this timestamp
        current_blue = blue_df[blue_df['timestamp'] <= current_time].drop_duplicates('id', keep='last')
        current_red = red_df[red_df['timestamp'] == current_time]
        
        # Update scatter plots
        if len(current_blue) > 0:
            blue_scatter.set_offsets(current_blue[['x_coord', 'y_coord']].values)
        else:
            blue_scatter.set_offsets(np.empty((0, 2)))
            
        if len(current_red) > 0:
            red_scatter.set_offsets(current_red[['x_coord', 'y_coord']].values)
        else:
            red_scatter.set_offsets(np.empty((0, 2)))
        
        # Update time display
        time_text.set_text(f'Time: {current_time.strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Update paths (trails) for red targets
        for idx, row in current_red.iterrows():
            target_id = row['id']
            pos = (row['x_coord'], row['y_coord'])
            
            if target_id not in paths:
                # Create new path
                paths[target_id], = ax.plot([], [], 'r-', alpha=0.3, linewidth=1)
                paths[target_id].set_data([pos[0]], [pos[1]])
            else:
                # Update existing path
                x_data, y_data = paths[target_id].get_data()
                x_data = np.append(x_data, pos[0])
                y_data = np.append(y_data, pos[1])
                paths[target_id].set_data(x_data, y_data)
        
        # Return updated artists
        artists = [blue_scatter, red_scatter, time_text]
        artists.extend(list(paths.values()))
        return artists
    
    # Determine number of frames
    n_frames = len(all_timestamps)
    
    # Create animation
    print(f"Creating animation with {n_frames} frames...")
    ani = FuncAnimation(fig, update, frames=n_frames,
                      init_func=init, blit=True, interval=100)
    
    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_title('Battlefield Visualization')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    
    # Save animation
    print("Saving animation...")
    writer = FFMpegWriter(fps=10, metadata=dict(artist='RF Visualization'), bitrate=3600)
    ani.save(OUTPUT_FILE, writer=writer, dpi=150)
    
    print(f"Animation saved to {OUTPUT_FILE}")
    plt.close()

if __name__ == "__main__":
    create_animation()
    """
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created animation script at {script_path}")
    return script_path

def main():
    """Main function to run the adaptation process"""
    # Print information about the dataset
    print_data_info()
    
    # Create simple visualizations from the point data
    visualize_point_data()
    
    # Prepare the adapted data files for visualization and modeling
    prepare_adapted_data()
    
    # Create an animation script
    animation_script_path = create_animation_script()
    
    print("\nAdaptation complete! Here's what you can do next:")
    print(f"1. Check the visualizations in {os.path.join(OUTPUT_DIR, 'raw_visualizations')}")
    print(f"2. Run the animation script with: python {animation_script_path}")
    print("3. Use the adapted data to train your model with pattern similar to your model_jammer_prediction.py file")

if __name__ == "__main__":
    main()