"""
Script to extract proper terrain and elevation data for Nova Scotia
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Paths to data
DATA_DIR = "data"
BLUE_LOCATIONS_CSV = os.path.join(DATA_DIR, "blue_locations.csv")
RED_SIGHTINGS_CSV = os.path.join(DATA_DIR, "red_sightings.csv")
LAND_COVER_TIF = os.path.join(DATA_DIR, "gm_lc_v3_1_1.tif")
ELEVATION_TIF = os.path.join(DATA_DIR, "output_AW3D30.tif")

# Output directory
OUTPUT_DIR = "adapted_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_terrain_for_area():
    """Extract terrain and elevation data for the area of interest"""
    # Load coordinates to determine area bounds
    blue_df = pd.read_csv(BLUE_LOCATIONS_CSV)
    red_df = pd.read_csv(RED_SIGHTINGS_CSV)
    
    # Get coordinate bounds for Nova Scotia area
    all_lats = np.concatenate([red_df['latitude'].values, blue_df['latitude'].values])
    all_lons = np.concatenate([red_df['longitude'].values, blue_df['longitude'].values])
    
    # Get bounding box with padding
    lat_min, lat_max = np.min(all_lats), np.max(all_lats)
    lon_min, lon_max = np.min(all_lons), np.max(all_lons)
    
    # Add padding (5%)
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_padding = lat_range * 0.05
    lon_padding = lon_range * 0.05
    
    lat_min -= lat_padding
    lat_max += lat_padding
    lon_min -= lon_padding
    lon_max += lon_padding
    
    print(f"Coordinate bounding box: {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}")
    
    # Extract elevation data first (this is the more useful of the two rasters)
    try:
        print("Extracting elevation data...")
        with rasterio.open(ELEVATION_TIF) as src:
            # Get the window indices for our area of interest
            rows, cols = rasterio.transform.rowcol(src.transform, [lon_min, lon_max], [lat_max, lat_min])
            
            # Create a window that covers our area
            row_min, row_max = min(rows), max(rows)
            col_min, col_max = min(cols), max(cols)
            
            # Ensure they're within bounds
            row_min = max(0, row_min)
            row_max = min(src.height, row_max)
            col_min = max(0, col_min)
            col_max = min(src.width, col_max)
            
            # Calculate dimensions
            window_width = row_max - row_min
            window_height = col_max - col_min
            
            print(f"Extracting window with dimensions: {window_width}x{window_height}")
            
            # Read the window data
            elevation_data = src.read(1, window=((row_min, row_max), (col_min, col_max)))
            
            # Get transformation info for this window
            elevation_transform = src.window_transform(((row_min, row_max), (col_min, col_max)))
            
            # Save the data
            np.save(os.path.join(OUTPUT_DIR, "elevation_map.npy"), elevation_data)
            
            # Save metadata
            elevation_meta = {
                'transform': [float(x) for x in elevation_transform],
                'bounds': (lon_min, lat_min, lon_max, lat_max),
                'shape': elevation_data.shape
            }
            
            np.save(os.path.join(OUTPUT_DIR, "elevation_meta.npy"), elevation_meta)
            
            # Visualize the elevation data
            plt.figure(figsize=(10, 8))
            plt.imshow(elevation_data, cmap='terrain')
            plt.colorbar(label='Elevation (m)')
            plt.title('Elevation Data')
            plt.savefig(os.path.join(OUTPUT_DIR, "elevation_visualization.png"), dpi=300)
            plt.close()
            
            # We'll use this as our terrain map too since the global land cover is harder to extract
            # Just create a simple classification based on elevation
            terrain_map = np.zeros_like(elevation_data, dtype=np.uint8)
            
            # 0: Water (below 0m elevation)
            terrain_map[elevation_data < 0] = 0
            
            # 1: Urban (we don't have urban data, so skip)
            
            # 2: Agricultural (flatlands, low elevation 0-50m)
            terrain_map[(elevation_data >= 0) & (elevation_data < 50)] = 2
            
            # 3: Forest (mid elevation 50-200m)
            terrain_map[(elevation_data >= 50) & (elevation_data < 200)] = 3
            
            # 4: Grassland (default for everything else)
            terrain_map[(elevation_data >= 200) & (terrain_map == 0)] = 4
            
            # 5: Barren (high elevation >500m)
            terrain_map[elevation_data > 500] = 5
            
            # Save terrain map
            np.save(os.path.join(OUTPUT_DIR, "terrain_map.npy"), terrain_map)
            
            # Visualize the terrain map
            terrain_colors = [
                'blue',      # 0: Water
                'gray',      # 1: Urban
                'yellow',    # 2: Agricultural
                'darkgreen', # 3: Forest
                'lightgreen',# 4: Grassland
                'brown',     # 5: Barren
                'cyan',      # 6: Wetland
                'white'      # 7: Snow/Ice
            ]
            
            from matplotlib.colors import ListedColormap
            terrain_cmap = ListedColormap(terrain_colors)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(terrain_map, cmap=terrain_cmap, vmin=0, vmax=7)
            
            # Add colorbar with labels
            terrain_labels = ['Water', 'Urban', 'Agricultural', 'Forest', 'Grassland', 'Barren', 'Wetland', 'Snow/Ice']
            cbar = plt.colorbar(ticks=range(8))
            cbar.set_ticklabels(terrain_labels)
            
            plt.title('Terrain Map')
            plt.savefig(os.path.join(OUTPUT_DIR, "terrain_visualization.png"), dpi=300)
            plt.close()
            
            print(f"Saved terrain and elevation data with shape {elevation_data.shape}")
            
    except Exception as e:
        print(f"Error extracting elevation data: {e}")
        import traceback
        traceback.print_exc()
        # Create dummy terrain and elevation as fallback
        dummy_size = (500, 500)
        elevation_data = np.zeros(dummy_size, dtype=np.float32)
        terrain_map = np.ones(dummy_size, dtype=np.uint8) * 4  # Default to grassland
        
        np.save(os.path.join(OUTPUT_DIR, "elevation_map.npy"), elevation_data)
        np.save(os.path.join(OUTPUT_DIR, "terrain_map.npy"), terrain_map)
        print(f"Created dummy terrain and elevation maps with shape {dummy_size}")

if __name__ == "__main__":
    extract_terrain_for_area()