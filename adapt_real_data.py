"""
Adapter script to convert real dataset into a format compatible with existing visualization code.
"""
import os
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from battlefield_simulation import BattlefieldSimulation
from elevation_data_loader import DEMDataLoader

# Paths to the real data
DATA_DIR = "data"
BLUE_LOCATIONS_CSV = os.path.join(DATA_DIR, "blue_locations.csv")
RED_SIGHTINGS_CSV = os.path.join(DATA_DIR, "red_sightings.csv")
LAND_COVER_TIF = os.path.join(DATA_DIR, "gm_lc_v3_1_1.tif")
ELEVATION_TIF = os.path.join(DATA_DIR, "output_AW3D30.tif")

# Paths for adapted data
OUTPUT_DIR = "adapted_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_data_info():
    """Print information about the new dataset to understand its structure"""
    # Check CSV files
    try:
        blue_df = pd.read_csv(BLUE_LOCATIONS_CSV)
        print(f"Blue locations file has {len(blue_df)} rows and columns: {blue_df.columns.tolist()}")
        print("First few rows of blue locations:")
        print(blue_df.head())
        print()
    except Exception as e:
        print(f"Error reading blue locations: {e}")
    
    try:
        red_df = pd.read_csv(RED_SIGHTINGS_CSV)
        print(f"Red sightings file has {len(red_df)} rows and columns: {red_df.columns.tolist()}")
        print("First few rows of red sightings:")
        print(red_df.head())
        print()
    except Exception as e:
        print(f"Error reading red sightings: {e}")
    
    # Check geospatial files
    try:
        with rasterio.open(LAND_COVER_TIF) as src:
            print(f"Land cover raster: {src.width}x{src.height} pixels, {src.count} bands")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print()
    except Exception as e:
        print(f"Error reading land cover TIF: {e}")
    
    try:
        with rasterio.open(ELEVATION_TIF) as src:
            print(f"Elevation raster: {src.width}x{src.height} pixels, {src.count} bands")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print()
    except Exception as e:
        print(f"Error reading elevation TIF: {e}")

def adapt_terrain_data():
    """
    Convert TIFF files to the format expected by the visualization code.
    Uses DEMDataLoader to process terrain data.
    """
    print("Adapting terrain data...")
    
    # Initialize the DEM loader
    loader = DEMDataLoader(cache_dir=OUTPUT_DIR)
    
    # Load and process elevation data
    dem_data = loader.load_dem_from_file(ELEVATION_TIF, identifier="real_elevation")
    
    # Load and process land cover data
    # For land cover, we need to map the values to match the expected classes
    # in the simulation code
    with rasterio.open(LAND_COVER_TIF) as src:
        land_cover = src.read(1)
        # Create a mapping based on the real data's classification scheme
        # This will need to be adjusted based on the actual land cover classes
        # in your data
        land_use_classes = {
            0: 0,  # Water
            1: 1,  # Urban
            2: 2,  # Agricultural
            3: 3,  # Forest
            4: 4,  # Grassland
            5: 5,  # Barren
            6: 6,  # Wetland
            7: 7   # Snow/Ice
        }
        
        # Map the values (this is simplified - adjust based on real data)
        land_use = land_cover.copy()
        # If your land cover uses different class codes, modify this mapping
        
        # Save as NumPy array
        land_use_path = os.path.join(OUTPUT_DIR, "terrain_map.npy")
        np.save(land_use_path, land_use)
        print(f"Saved terrain map to {land_use_path}")
    
    # Prepare for simulation
    data_paths = loader.prepare_data_for_simulation(
        output_dir=OUTPUT_DIR
    )
    
    return data_paths

def adapt_location_data():
    """
    Convert the location CSV files to the format expected by the visualization code.
    """
    print("Adapting location data...")
    
    # Load the data
    blue_df = pd.read_csv(BLUE_LOCATIONS_CSV)
    red_df = pd.read_csv(RED_SIGHTINGS_CSV)
    
    # Adapt blue locations to match blue_force_observations.csv format
    # This will need to be adjusted based on the columns in your data
    blue_force_data = []
    
    # Example conversion (modify based on your actual column names)
    for _, row in blue_df.iterrows():
        # Assuming blue_df has columns like timestamp, id, lat, lon, etc.
        blue_force_data.append({
            'id': row.get('id', f"blue-{_}"),
            'timestamp': row.get('timestamp', datetime.now().isoformat()),
            'x_coord': row.get('x', row.get('lon', 0)),
            'y_coord': row.get('y', row.get('lat', 0)),
            'force_class': row.get('type', 'infantry_squad')
            # Add other expected columns with default values if needed
        })
    
    # Adapt red sightings to match target_observations.csv format
    target_data = []
    
    # Example conversion (modify based on your actual column names)
    for _, row in red_df.iterrows():
        # Assuming red_df has columns like timestamp, id, lat, lon, etc.
        target_data.append({
            'id': row.get('id', f"red-{_}"),
            'timestamp': row.get('timestamp', datetime.now().isoformat()),
            'x_coord': row.get('x', row.get('lon', 0)),
            'y_coord': row.get('y', row.get('lat', 0)),
            'target_class': row.get('type', 'infantry')
            # Add other expected columns with default values if needed
        })
    
    # Save to CSV files
    blue_force_df = pd.DataFrame(blue_force_data)
    target_df = pd.DataFrame(target_data)
    
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
    os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)
    
    # Visualize terrain
    terrain_fig = sim.visualize_terrain(show_elevation=True)
    terrain_fig.savefig(os.path.join(OUTPUT_DIR, "visualizations", "terrain_visualization.png"))
    
    # Visualize entities
    entities_fig = sim.visualize_entities(timestamp=None, show_terrain=True)
    entities_fig.savefig(os.path.join(OUTPUT_DIR, "visualizations", "entities_visualization.png"))
    
    # Visualize trajectories
    trajectories_fig = sim.visualize_trajectories(target_ids=None, show_terrain=True)
    trajectories_fig.savefig(os.path.join(OUTPUT_DIR, "visualizations", "trajectories_visualization.png"))
    
    print("Visualization complete. Files saved to:", os.path.join(OUTPUT_DIR, "visualizations"))

def main():
    """Main function to run the adaptation process"""
    # Print information about the dataset
    print_data_info()
    
    # Adapt the terrain data
    terrain_paths = adapt_terrain_data()
    
    # Adapt the location data
    location_paths = adapt_location_data()
    
    # Visualize the adapted data
    visualize_adapted_data()
    
    print("Adaptation complete!")

if __name__ == "__main__":
    main()