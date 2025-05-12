import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
import rasterio
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import box
import pyproj
from pyproj import CRS, Transformer
import zipfile
import tempfile
import glob
from pathlib import Path
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")
warnings.filterwarnings("ignore", category=FutureWarning)

class DEMDataLoader:
    """
    Digital Elevation Map (DEM) data loader.
    
    Supports downloading, loading, and processing elevation data from various sources
    including SRTM and USGS National Elevation Dataset.
    """
    
    # URLs for various elevation data sources
    DATA_SOURCES = {
        'srtm_30m': {
            'url': 'https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/',
            'resolution': 30,  # meters
            'description': 'SRTM 30m resolution global elevation data'
        },
        'copernicus_30m': {
            'url': 'https://copernicus-dem-30m.s3.amazonaws.com/',
            'resolution': 30,  # meters
            'description': 'Copernicus 30m resolution global elevation data'
        },
        'usgs_ned_10m': {
            'url': 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/',
            'resolution': 10,  # meters
            'description': 'USGS 10m resolution NED (Continental US only)'
        }
    }
    
    # Land cover dataset sources (for land use data)
    LAND_COVER_SOURCES = {
        'nlcd': {
            'url': 'https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/latest/NLCD_2019_Land_Cover_L48.zip',
            'description': 'National Land Cover Database (NLCD) for the Continental US'
        },
        'corine': {
            'url': 'https://land.copernicus.eu/pan-european/corine-land-cover/clc2018',
            'description': 'CORINE Land Cover for Europe'
        },
        'esri_global': {
            'url': 'https://www.arcgis.com/home/item.html?id=d6642f8a4f6d4685a24ae2dc0c73d4ac',
            'description': 'ESRI 2020 Global Land Cover'
        }
    }
    
    def __init__(self, cache_dir=None):
        """
        Initialize the DEM data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data files
        """
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.expanduser("~"), ".dem_data_cache")
        else:
            self.cache_dir = cache_dir
            
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Dictionary to store loaded datasets
        self.loaded_data = {}
        
    def download_sample_data(self, location="grand_canyon", source="srtm_30m"):
        """
        Download a sample DEM dataset for quick testing.
        
        Args:
            location: Predefined location name or "custom" for custom coordinates
            source: Data source to use
            
        Returns:
            Path to the downloaded DEM file
        """
        # Predefined sample locations (lat, lon, name)
        sample_locations = {
            "grand_canyon": (36.1069, -112.1129, "Grand Canyon"),
            "everest": (27.9881, 86.9250, "Mount Everest"),
            "alps": (45.9769, 7.6583, "The Alps"),
            "death_valley": (36.2679, -116.8343, "Death Valley"),
            "amazon": (-3.4653, -62.2159, "Amazon Rainforest"),
            "kilimanjaro": (-3.0674, 37.3556, "Mount Kilimanjaro")
        }
        
        if location in sample_locations:
            lat, lon, name = sample_locations[location]
            print(f"Downloading sample data for {name} ({lat}, {lon})")
        else:
            raise ValueError(f"Unknown location: {location}. Choose from {list(sample_locations.keys())}")
        
        # Different download methods based on source
        if source == "srtm_30m":
            # Determine SRTM tile based on lat/lon
            tile_x = int((lon + 180) / 5) + 1
            tile_y = int((60 - lat) / 5) + 1
            
            # SRTM tiles are named with specific format
            if lon >= 0:
                lon_prefix = "E"
            else:
                lon_prefix = "W"
            
            if lat >= 0:
                lat_prefix = "N"
            else:
                lat_prefix = "S"
            
            abs_lon = abs(int(lon))
            abs_lat = abs(int(lat))
            
            # Format the tile name
            tile_name = f"srtm_{lon_prefix}{abs_lon:03d}_{lat_prefix}{abs_lat:02d}"
            
            # Construct URL and download path
            base_url = self.DATA_SOURCES[source]['url']
            zip_url = f"{base_url}{tile_name}.zip"
            
            # Create download directory
            download_dir = os.path.join(self.cache_dir, source, location)
            os.makedirs(download_dir, exist_ok=True)
            
            # Download the file
            zip_path = os.path.join(download_dir, f"{tile_name}.zip")
            
            # Check if already downloaded
            if os.path.exists(zip_path):
                print(f"File already exists at {zip_path}")
            else:
                print(f"Downloading from {zip_url} to {zip_path}")
                try:
                    self._download_file(zip_url, zip_path)
                except Exception as e:
                    print(f"Error downloading from {zip_url}: {e}")
                    
                    # Provide alternative download instructions
                    print("\nAlternative download instructions:")
                    print("1. Visit https://earthexplorer.usgs.gov/")
                    print("2. Create an account and log in")
                    print("3. Search for coordinates:", lat, lon)
                    print("4. Select 'Digital Elevation' -> 'SRTM'")
                    print("5. Download the data manually and save to:", download_dir)
                    print("6. Extract the .tif or .hgt file to the same directory")
                    return None
            
            # Extract the zip file if it exists
            if os.path.exists(zip_path):
                # Extract the content
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                
                # Find the .tif or .hgt file
                tif_files = glob.glob(os.path.join(download_dir, "**", "*.tif"), recursive=True)
                hgt_files = glob.glob(os.path.join(download_dir, "**", "*.hgt"), recursive=True)
                
                if tif_files:
                    dem_path = tif_files[0]
                elif hgt_files:
                    dem_path = hgt_files[0]
                else:
                    print("No .tif or .hgt files found in the zip file")
                    return None
                
                print(f"DEM data extracted to {dem_path}")
                return dem_path
            
            return None
        
        elif source == "usgs_ned_10m":
            # For USGS NED, we need to find the specific region file
            # This is a simplified approach - in practice, you'd need to lookup
            # the specific URL based on the location
            print("For USGS NED data, please download manually from:")
            print("https://apps.nationalmap.gov/downloader/")
            print("Select 'Elevation Products (3DEP)' -> '1/3 arc-second DEM'")
            print(f"Search for coordinates: {lat}, {lon}")
            print(f"Save the downloaded file to: {self.cache_dir}")
            return None
        
        else:
            print(f"Download for source '{source}' not implemented")
            print("Please download data manually and place in the cache directory")
            return None
    
    def _download_file(self, url, output_path):
        """
        Download a file from URL with progress bar.
        
        Args:
            url: URL to download
            output_path: Path to save the file
        """
        # Send a GET request with stream=True to enable streaming download
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Get the total file size in bytes
        total_size = int(response.headers.get('content-length', 0))
        
        # Create a temporary file to download to
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Use tqdm for progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        temp_file.write(chunk)
                        pbar.update(len(chunk))
        
        # Move the temporary file to the final location
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.rename(temp_path, output_path)
    
    def generate_synthetic_dem(self, width=1000, height=1000, feature_count=20, output_path=None):
        """
        Generate a synthetic DEM for testing when real data isn't available.
        
        Args:
            width: Width of the DEM in pixels
            height: Height of the DEM in pixels
            feature_count: Number of terrain features to generate
            output_path: Path to save the generated DEM (None to not save)
            
        Returns:
            Dictionary with elevation array and metadata
        """
        print(f"Generating synthetic {width}x{height} DEM with {feature_count} features...")
        
        # Initialize an empty elevation array
        dem = np.zeros((height, width), dtype=np.float32)
        
        # Create multiple random features (mountains, valleys)
        for _ in range(feature_count):
            # Random feature center
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            
            # Random feature properties
            radius = np.random.randint(50, max(100, min(width, height) // 3))
            amplitude = np.random.uniform(100, 1000)  # Elevation in meters
            
            # Determine if it's a mountain (positive) or valley (negative)
            if np.random.random() < 0.7:  # 70% chance of being a mountain
                sign = 1
            else:
                sign = -1
                
            # Generate a grid of coordinates
            y, x = np.ogrid[:height, :width]
            
            # Calculate distance from center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Apply a bell curve based on distance (gaussian)
            feature = sign * amplitude * np.exp(-(dist**2) / (2 * (radius/2)**2))
            
            # Add the feature to the DEM
            dem += feature
        
        # Add some perlin-like noise for texture
        noise_scale = np.random.uniform(10, 50)
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Generate multiple frequencies of noise
        noise = np.zeros((height, width))
        for i in range(1, 5):
            freq = 2**i
            amp = 1.0 / freq
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            noise += amp * np.sin(freq * 2*np.pi * X + phase_x) * np.cos(freq * 2*np.pi * Y + phase_y)
        
        # Normalize noise to [0, 1] range
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Add scaled noise to the DEM
        dem += noise_scale * noise
        
        # Ensure minimum elevation is 0
        dem = dem - min(0, dem.min())
        
        # Create some rivers by finding low points and creating paths
        if width > 100 and height > 100:  # Only for larger DEMs
            river_count = np.random.randint(1, 5)
            for _ in range(river_count):
                # Start from a random point in the top 20% elevation
                high_points = np.where(dem > np.percentile(dem, 80))
                if len(high_points[0]) > 0:
                    idx = np.random.randint(0, len(high_points[0]))
                    y, x = high_points[0][idx], high_points[1][idx]
                    
                    # Create a river path following steepest descent
                    path_length = np.random.randint(width//10, width//2)
                    river_width = np.random.randint(1, 5)
                    
                    for _ in range(path_length):
                        # Find lowest neighbor
                        neighbors = []
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    neighbors.append((nx, ny, dem[ny, nx]))
                        
                        # Sort by elevation (ascending)
                        neighbors.sort(key=lambda p: p[2])
                        
                        # If we can't go lower, stop
                        if len(neighbors) == 0 or neighbors[0][2] >= dem[y, x]:
                            break
                            
                        # Move to the lowest neighbor
                        x, y = neighbors[0][0], neighbors[0][1]
                        
                        # Carve the river by lowering elevation in a small area
                        for dx in range(-river_width, river_width+1):
                            for dy in range(-river_width, river_width+1):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    dem[ny, nx] *= 0.95  # Lower elevation slightly
        
        # Create metadata
        metadata = {
            'width': width,
            'height': height,
            'resolution': 30,  # 30 meters per pixel
            'crs': 'EPSG:4326',  # WGS84
            'bounds': [0, 0, width * 30, height * 30],  # in meters
            'synthetic': True
        }
        
        # Save to file if requested
        if output_path:
            try:
                # Create a temporary GeoTIFF
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=dem.dtype,
                    crs=metadata['crs'],
                    transform=rasterio.transform.from_bounds(
                        metadata['bounds'][0], metadata['bounds'][1],
                        metadata['bounds'][2], metadata['bounds'][3],
                        width, height
                    )
                ) as dst:
                    dst.write(dem, 1)
                print(f"Synthetic DEM saved to {output_path}")
            except Exception as e:
                print(f"Error saving DEM to {output_path}: {e}")
        
        # Store the data
        dem_data = {
            'elevation': dem,
            'metadata': metadata
        }
        
        self.loaded_data['synthetic'] = dem_data
        return dem_data
    
    def load_dem_from_file(self, file_path, identifier=None):
        """
        Load a DEM from a file.
        
        Args:
            file_path: Path to the DEM file (.tif, .hgt, etc.)
            identifier: Optional identifier to store the loaded data
            
        Returns:
            Dictionary with elevation array and metadata
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        try:
            # Check file extension
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() in ['.tif', '.tiff']:
                # Load with rasterio
                with rasterio.open(file_path) as src:
                    dem = src.read(1).astype(np.float32)
                    
                    # Create metadata
                    metadata = {
                        'width': src.width,
                        'height': src.height,
                        'resolution': src.res[0],  # Resolution in units of CRS
                        'crs': src.crs.to_string(),
                        'bounds': src.bounds,
                        'transform': src.transform,
                        'nodata': src.nodata,
                        'synthetic': False
                    }
            
            elif ext.lower() == '.hgt':
                # Load SRTM .hgt file
                with open(file_path, 'rb') as f:
                    # Determine array size from file size
                    file_size = os.path.getsize(file_path)
                    array_size = int(np.sqrt(file_size / 2))
                    
                    # Read data as signed 16-bit integers, big-endian
                    dem = np.fromfile(f, np.dtype('>i2'), array_size**2).reshape((array_size, array_size))
                    
                    # Replace SRTM nodata value (-32768) with NaN
                    dem = dem.astype(np.float32)
                    dem[dem == -32768] = np.nan
                    
                    # Extract coordinates from filename
                    filename = os.path.basename(file_path)
                    if len(filename) >= 7:
                        lat_dir = filename[0]
                        lat = int(filename[1:3])
                        if lat_dir.lower() == 's':
                            lat = -lat
                            
                        lon_dir = filename[3]
                        lon = int(filename[4:7])
                        if lon_dir.lower() == 'w':
                            lon = -lon
                    else:
                        # Default coordinates if filename doesn't follow the convention
                        lat, lon = 0, 0
                    
                    # Create metadata - SRTM is always 1 arc-second (about 30m) or 3 arc-seconds (about 90m)
                    res = 30 if array_size in [3601, 1201] else 90
                    
                    metadata = {
                        'width': array_size,
                        'height': array_size,
                        'resolution': res,
                        'crs': 'EPSG:4326',  # WGS84
                        'bounds': [lon, lat, lon + 1, lat + 1],  # 1 degree tiles
                        'synthetic': False
                    }
            
            else:
                print(f"Unsupported file format: {ext}")
                return None
            
            # Store the data
            dem_data = {
                'elevation': dem,
                'metadata': metadata
            }
            
            if identifier:
                self.loaded_data[identifier] = dem_data
            else:
                # Use filename as identifier
                self.loaded_data[os.path.basename(file_path)] = dem_data
            
            print(f"DEM loaded from {file_path}")
            print(f"Dimensions: {dem.shape}, Resolution: {metadata['resolution']}m")
            print(f"Elevation range: {np.nanmin(dem):.1f}m to {np.nanmax(dem):.1f}m")
            
            return dem_data
            
        except Exception as e:
            print(f"Error loading DEM from {file_path}: {e}")
            return None
    
    def generate_synthetic_land_use(self, dem_data, output_path=None):
        """
        Generate synthetic land use data based on a DEM.
        
        Args:
            dem_data: DEM data dictionary as returned by load_dem_from_file
            output_path: Path to save the generated land use (None to not save)
            
        Returns:
            Dictionary with land use array and metadata
        """
        if dem_data is None:
            print("No DEM data provided")
            return None
        
        # Extract DEM
        dem = dem_data['elevation']
        metadata = dem_data['metadata']
        
        height, width = dem.shape
        print(f"Generating synthetic land use for {width}x{height} DEM...")
        
        # Define land use classes
        # 0: Water
        # 1: Urban
        # 2: Agricultural
        # 3: Forest
        # 4: Grassland
        # 5: Barren
        # 6: Wetland
        # 7: Snow/Ice
        
        # Initialize land use array
        land_use = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate slope from DEM
        dx, dy = np.gradient(dem)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Normalize elevation to 0-1 range
        norm_elevation = (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem))
        
        # Create masks based on elevation and slope
        water_mask = (norm_elevation < 0.1) & (slope < 0.1)  # Low elevation, low slope
        urban_mask = np.zeros_like(dem, dtype=bool)
        forest_mask = (norm_elevation > 0.3) & (norm_elevation < 0.8)  # Mid elevation
        grass_mask = (norm_elevation > 0.1) & (norm_elevation < 0.4) & (slope < 0.2)  # Low to mid elevation, low slope
        barren_mask = (norm_elevation > 0.7) & (slope > 0.3)  # High elevation, high slope
        ag_mask = (norm_elevation > 0.1) & (norm_elevation < 0.3) & (slope < 0.1)  # Low to mid elevation, very low slope
        wetland_mask = (norm_elevation < 0.15) & (slope < 0.05) & (~water_mask)  # Very low but not water
        snow_mask = (norm_elevation > 0.9)  # Very high elevation
        
        # Add urban centers
        num_urban_centers = max(1, width * height // 1000000)  # Scale with DEM size
        for _ in range(num_urban_centers):
            # Pick random point in lower elevation areas
            coords = np.where((norm_elevation < 0.4) & (slope < 0.15))
            if len(coords[0]) > 0:
                idx = np.random.randint(0, len(coords[0]))
                center_y, center_x = coords[0][idx], coords[1][idx]
                
                # Create urban area with random size
                radius = np.random.randint(width//50, width//20)
                for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
                    for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < radius:
                            urban_mask[y, x] = True
        
        # Apply masks to create land use
        land_use[water_mask] = 0
        land_use[urban_mask] = 1
        land_use[ag_mask] = 2
        land_use[forest_mask] = 3
        land_use[grass_mask] = 4
        land_use[barren_mask] = 5
        land_use[wetland_mask] = 6
        land_use[snow_mask] = 7
        
        # Clean up edge cases and special assignments
        # Forests are more likely near water
        near_water = np.zeros_like(dem, dtype=bool)
        water_y, water_x = np.where(water_mask)
        for i in range(len(water_y)):
            y, x = water_y[i], water_x[i]
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        near_water[ny, nx] = True
        
        # More forests near water
        forest_near_water = near_water & (~water_mask) & (norm_elevation < 0.5) & (np.random.rand(height, width) < 0.7)
        land_use[forest_near_water] = 3
        
        # Add some randomness to make it look more natural
        random_mask = np.random.rand(height, width) < 0.1  # 10% random noise
        random_classes = np.random.randint(0, 8, size=(height, width))
        land_use[random_mask] = random_classes[random_mask]
        
        # Create metadata
        land_use_metadata = metadata.copy()
        land_use_metadata['classes'] = {
            0: 'Water',
            1: 'Urban',
            2: 'Agricultural',
            3: 'Forest',
            4: 'Grassland',
            5: 'Barren',
            6: 'Wetland',
            7: 'Snow/Ice'
        }
        
        # Save to file if requested
        if output_path:
            try:
                # Create a GeoTIFF
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=land_use.dtype,
                    crs=metadata.get('crs', 'EPSG:4326'),
                    transform=metadata.get('transform', rasterio.transform.from_bounds(
                        metadata['bounds'][0], metadata['bounds'][1],
                        metadata['bounds'][2], metadata['bounds'][3],
                        width, height
                    ))
                ) as dst:
                    dst.write(land_use, 1)
                print(f"Synthetic land use saved to {output_path}")
            except Exception as e:
                print(f"Error saving land use to {output_path}: {e}")
        
        # Store the data
        land_use_data = {
            'land_use': land_use,
            'metadata': land_use_metadata
        }
        
        self.loaded_data['land_use'] = land_use_data
        return land_use_data
    
    def visualize_data(self, identifier=None, plot_dem=True, plot_land_use=True, hillshade=True, 
                      figsize=(12, 10), show_plot=True, save_path=None):
        """
        Visualize loaded DEM and/or land use data.
        
        Args:
            identifier: Identifier for the data to visualize (None for the first available)
            plot_dem: Whether to plot DEM
            plot_land_use: Whether to plot land use
            hillshade: Whether to apply hillshade to DEM visualization
            figsize: Figure size
            show_plot: Whether to display the plot
            save_path: Path to save the visualization (None to not save)
            
        Returns:
            Matplotlib figure
        """
        # Find data to visualize
        dem_data = None
        land_use_data = None
        
        if identifier and identifier in self.loaded_data:
            if 'elevation' in self.loaded_data[identifier]:
                dem_data = self.loaded_data[identifier]
        else:
            # Use the first available DEM data
            for key, data in self.loaded_data.items():
                if 'elevation' in data:
                    dem_data = data
                    identifier = key
                    break
        
        # Check for land use data
        if 'land_use' in self.loaded_data:
            land_use_data = self.loaded_data['land_use']
        
        # Determine what to plot
        if not plot_dem and not plot_land_use:
            print("Nothing to plot - set plot_dem and/or plot_land_use to True")
            return None
        
        if plot_dem and dem_data is None:
            print("No DEM data available for visualization")
            plot_dem = False
            
        if plot_land_use and land_use_data is None:
            print("No land use data available for visualization")
            plot_land_use = False
            
        if not plot_dem and not plot_land_use:
            return None
        
        # Determine figure layout
        if plot_dem and plot_land_use:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax_dem = axes[0]
            ax_land_use = axes[1]
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            if plot_dem:
                ax_dem = ax
                ax_land_use = None
            else:
                ax_land_use = ax
                ax_dem = None
        
        # Plot DEM
        if plot_dem:
            dem = dem_data['elevation']
            
            if hillshade:
                # Calculate hillshade
                from matplotlib.colors import LightSource
                ls = LightSource(azdeg=315, altdeg=45)
                
                # Get colormap
                cmap = plt.cm.terrain
                
                # Combine hillshade and colormap
                rgb = ls.shade(dem, cmap=cmap, blend_mode='soft', vert_exag=0.3)
                ax_dem.imshow(rgb)
            else:
                # Plot without hillshade
                im = ax_dem.imshow(dem, cmap='terrain')
                plt.colorbar(im, ax=ax_dem, label='Elevation (m)')
            
            ax_dem.set_title(f"Digital Elevation Model{' - ' + identifier if identifier else ''}")
            ax_dem.set_xlabel("X")
            ax_dem.set_ylabel("Y")
            
            # Add elevation range info
            min_elev = np.nanmin(dem)
            max_elev = np.nanmax(dem)
            ax_dem.text(0.05, 0.05, f"Elevation range: {min_elev:.1f}m - {max_elev:.1f}m",
                       transform=ax_dem.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        # Plot land use
        if plot_land_use and land_use_data:
            land_use = land_use_data['land_use']
            metadata = land_use_data['metadata']
            
            # Get class names
            class_names = metadata.get('classes', {})
            
            # Create custom colormap
            colors = [
                (0.0, 0.0, 0.8),      # 0: Water (blue)
                (0.7, 0.0, 0.0),      # 1: Urban (red)
                (1.0, 1.0, 0.0),      # 2: Agricultural (yellow)
                (0.0, 0.5, 0.0),      # 3: Forest (green)
                (0.8, 0.8, 0.4),      # 4: Grassland (tan)
                (0.7, 0.7, 0.7),      # 5: Barren (gray)
                (0.0, 0.7, 0.7),      # 6: Wetland (cyan)
                (1.0, 1.0, 1.0)       # 7: Snow/Ice (white)
            ]
            
            from matplotlib.colors import ListedColormap
            land_cmap = ListedColormap(colors)
            
            # Plot
            im = ax_land_use.imshow(land_use, cmap=land_cmap, vmin=0, vmax=len(colors)-1)
            
            # Add colorbar with class names
            cbar = plt.colorbar(im, ax=ax_land_use)
            
            # Set colorbar ticks and labels if class names are available
            if class_names:
                tick_locs = np.arange(len(class_names)) + 0.5
                cbar.set_ticks(tick_locs)
                cbar.set_ticklabels([class_names.get(i, f"Class {i}") for i in range(len(class_names))])
            
            ax_land_use.set_title("Land Use Classification")
            ax_land_use.set_xlabel("X")
            ax_land_use.set_ylabel("Y")
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def prepare_data_for_simulation(self, dem_identifier=None, land_use_identifier='land_use', 
                                   output_dir='simulation_data'):
        """
        Prepare DEM and land use data for the battlefield simulation.
        
        Args:
            dem_identifier: Identifier for the DEM data (None for the first available)
            land_use_identifier: Identifier for the land use data
            output_dir: Directory to save the prepared data
            
        Returns:
            Dictionary with paths to the prepared data files
        """
        # Find DEM data
        dem_data = None
        if dem_identifier and dem_identifier in self.loaded_data:
            if 'elevation' in self.loaded_data[dem_identifier]:
                dem_data = self.loaded_data[dem_identifier]
        else:
            # Use the first available DEM data
            for key, data in self.loaded_data.items():
                if 'elevation' in data:
                    dem_data = data
                    dem_identifier = key
                    break
        
        if dem_data is None:
            print("No DEM data available. Generate or load DEM data first.")
            return None
        
        # Find land use data
        land_use_data = None
        if land_use_identifier in self.loaded_data:
            land_use_data = self.loaded_data[land_use_identifier]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save DEM as numpy array
        dem_path = os.path.join(output_dir, 'elevation_map.npy')
        np.save(dem_path, dem_data['elevation'])
        
        # Save metadata as JSON
        import json
        metadata_path = os.path.join(output_dir, 'dem_metadata.json')
        with open(metadata_path, 'w') as f:
            # Convert metadata to JSON-serializable format
            metadata = dem_data['metadata'].copy()
            if 'transform' in metadata:
                metadata['transform'] = list(metadata['transform'])
            if 'bounds' in metadata and hasattr(metadata['bounds'], '__iter__'):
                metadata['bounds'] = list(metadata['bounds'])
            
            json.dump(metadata, f, indent=2)
        
        # Save land use if available
        land_use_path = None
        if land_use_data is not None:
            land_use_path = os.path.join(output_dir, 'terrain_map.npy')
            np.save(land_use_path, land_use_data['land_use'])
            
            # Save land use metadata
            land_use_metadata_path = os.path.join(output_dir, 'land_use_metadata.json')
            with open(land_use_metadata_path, 'w') as f:
                # Convert metadata to JSON-serializable format
                metadata = land_use_data['metadata'].copy()
                if 'transform' in metadata:
                    metadata['transform'] = list(metadata['transform'])
                if 'bounds' in metadata and hasattr(metadata['bounds'], '__iter__'):
                    metadata['bounds'] = list(metadata['bounds'])
                
                json.dump(metadata, f, indent=2)
        
        print(f"Data prepared for simulation and saved to {output_dir}")
        return {
            'dem_path': dem_path,
            'dem_metadata_path': metadata_path,
            'land_use_path': land_use_path
        }
    
    def load_prepared_data(self, data_dir='simulation_data'):
        """
        Load previously prepared simulation data.
        
        Args:
            data_dir: Directory with prepared data
            
        Returns:
            Dictionary with loaded data arrays and metadata
        """
        # Check if directory exists
        if not os.path.isdir(data_dir):
            print(f"Data directory not found: {data_dir}")
            return None
        
        # Look for DEM and land use files
        dem_path = os.path.join(data_dir, 'elevation_map.npy')
        dem_metadata_path = os.path.join(data_dir, 'dem_metadata.json')
        land_use_path = os.path.join(data_dir, 'terrain_map.npy')
        land_use_metadata_path = os.path.join(data_dir, 'land_use_metadata.json')
        
        # Load data if files exist
        result = {}
        
        if os.path.exists(dem_path):
            try:
                elevation = np.load(dem_path)
                result['elevation'] = elevation
                
                # Load metadata if available
                if os.path.exists(dem_metadata_path):
                    import json
                    with open(dem_metadata_path, 'r') as f:
                        dem_metadata = json.load(f)
                    result['dem_metadata'] = dem_metadata
                
                print(f"Loaded elevation data with shape {elevation.shape}")
            except Exception as e:
                print(f"Error loading elevation data: {e}")
        
        if os.path.exists(land_use_path):
            try:
                land_use = np.load(land_use_path)
                result['land_use'] = land_use
                
                # Load metadata if available
                if os.path.exists(land_use_metadata_path):
                    import json
                    with open(land_use_metadata_path, 'r') as f:
                        land_use_metadata = json.load(f)
                    result['land_use_metadata'] = land_use_metadata
                
                print(f"Loaded land use data with shape {land_use.shape}")
            except Exception as e:
                print(f"Error loading land use data: {e}")
        
        if not result:
            print("No valid data found in the directory")
            return None
        
        return result

def download_and_prepare_sample_data(cache_dir=None, sample_location="grand_canyon"):
    """
    Download and prepare a sample dataset for the hackathon.
    
    Args:
        cache_dir: Directory to cache downloaded data (None for default)
        sample_location: Sample location name
        
    Returns:
        Dictionary with paths to the prepared data files
    """
    # Create data loader
    loader = DEMDataLoader(cache_dir=cache_dir)
    
    # Try to download real data
    print(f"Attempting to download sample elevation data for {sample_location}...")
    dem_path = loader.download_sample_data(location=sample_location)
    
    # If real data is available, load it
    if dem_path and os.path.exists(dem_path):
        dem_data = loader.load_dem_from_file(dem_path)
    else:
        # Generate synthetic data if download fails
        print("Generating synthetic data instead...")
        dem_data = loader.generate_synthetic_dem(
            width=1000, height=1000, feature_count=30,
            output_path="synthetic_dem.tif"
        )
    
    # Generate synthetic land use based on the DEM
    land_use_data = loader.generate_synthetic_land_use(
        dem_data, output_path="synthetic_land_use.tif"
    )
    
    # Visualize the data
    loader.visualize_data(
        plot_dem=True, plot_land_use=True, hillshade=True,
        figsize=(16, 8), save_path="terrain_visualization.png"
    )
    
    # Prepare data for simulation
    data_paths = loader.prepare_data_for_simulation(output_dir="simulation_data")
    
    return data_paths

if __name__ == "__main__":
    # Initialize loader
    loader = DEMDataLoader()

    # Option 1: Try to download real terrain data (may not work if behind firewall)
    try:
        dem_path = loader.download_sample_data(location="grand_canyon")
        dem_data = loader.load_dem_from_file(dem_path)
    except:
        # Option 2: Generate synthetic terrain data
        dem_data = loader.generate_synthetic_dem(
            width=1000, height=1000, feature_count=30,
            output_path="synthetic_dem.tif"
        )

    # Generate synthetic land use data based on the DEM
    land_use_data = loader.generate_synthetic_land_use(
        dem_data, output_path="synthetic_land_use.tif"
    )

    # Visualize the data
    loader.visualize_data(
        plot_dem=True, plot_land_use=True, 
        save_path="terrain_visualization.png"
    )

    # Prepare data for simulation
    data_paths = loader.prepare_data_for_simulation(output_dir="simulation_data")
    print("Data prepared for simulation:", data_paths)