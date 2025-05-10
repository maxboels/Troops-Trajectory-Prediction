"""
Nova Scotia Terrain Analysis with Land Use and Forces

This script creates comprehensive terrain visualizations that combine:
1. Elevation data (topography)
2. Land use categories (forests, water, urban areas, etc.)
3. Red and blue force positions

This provides a complete battlefield picture for predicting target movements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
import os
from datetime import datetime
import rasterio
from helpers import LandUseCategory, get_raster_value_at_coords

def load_data(red_csv, blue_csv, terrain_path=None, elevation_path=None):
    """
    Load all necessary data for visualization.
    
    Args:
        red_csv: Path to red forces CSV
        blue_csv: Path to blue forces CSV
        terrain_path: Path to terrain data (NPY file)
        elevation_path: Path to elevation data (NPY file)
    
    Returns:
        Dictionary with all loaded data
    """
    data = {}
    
    # Load red forces data
    data['red_forces'] = pd.read_csv(red_csv)
    if 'datetime' in data['red_forces'].columns:
        data['red_forces']['datetime'] = pd.to_datetime(data['red_forces']['datetime'])
    
    # Load blue forces data
    data['blue_forces'] = pd.read_csv(blue_csv)
    
    # Load terrain data if available
    if terrain_path and os.path.exists(terrain_path):
        data['terrain'] = np.load(terrain_path)
        print(f"Loaded terrain data with shape {data['terrain'].shape}")
    
    # Load elevation data if available
    if elevation_path and os.path.exists(elevation_path):
        data['elevation'] = np.load(elevation_path)
        print(f"Loaded elevation data with shape {data['elevation'].shape}")
    
    return data

def create_elevation_colormap():
    """Create a topographical colormap for elevation visualization."""
    # Create custom colors for a realistic topographic map
    colors = [
        # Deep blue for water (< 0m)
        (0.00, '#0000FF'),  # Deep blue
        (0.05, '#0077FF'),  # Medium blue
        # Coastal areas and low elevations (0-100m)
        (0.10, '#90EE90'),  # Light green
        (0.15, '#ADFF2F'),  # Yellow-green
        (0.30, '#FFFF00'),  # Yellow
        # Mid elevations (100-200m)
        (0.45, '#FFA500'),  # Orange
        (0.60, '#FF4500'),  # Red-orange
        # High elevations (200-300m)
        (0.75, '#A52A2A'),  # Brown
        (0.90, '#800000'),  # Maroon
        # Highest peaks (>300m)
        (1.00, '#FFFFFF')   # White for peaks
    ]
    
    # Extract positions and RGB colors
    position = [x[0] for x in colors]
    rgb_colors = [x[1] for x in colors]
    
    # Convert hex to rgb tuples
    from matplotlib.colors import to_rgb
    rgb_tuples = [to_rgb(color) for color in rgb_colors]
    
    # Create the colormap
    return LinearSegmentedColormap.from_list('topo_cmap', list(zip(position, rgb_tuples)), N=256)

def create_landuse_colormap():
    """Create a colormap for land use categories."""
    # Define colors for each land use category based on the LandUseCategory enum
    colors = [
        '#FFFFFF',  # 0: Undefined
        '#008000',  # 1: Broadleaf Evergreen Forest - dark green
        '#228B22',  # 2: Broadleaf Deciduous Forest - forest green
        '#006400',  # 3: Needleleaf Evergreen Forest - dark green
        '#32CD32',  # 4: Needleleaf Deciduous Forest - lime green
        '#556B2F',  # 5: Mixed Forest - olive green
        '#8FBC8F',  # 6: Tree Open - dark sea green
        '#D2B48C',  # 7: Shrub - tan
        '#7CFC00',  # 8: Herbaceous - lawn green
        '#ADFF2F',  # 9: Herbaceous with Sparse Tree/Shrub - yellow-green
        '#F0E68C',  # 10: Sparse vegetation - khaki
        '#FFD700',  # 11: Cropland - gold
        '#DAA520',  # 12: Paddy field - goldenrod
        '#F4A460',  # 13: Cropland / Other Vegetation Mosaic - sandy brown
        '#2F4F4F',  # 14: Mangrove - dark slate gray
        '#00FFFF',  # 15: Wetland - cyan
        '#A0522D',  # 16: Bare area, consolidated (gravel, rock) - sienna
        '#DEB887',  # 17: Bare area, unconsolidated (sand) - burlywood
        '#A9A9A9',  # 18: Urban - dark gray
        '#FFFFFF',  # 19: Snow / Ice - white
        '#0000FF',  # 20: Water bodies - blue
    ]
    
    # Create a ListedColormap
    return ListedColormap(colors)

def read_landuse_data(tif_path, bounds):
    """
    Read land use data directly from GeoTIFF for the given bounds.
    
    Args:
        tif_path: Path to land use GeoTIFF file
        bounds: Dictionary with lat/lon bounds
    
    Returns:
        Land use data array
    """
    try:
        print(f"Attempting to read land use data from {tif_path}")
        with rasterio.open(tif_path) as src:
            print(f"Successfully opened {tif_path}")
            
            # Create a regular grid of lat/lon points
            lon_step = 0.01
            lat_step = 0.01
            
            lons = np.arange(bounds['lon_min'], bounds['lon_max'], lon_step)
            lats = np.arange(bounds['lat_min'], bounds['lat_max'], lat_step)
            
            # Initialize empty array
            landuse = np.zeros((len(lats), len(lons)), dtype=np.int32)
            
            # Sample values at each point
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    try:
                        # Use the helper function to get the land use value
                        landuse[i, j] = get_raster_value_at_coords(lat, lon, tif_path)
                    except Exception as e:
                        print(f"Error reading value at {lat}, {lon}: {e}")
                        landuse[i, j] = 0
            
            return landuse
    except Exception as e:
        print(f"Error reading land use data: {e}")
        return None

def create_elevation_landuse_composite(elevation, landuse, bounds):
    """
    Create a composite visualization with elevation and land use.
    
    Args:
        elevation: Elevation data array
        landuse: Land use data array
        bounds: Dictionary with lat/lon bounds
    
    Returns:
        Composite visualization figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # 1. Plot elevation
    elev_cmap = create_elevation_colormap()
    
    # Create proper normalization
    elev_min = np.min(elevation)
    elev_max = np.max(elevation)
    
    # Make water areas blue
    if elev_min < 0:
        elev_norm = plt.cm.colors.TwoSlopeNorm(vmin=elev_min, vcenter=0, vmax=elev_max)
    else:
        elev_norm = plt.cm.colors.Normalize(vmin=elev_min, vmax=elev_max)
    
    elev_im = ax1.imshow(
        elevation,
        cmap=elev_cmap,
        norm=elev_norm,
        extent=[bounds['lon_min'], bounds['lon_max'], bounds['lat_min'], bounds['lat_max']],
        aspect='auto',
        origin='lower'
    )
    
    # Add colorbar for elevation
    cbar1 = fig.colorbar(elev_im, ax=ax1, shrink=0.7)
    cbar1.set_label('Elevation (m)')
    
    ax1.set_title('Elevation (Topography)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # 2. Plot land use
    if landuse is not None:
        landuse_cmap = create_landuse_colormap()
        
        landuse_im = ax2.imshow(
            landuse,
            cmap=landuse_cmap,
            extent=[bounds['lon_min'], bounds['lon_max'], bounds['lat_min'], bounds['lat_max']],
            aspect='auto',
            origin='lower',
            vmin=0,
            vmax=20
        )
        
        # Create legend patches for land use categories
        landuse_legend = []
        categories = [
            (1, "Broadleaf Forest"),
            (5, "Mixed Forest"),
            (8, "Herbaceous"),
            (11, "Cropland"),
            (15, "Wetland"),
            (18, "Urban"),
            (20, "Water")
        ]
        
        for value, label in categories:
            color = landuse_cmap(value / 20)  # Normalize value to 0-1
            landuse_legend.append(Patch(facecolor=color, label=label))
        
        # Add legend for land use
        ax2.legend(handles=landuse_legend, loc='lower right', fontsize=8)
        
        ax2.set_title('Land Use Categories')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
    else:
        ax2.text(0.5, 0.5, "Land Use Data Not Available", 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Land Use Categories')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
    
    # 3. Create blended visualization
    # Use elevation data as base with land use transparency
    if landuse is not None:
        # First show elevation
        blended_im = ax3.imshow(
            elevation,
            cmap=elev_cmap,
            norm=elev_norm,
            extent=[bounds['lon_min'], bounds['lon_max'], bounds['lat_min'], bounds['lat_max']],
            aspect='auto',
            origin='lower'
        )
        
        # Overlay land use with transparency for important categories
        # Create a mask for important land use categories (water, urban, forest)
        mask = np.zeros_like(landuse, dtype=bool)
        
        # Mark water bodies
        mask = mask | (landuse == LandUseCategory.WATER_BODIES.value)
        # Mark urban areas
        mask = mask | (landuse == LandUseCategory.URBAN.value)
        # Mark forests (all types)
        for forest_val in range(1, 6):
            mask = mask | (landuse == forest_val)
        # Mark wetlands
        mask = mask | (landuse == LandUseCategory.WETLAND.value)
        
        # Create simplified land use array showing only important categories
        simplified = np.zeros_like(landuse)
        simplified[landuse == LandUseCategory.WATER_BODIES.value] = LandUseCategory.WATER_BODIES.value
        simplified[landuse == LandUseCategory.URBAN.value] = LandUseCategory.URBAN.value
        for forest_val in range(1, 6):
            simplified[landuse == forest_val] = 5  # Mixed forest
        simplified[landuse == LandUseCategory.WETLAND.value] = LandUseCategory.WETLAND.value
        
        # Overlay with transparency
        ax3.imshow(
            simplified,
            cmap=landuse_cmap,
            extent=[bounds['lon_min'], bounds['lon_max'], bounds['lat_min'], bounds['lat_max']],
            aspect='auto',
            origin='lower',
            alpha=0.3,  # Transparency
            vmin=0,
            vmax=20
        )
    else:
        # Just show elevation if no land use data
        blended_im = ax3.imshow(
            elevation,
            cmap=elev_cmap,
            norm=elev_norm,
            extent=[bounds['lon_min'], bounds['lon_max'], bounds['lat_min'], bounds['lat_max']],
            aspect='auto',
            origin='lower'
        )
    
    ax3.set_title('Combined Terrain Analysis')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    
    plt.tight_layout()
    return fig

def visualize_forces_on_terrain(elevation, red_forces, blue_forces, timestamp=None, bounds=None):
    """
    Create a visualization of red and blue forces on terrain.
    
    Args:
        elevation: Elevation data array
        red_forces: DataFrame with red forces data
        blue_forces: DataFrame with blue forces data
        timestamp: Optional specific timestamp to filter red forces
        bounds: Dictionary with lat/lon bounds
    
    Returns:
        Figure with forces on terrain
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # If no bounds provided, calculate from data
    if bounds is None:
        # Combine coordinates from both red and blue forces
        all_lons = np.concatenate([red_forces['longitude'].values, blue_forces['longitude'].values])
        all_lats = np.concatenate([red_forces['latitude'].values, blue_forces['latitude'].values])
        
        lon_min, lon_max = np.min(all_lons), np.max(all_lons)
        lat_min, lat_max = np.min(all_lats), np.max(all_lats)
        
        # Add padding
        padding = 0.05
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        lon_min -= lon_range * padding
        lon_max += lon_range * padding
        lat_min -= lat_range * padding
        lat_max += lat_range * padding
        
        bounds = {
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_min': lat_min,
            'lat_max': lat_max
        }
    
    # Plot elevation background
    elev_cmap = create_elevation_colormap()
    
    # Create proper normalization
    elev_min = np.min(elevation)
    elev_max = np.max(elevation)
    
    # Make water areas blue
    if elev_min < 0:
        elev_norm = plt.cm.colors.TwoSlopeNorm(vmin=elev_min, vcenter=0, vmax=elev_max)
    else:
        elev_norm = plt.cm.colors.Normalize(vmin=elev_min, vmax=elev_max)
    
    im = ax.imshow(
        elevation,
        cmap=elev_cmap,
        norm=elev_norm,
        extent=[bounds['lon_min'], bounds['lon_max'], bounds['lat_min'], bounds['lat_max']],
        aspect='auto',
        origin='lower'
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Elevation (m)')
    
    # Filter red forces by timestamp if provided
    if timestamp is not None and 'datetime' in red_forces.columns:
        red_forces = red_forces[red_forces['datetime'] == timestamp].copy()
        title_time = f" at {timestamp}"
    else:
        title_time = ""
    
    # Plot blue forces
    blue_scatter = ax.scatter(
        blue_forces['longitude'], 
        blue_forces['latitude'],
        s=120, 
        c='blue', 
        marker='^', 
        label='Blue Forces',
        edgecolor='black',
        zorder=10
    )
    
    # Plot red forces by category
    target_colors = {
        'tank': '#8B0000',                     # Dark red
        'armoured personnel carrier': '#FF4500',  # Orange red
        'light vehicle': '#FFA07A'              # Light salmon
    }
    
    # Check if target_class column exists
    if 'target_class' in red_forces.columns:
        for target_class, group in red_forces.groupby('target_class'):
            color = target_colors.get(target_class, 'red')
            
            ax.scatter(
                group['longitude'], 
                group['latitude'],
                s=80, 
                c=color,
                marker='o',
                label=f'{target_class.title()}',
                edgecolor='black',
                zorder=10
            )
            
            # Add target ID labels
            for _, row in group.iterrows():
                ax.annotate(
                    row['target_id'],
                    (row['longitude'], row['latitude']),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                )
    else:
        # If no target_class, plot all as red
        ax.scatter(
            red_forces['longitude'], 
            red_forces['latitude'],
            s=80, 
            c='red',
            marker='o',
            label='Red Forces',
            edgecolor='black',
            zorder=10
        )
    
    # Add grid and labels
    ax.grid(alpha=0.3)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add title
    ax.set_title(f'Nova Scotia Battlefield Visualization{title_time}', fontsize=14)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add info box
    info_text = (
        f"Red Forces: {len(red_forces)}\n"
        f"Blue Forces: {len(blue_forces)}\n"
        f"Nova Scotia Battlefield"
    )
    if timestamp is not None:
        info_text = f"Time: {timestamp}\n" + info_text
        
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    return fig

def create_comprehensive_terrain_analysis(data_dir="data", output_dir="output/terrain_analysis"):
    """
    Create a comprehensive terrain analysis with all available data.
    
    Args:
        data_dir: Directory containing data files
        output_dir: Directory to save output files
    
    Returns:
        Dictionary with paths to generated visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths
    red_csv = os.path.join(data_dir, "red_sightings.csv")
    blue_csv = os.path.join(data_dir, "blue_locations.csv")
    terrain_path = os.path.join("adapted_data", "terrain_map.npy")
    elevation_path = os.path.join("adapted_data", "elevation_map.npy")
    landuse_tif = os.path.join(data_dir, "gm_lc_v3_1_1.tif")
    
    # Load data
    print("Loading data...")
    data = load_data(red_csv, blue_csv, terrain_path, elevation_path)
    
    # Check if we have necessary data
    if 'elevation' not in data:
        print("Error: Elevation data is required but not found")
        return {}
    
    # Calculate bounds based on forces data
    all_lons = np.concatenate([
        data['red_forces']['longitude'].values, 
        data['blue_forces']['longitude'].values
    ])
    all_lats = np.concatenate([
        data['red_forces']['latitude'].values, 
        data['blue_forces']['latitude'].values
    ])
    
    lon_min, lon_max = np.min(all_lons), np.max(all_lons)
    lat_min, lat_max = np.min(all_lats), np.max(all_lats)
    
    # Add padding
    padding = 0.1
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    lon_min -= lon_range * padding
    lon_max += lon_range * padding
    lat_min -= lat_range * padding
    lat_max += lat_range * padding
    
    bounds = {
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lat_min': lat_min,
        'lat_max': lat_max
    }
    
    # Try to load land use data
    landuse = None
    if os.path.exists(landuse_tif):
        print(f"Attempting to load land use data from {landuse_tif}")
        landuse = read_landuse_data(landuse_tif, bounds)
    else:
        print(f"Land use data not found at {landuse_tif}")
    
    # Create composite visualization
    print("Creating composite visualization...")
    fig1 = create_elevation_landuse_composite(data['elevation'], landuse, bounds)
    composite_path = os.path.join(output_dir, "terrain_composite.png")
    fig1.savefig(composite_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Create forces visualization with elevation
    print("Creating forces visualization...")
    # Get latest timestamp if datetime column exists
    timestamp = None
    if 'datetime' in data['red_forces'].columns:
        timestamp = data['red_forces']['datetime'].max()
        
    fig2 = visualize_forces_on_terrain(
        data['elevation'], 
        data['red_forces'], 
        data['blue_forces'],
        timestamp=timestamp,
        bounds=bounds
    )
    forces_path = os.path.join(output_dir, "forces_on_terrain.png")
    fig2.savefig(forces_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Create animation frame for each timestamp if datetime column exists
    frames = []
    if 'datetime' in data['red_forces'].columns:
        timestamps = sorted(data['red_forces']['datetime'].unique())
        
        # Limit to a reasonable number of frames
        if len(timestamps) > 20:
            step = len(timestamps) // 20
            timestamps = timestamps[::step]
        
        print(f"Creating {len(timestamps)} frames for animation...")
        for i, ts in enumerate(timestamps):
            print(f"Creating frame {i+1}/{len(timestamps)} for {ts}")
            fig = visualize_forces_on_terrain(
                data['elevation'], 
                data['red_forces'], 
                data['blue_forces'],
                timestamp=ts,
                bounds=bounds
            )
            frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
            fig.savefig(frame_path, dpi=300, bbox_inches='tight')
            frames.append(frame_path)
            plt.close(fig)
    
    results = {
        "composite": composite_path,
        "forces": forces_path,
        "frames": frames
    }
    
    print(f"All visualizations saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    results = create_comprehensive_terrain_analysis()
    print("Visualizations created:")
    for key, path in results.items():
        if key != "frames":
            print(f"- {key}: {path}")
    if results.get("frames"):
        print(f"- {len(results['frames'])} animation frames")
