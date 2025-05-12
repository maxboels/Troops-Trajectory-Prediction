"""
Map Utilities for Nova Scotia Visualization

This module provides helper functions for handling map data and coordinates
to properly display Nova Scotia's terrain and target predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd
import os

def load_terrain_data(terrain_path, elevation_path=None):
    """
    Load terrain and elevation data with validation.
    
    Args:
        terrain_path: Path to terrain classification data
        elevation_path: Path to elevation data (optional)
    
    Returns:
        Tuple of (terrain_data, elevation_data)
    """
    terrain_data = None
    elevation_data = None
    
    # Load terrain data if available
    if terrain_path and os.path.exists(terrain_path):
        terrain_data = np.load(terrain_path)
        print(f"Loaded terrain data with shape {terrain_data.shape}")
        print(f"Terrain data range: {np.min(terrain_data)} to {np.max(terrain_data)}")
        print(f"Unique terrain values: {np.unique(terrain_data)}")
    else:
        print(f"Warning: Terrain data not found at {terrain_path}")
    
    # Load elevation data if available
    if elevation_path and os.path.exists(elevation_path):
        elevation_data = np.load(elevation_path)
        print(f"Loaded elevation data with shape {elevation_data.shape}")
        print(f"Elevation range: {np.min(elevation_data)} to {np.max(elevation_data)} meters")
    
    return terrain_data, elevation_data

def create_nova_scotia_bounds():
    """
    Create bounding box coordinates for Nova Scotia.
    
    Returns:
        Dictionary with min/max latitude and longitude
    """
    # Approximate bounds for Nova Scotia
    return {
        'lat_min': 43.5,  # Southern extent
        'lat_max': 47.0,  # Northern extent
        'lon_min': -66.5, # Eastern extent
        'lon_max': -60.0  # Western extent
    }

def georef_terrain_to_coords(terrain_data, bounds):
    """
    Georeference terrain data to lat/lon coordinates.
    
    Args:
        terrain_data: Numpy array of terrain data
        bounds: Dictionary with lat/lon bounds
    
    Returns:
        Tuple of (lats, lons, terrain_data)
    """
    if terrain_data is None:
        return None, None, None
    
    # Create coordinate grids
    height, width = terrain_data.shape
    lats = np.linspace(bounds['lat_max'], bounds['lat_min'], height)
    lons = np.linspace(bounds['lon_min'], bounds['lon_max'], width)
    
    return lats, lons, terrain_data

def create_topographic_colormap():
    """
    Create a topographic-style colormap for elevation data.
    
    Returns:
        Matplotlib colormap
    """
    # Define colors for a topographic map (similar to the reference map)
    colors = [
        (0.0, '#0000FF'),   # Deep blue for below sea level/water
        (0.1, '#6BAED6'),   # Light blue for water/coast
        (0.2, '#74C476'),   # Light green for lowlands
        (0.3, '#A1D99B'),   # Medium green 
        (0.4, '#C7E9C0'),   # Pale green
        (0.5, '#FFFFCC'),   # Light yellow
        (0.6, '#FED976'),   # Light orange
        (0.7, '#FD8D3C'),   # Orange
        (0.8, '#E31A1C'),   # Light red
        (0.9, '#BD0026'),   # Medium red
        (1.0, '#800026')    # Dark red for high elevations
    ]
    
    # Extract positions and RGB colors
    position = [x[0] for x in colors]
    rgb_colors = [x[1] for x in colors]
    
    # Convert hex to rgb tuples
    from matplotlib.colors import to_rgb
    rgb_tuples = [to_rgb(color) for color in rgb_colors]
    
    # Create the colormap
    return LinearSegmentedColormap.from_list('topo_cmap', list(zip(position, rgb_tuples)), N=256)

def create_nova_scotia_basemap(ax, terrain_data=None, elevation_data=None, bounds=None):
    """
    Create a base map of Nova Scotia with proper georeferencing.
    
    Args:
        ax: Matplotlib axis to plot on
        terrain_data: Optional terrain classification data
        elevation_data: Optional elevation data
        bounds: Optional dictionary with lat/lon bounds
    
    Returns:
        Matplotlib axis with basemap
    """
    if bounds is None:
        bounds = create_nova_scotia_bounds()
    
    # Set map extent
    ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
    ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
    
    # Add elevation data if available
    if elevation_data is not None:
        # Create a colormap for elevation
        cmap = create_topographic_colormap()
        
        # Set up normalization to make water blue
        elev_min = np.min(elevation_data)
        elev_max = np.max(elevation_data)
        
        # Make sure water areas are blue
        if elev_min < 0:
            norm = plt.cm.colors.TwoSlopeNorm(vmin=elev_min, vcenter=0, vmax=elev_max)
        else:
            norm = Normalize(vmin=elev_min, vmax=elev_max)
        
        # Plot elevation data with proper orientation
        im = ax.imshow(
            elevation_data,
            origin='lower',  # Important for correct orientation
            extent=[bounds['lon_min'], bounds['lon_max'], bounds['lat_min'], bounds['lat_max']],
            cmap=cmap,
            norm=norm,
            aspect='auto',
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Elevation (m)')
    
    # Add terrain data if available (used for land classification)
    elif terrain_data is not None:
        # Use our best guess for terrain categories based on the LandUseCategory enum
        terrain_cmap = plt.cm.colors.ListedColormap([
            '#F5DEB3',  # Agricultural/Land
            '#228B22',  # Forest 
            '#90EE90',  # Grassland
            '#808080',  # Urban
            '#0000FF',  # Water
            '#00FFFF',  # Wetland
            '#A0522D',  # Barren/Mountain
        ])
        
        # Plot terrain with correct orientation
        ax.imshow(
            terrain_data,
            origin='lower',  # Important for correct orientation
            extent=[bounds['lon_min'], bounds['lon_max'], bounds['lat_min'], bounds['lat_max']],
            cmap=terrain_cmap,
            aspect='auto',
            alpha=0.8
        )
    
    # Add grid and labels
    ax.grid(alpha=0.3)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Nova Scotia Terrain')
    
    return ax

def visualize_terrain_and_targets(terrain_data, elevation_data, target_data, blue_force_data=None):
    """
    Create a comprehensive visualization of terrain with target positions.
    
    Args:
        terrain_data: Terrain classification data
        elevation_data: Elevation data
        target_data: DataFrame with target positions
        blue_force_data: Optional DataFrame with blue force positions
    
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create bounds based on target data with some padding
    lon_min = target_data['longitude'].min() - 0.2
    lon_max = target_data['longitude'].max() + 0.2
    lat_min = target_data['latitude'].min() - 0.2
    lat_max = target_data['latitude'].max() + 0.2
    
    bounds = {
        'lon_min': lon_min,
        'lon_max': lon_max, 
        'lat_min': lat_min,
        'lat_max': lat_max
    }
    
    # Create base map
    create_nova_scotia_basemap(
        ax, 
        terrain_data=terrain_data, 
        elevation_data=elevation_data,
        bounds=bounds
    )
    
    # Plot blue forces if available
    if blue_force_data is not None:
        ax.scatter(
            blue_force_data['longitude'], 
            blue_force_data['latitude'],
            c='blue', 
            marker='^', 
            s=100, 
            label='Blue Forces',
            edgecolor='white',
            linewidth=1,
            zorder=10
        )
    
    # Plot target positions with different colors by type
    target_colors = {
        'tank': '#8B0000',                    # Dark red
        'armoured personnel carrier': '#FF4500',  # Orange red
        'light vehicle': '#FFA07A'             # Light salmon
    }
    
    # Loop through target types
    for target_type, group in target_data.groupby('target_class'):
        color = target_colors.get(target_type, 'red')
        
        # Plot target positions
        ax.scatter(
            group['longitude'],
            group['latitude'],
            c=color,
            marker='o',
            s=80,
            label=f'{target_type.title()}',
            edgecolor='white',
            linewidth=0.5,
            alpha=0.9,
            zorder=10
        )
        
        # Add target IDs as labels
        for _, row in group.iterrows():
            ax.annotate(
                row['target_id'],
                (row['longitude'], row['latitude']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8)
            )
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add title
    current_time = target_data['datetime'].max() if 'datetime' in target_data.columns else None
    title = f'Nova Scotia Battlefield - Target Positions'
    if current_time:
        title += f' at {current_time}'
    ax.set_title(title, fontsize=14)
    
    return fig

# Add more utility functions as needed
