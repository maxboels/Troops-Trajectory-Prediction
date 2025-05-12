"""
Fixed Nova Scotia Map Example

This script demonstrates the corrected terrain mapping and visualization
to properly align with the actual topography of Nova Scotia.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Import our utility modules
from map_utils import load_terrain_data, create_nova_scotia_basemap, visualize_terrain_and_targets
from target_visualization import TargetVisualizer
from target_movement_prediction import load_and_process_data

def create_fixed_visualizations():
    """Create visualizations with fixed terrain mapping."""
    
    # Setup paths
    data_dir = "data"
    output_dir = "output/fixed_maps"
    terrain_path = "adapted_data/terrain_map.npy"
    elevation_path = "adapted_data/elevation_map.npy"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_and_process_data(
        target_csv=os.path.join(data_dir, "red_sightings.csv"),
        blue_csv=os.path.join(data_dir, "blue_locations.csv"),
        terrain_path=terrain_path,
        elevation_path=elevation_path
    )
    
    target_data = data['target_data']
    blue_force_data = data['blue_force_data']
    terrain_data = data['terrain_data']
    elevation_data = data['elevation_data']
    
    # Ensure datetime column is parsed properly
    if 'datetime' in target_data.columns and not pd.api.types.is_datetime64_any_dtype(target_data['datetime']):
        target_data['datetime'] = pd.to_datetime(target_data['datetime'])
    
    print(f"Loaded target data with {len(target_data)} points")
    
    # Create visualizer with fixed terrain mapping
    visualizer = TargetVisualizer(terrain_data, elevation_data)
    
    # 1. Create a terrain-only map with proper mapping
    print("Creating terrain-only map...")
    fig, ax = plt.subplots(figsize=(12, 10))
    create_nova_scotia_basemap(ax, terrain_data=terrain_data, elevation_data=elevation_data)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "terrain_only.png"), dpi=300, bbox_inches='tight')
    
    # 2. Create a map with terrain and targets
    print("Creating terrain and targets map...")
    fig = visualize_terrain_and_targets(
        terrain_data=terrain_data,
        elevation_data=elevation_data,
        target_data=target_data,
        blue_force_data=blue_force_data
    )
    fig.savefig(os.path.join(output_dir, "terrain_with_targets.png"), dpi=300, bbox_inches='tight')
    
    # Choose a timestamp for prediction visualization
    timestamps = sorted(target_data['datetime'].unique())
    mid_idx = len(timestamps) // 2
    timestamp = timestamps[mid_idx]
    print(f"Using timestamp: {timestamp}")
    
    # 3. Create a global terrain visualization with target data (fixed)
    print("Creating battlefield visualization...")
    
    # Create special figure for the full battlefield view
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get coordinate bounds
    lon_min, lon_max = target_data['longitude'].min(), target_data['longitude'].max()
    lat_min, lat_max = target_data['latitude'].min(), target_data['latitude'].max()
    
    # Add padding
    padding = 0.05
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    lon_min -= lon_range * padding
    lon_max += lon_range * padding
    lat_min -= lat_range * padding
    lat_max += lat_range * padding
    
    # Set explicit axis limits
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    
    # Add elevation data with correct colormap
    if elevation_data is not None:
        # Get the proper topographic colormap
        cmap = visualizer.elevation_cmap
        
        # Set up the normalization to handle water areas correctly
        vmin = np.min(elevation_data)
        vmax = np.max(elevation_data)
        
        # Ensure water is blue
        if vmin < 0:
            norm = plt.cm.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Plot the elevation with correct orientation
        im = ax.imshow(
            elevation_data,
            cmap=cmap,
            norm=norm,
            extent=[lon_min, lon_max, lat_min, lat_max],
            aspect='auto',
            origin='lower',  # Proper orientation
            alpha=0.8,
            zorder=0
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Elevation (m)')
    
    # Plot blue forces
    if blue_force_data is not None:
        ax.scatter(
            blue_force_data['longitude'], blue_force_data['latitude'],
            c='blue', s=120, marker='^', label='Blue Forces',
            zorder=10, edgecolor='black'
        )
    
    # Plot red forces at the timestamp
    current_red = target_data[target_data['datetime'] == timestamp]
    ax.scatter(
        current_red['longitude'], current_red['latitude'],
        c='red', s=80, marker='o', label='Red Forces',
        zorder=10, edgecolor='black'
    )
    
    # Add labels for targets
    for _, row in current_red.iterrows():
        target_class = row['target_class'] if 'target_class' in row else 'Unknown'
        ax.annotate(
            f"ID: {row['target_id']}: {target_class}",
            (row['longitude'], row['latitude']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
        )
    
    # Add grid and labels
    ax.grid(alpha=0.3, zorder=1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add title
    ax.set_title(f'Nova Scotia Battlefield Visualization - {timestamp}',
                fontsize=14)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add info text
    info_text = (
        f"Time: {timestamp}\n"
        f"Red Forces: {len(current_red)}\n"
        f"Blue Forces: {len(blue_force_data)}\n"
        f"Nova Scotia Battlefield"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "battlefield_visualization.png"), dpi=300, bbox_inches='tight')
    
    print(f"All visualizations saved to {output_dir}")
    return {
        "terrain_only": os.path.join(output_dir, "terrain_only.png"),
        "terrain_with_targets": os.path.join(output_dir, "terrain_with_targets.png"),
        "battlefield": os.path.join(output_dir, "battlefield_visualization.png")
    }

if __name__ == "__main__":
    create_fixed_visualizations()
