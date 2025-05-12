"""
Nova Scotia Interactive Terrain and Force Visualization

This script creates an interactive visualization with terrain features and forces.
Uses helpers.py functions to access land use directly from the raster files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
import os
import time
from datetime import datetime
from helpers import LandUseCategory, get_raster_value_at_coords, is_water, is_forest, get_altitude

class TerrainForceVisualizer:
    """Class to create interactive terrain and force visualizations."""
    
    def __init__(self, data_dir="data"):
        """Initialize with data paths."""
        self.data_dir = data_dir
        
        # Define file paths
        self.red_csv = os.path.join(data_dir, "red_sightings.csv")
        self.blue_csv = os.path.join(data_dir, "blue_locations.csv")
        self.terrain_path = os.path.join("adapted_data", "terrain_map.npy")
        self.elevation_path = os.path.join("adapted_data", "elevation_map.npy")
        self.landuse_tif = os.path.join(data_dir, "gm_lc_v3_1_1.tif")
        self.elevation_tif = os.path.join(data_dir, "output_AW3D30.tif")
        
        # Load data
        self.load_data()
        
        # Create colormaps
        self.elevation_cmap = self.create_elevation_colormap()
        self.landuse_cmap = self.create_landuse_colormap()
        
        # Set target colors
        self.target_colors = {
            'tank': '#8B0000',                     # Dark red
            'armoured personnel carrier': '#FF4500',  # Orange red
            'light vehicle': '#FFA07A'              # Light salmon
        }
    
    def load_data(self):
        """Load required data files."""
        # Load red forces
        self.red_forces = pd.read_csv(self.red_csv)
        if 'datetime' in self.red_forces.columns:
            self.red_forces['datetime'] = pd.to_datetime(self.red_forces['datetime'])
        
        # Load blue forces
        self.blue_forces = pd.read_csv(self.blue_csv)
        
        # Load elevation data if available
        if os.path.exists(self.elevation_path):
            self.elevation = np.load(self.elevation_path)
            print(f"Loaded elevation data with shape {self.elevation.shape}")
        else:
            self.elevation = None
            print(f"Elevation data not found at {self.elevation_path}")
        
        # Get data bounds
        self.calculate_bounds()
    
    def calculate_bounds(self):
        """Calculate bounds from force positions."""
        # Get bounds from forces
        all_lons = np.concatenate([
            self.red_forces['longitude'].values, 
            self.blue_forces['longitude'].values
        ])
        all_lats = np.concatenate([
            self.red_forces['latitude'].values, 
            self.blue_forces['latitude'].values
        ])
        
        # Calculate bounds
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
        
        self.bounds = {
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_min': lat_min,
            'lat_max': lat_max
        }
    
    def create_elevation_colormap(self):
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
    
    def create_landuse_colormap(self):
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
    
    def sample_terrain_directly(self, resolution=100):
        """
        Sample terrain directly using helper functions.
        
        Args:
            resolution: Number of points to sample in each dimension
        
        Returns:
            Dictionary with terrain samples
        """
        print("Sampling terrain directly from raster files...")
        
        # Create coordinate grids
        lons = np.linspace(self.bounds['lon_min'], self.bounds['lon_max'], resolution)
        lats = np.linspace(self.bounds['lat_min'], self.bounds['lat_max'], resolution)
        
        # Create empty arrays
        elevation_grid = np.zeros((resolution, resolution))
        landuse_grid = np.zeros((resolution, resolution), dtype=np.int32)
        water_mask = np.zeros((resolution, resolution), dtype=bool)
        forest_mask = np.zeros((resolution, resolution), dtype=bool)
        
        # Sample at each point
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                try:
                    # Sample elevation
                    elevation_grid[i, j] = get_altitude(lat, lon)
                    
                    # Sample land use
                    try:
                        landuse_grid[i, j] = get_raster_value_at_coords(
                            lat, lon, self.landuse_tif
                        )
                    except:
                        landuse_grid[i, j] = 0
                    
                    # Check for water
                    water_mask[i, j] = is_water(lat, lon)
                    
                    # Check for forest
                    forest_mask[i, j] = is_forest(lat, lon)
                    
                except Exception as e:
                    print(f"Error sampling at {lat}, {lon}: {e}")
                    # Default values
                    elevation_grid[i, j] = 0
                    landuse_grid[i, j] = 0
                    water_mask[i, j] = False
                    forest_mask[i, j] = False
        
        return {
            'elevation': elevation_grid,
            'landuse': landuse_grid,
            'water_mask': water_mask,
            'forest_mask': forest_mask,
            'lons': lons,
            'lats': lats
        }
    
    def create_interactive_map(self, timestamp=None, output_dir="output/interactive_map"):
        """
        Create an interactive map with terrain features and forces.
        
        Args:
            timestamp: Optional timestamp to filter red forces
            output_dir: Directory to save output files
        
        Returns:
            Path to saved visualization
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample terrain directly if raster files are available
        if os.path.exists(self.landuse_tif) and os.path.exists(self.elevation_tif):
            # Try a lower resolution first because sampling can be slow
            terrain_samples = self.sample_terrain_directly(resolution=50)
            lats = terrain_samples['lats']
            lons = terrain_samples['lons']
            elevation_grid = terrain_samples['elevation']
            landuse_grid = terrain_samples['landuse']
            water_mask = terrain_samples['water_mask']
            forest_mask = terrain_samples['forest_mask']
        else:
            # Use pre-loaded data
            elevation_grid = self.elevation
            landuse_grid = None
            water_mask = None
            forest_mask = None
        
        # Filter red forces by timestamp if provided
        if timestamp is not None and 'datetime' in self.red_forces.columns:
            current_red = self.red_forces[self.red_forces['datetime'] == timestamp].copy()
            title_time = f" at {timestamp}"
        else:
            # Use latest timestamp if not specified
            if 'datetime' in self.red_forces.columns:
                timestamp = self.red_forces['datetime'].max()
                current_red = self.red_forces[self.red_forces['datetime'] == timestamp].copy()
                title_time = f" at {timestamp}"
            else:
                current_red = self.red_forces.copy()
                title_time = ""
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. Plot elevation with forces
        # -------------------------------------------------
        if elevation_grid is not None:
            # Create proper normalization
            elev_min = np.min(elevation_grid)
            elev_max = np.max(elevation_grid)
            
            # Make water areas blue
            if elev_min < 0:
                from matplotlib.colors import TwoSlopeNorm
                elev_norm = TwoSlopeNorm(vmin=elev_min, vcenter=0, vmax=elev_max)
            else:
                from matplotlib.colors import Normalize
                elev_norm = Normalize(vmin=elev_min, vmax=elev_max)
            
            # Plot elevation
            elev_im = ax1.imshow(
                elevation_grid,
                cmap=self.elevation_cmap,
                norm=elev_norm,
                extent=[self.bounds['lon_min'], self.bounds['lon_max'], 
                       self.bounds['lat_min'], self.bounds['lat_max']],
                aspect='auto',
                origin='lower'
            )
            
            # Add colorbar
            cbar1 = plt.colorbar(elev_im, ax=ax1, shrink=0.7)
            cbar1.set_label('Elevation (m)')
        else:
            ax1.text(0.5, 0.5, "Elevation Data Not Available", 
                    ha='center', va='center', fontsize=14)
        
        # Plot blue forces
        ax1.scatter(
            self.blue_forces['longitude'],
            self.blue_forces['latitude'],
            s=120,
            c='blue',
            marker='^',
            label='Blue Forces',
            edgecolor='black',
            zorder=10
        )
        
        # Plot red forces by category
        if 'target_class' in current_red.columns:
            for target_class, group in current_red.groupby('target_class'):
                color = self.target_colors.get(target_class, 'red')
                
                ax1.scatter(
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
                    ax1.annotate(
                        row['target_id'],
                        (row['longitude'], row['latitude']),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                    )
        else:
            # If no target_class, plot all as red
            ax1.scatter(
                current_red['longitude'],
                current_red['latitude'],
                s=80,
                c='red',
                marker='o',
                label='Red Forces',
                edgecolor='black',
                zorder=10
            )
        
        # Add grid and labels
        ax1.grid(alpha=0.3)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'Nova Scotia Battlefield - Elevation{title_time}', fontsize=14)
        ax1.legend(loc='upper right')
        
        # 2. Plot land use features with forces
        # -------------------------------------------------
        if landuse_grid is not None:
            # Plot land use
            landuse_im = ax2.imshow(
                landuse_grid,
                cmap=self.landuse_cmap,
                extent=[self.bounds['lon_min'], self.bounds['lon_max'], 
                       self.bounds['lat_min'], self.bounds['lat_max']],
                aspect='auto',
                origin='lower',
                vmin=0,
                vmax=20
            )
            
            # Create legend for land use
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
                color = self.landuse_cmap(value / 20)  # Normalize value to 0-1
                landuse_legend.append(Patch(facecolor=color, label=label))
            
            # Add land use legend
            ax2.legend(handles=landuse_legend, loc='upper right', fontsize=8)
        elif water_mask is not None and forest_mask is not None:
            # Create a simplified land use visualization from masks
            simplified = np.zeros_like(elevation_grid)
            
            # Fill water areas
            simplified[water_mask] = -50  # Low elevation to appear blue
            
            # Fill forest areas (only if not water)
            forest_mask = forest_mask & ~water_mask  # Remove overlap
            simplified[forest_mask] = 50  # Mid elevation to appear green
            
            # Plot simplified terrain
            landuse_im = ax2.imshow(
                simplified,
                cmap=self.elevation_cmap,
                extent=[self.bounds['lon_min'], self.bounds['lon_max'], 
                       self.bounds['lat_min'], self.bounds['lat_max']],
                aspect='auto',
                origin='lower'
            )
            
            # Create simplified legend
            landuse_legend = [
                Patch(facecolor='#0000FF', label="Water"),
                Patch(facecolor='#228B22', label="Forest"),
                Patch(facecolor='#F5DEB3', label="Land")
            ]
            
            # Add simplified legend
            ax2.legend(handles=landuse_legend, loc='upper right', fontsize=8)
        else:
            ax2.text(0.5, 0.5, "Land Use Data Not Available", 
                    ha='center', va='center', fontsize=14)
        
        # Plot blue forces on land use map
        ax2.scatter(
            self.blue_forces['longitude'],
            self.blue_forces['latitude'],
            s=120,
            c='blue',
            marker='^',
            label='Blue Forces',
            edgecolor='black',
            zorder=10
        )
        
        # Plot red forces by category on land use map
        if 'target_class' in current_red.columns:
            for target_class, group in current_red.groupby('target_class'):
                color = self.target_colors.get(target_class, 'red')
                
                ax2.scatter(
                    group['longitude'],
                    group['latitude'],
                    s=80,
                    c=color,
                    marker='o',
                    edgecolor='black',
                    zorder=10
                )
                
                # Add target ID labels
                for _, row in group.iterrows():
                    ax2.annotate(
                        row['target_id'],
                        (row['longitude'], row['latitude']),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                    )
        else:
            # If no target_class, plot all as red
            ax2.scatter(
                current_red['longitude'],
                current_red['latitude'],
                s=80,
                c='red',
                marker='o',
                edgecolor='black',
                zorder=10
            )
        
        # Add grid and labels for land use map
        ax2.grid(alpha=0.3)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(f'Nova Scotia Battlefield - Land Use{title_time}', fontsize=14)
        
        # Add info box
        info_text = (
            f"Red Forces: {len(current_red)}\n"
            f"Blue Forces: {len(self.blue_forces)}\n"
            f"Nova Scotia Battlefield"
        )
        if timestamp is not None:
            info_text = f"Time: {timestamp}\n" + info_text
            
        ax1.text(0.02, 0.02, info_text, transform=ax1.transAxes,
               bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp_str = ""
        if timestamp is not None:
            # Convert timestamp to string for filename
            if isinstance(timestamp, pd.Timestamp):
                timestamp_str = f"_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            else:
                timestamp_str = f"_{timestamp}"
        
        output_path = os.path.join(output_dir, f"terrain_forces{timestamp_str}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"Visualization saved to {output_path}")
        
        return output_path
    
    def create_trajectory_prediction_visualization(self, prediction_data=None, 
                                                  timestamp=None, 
                                                  output_dir="output/trajectory_prediction"):
        """
        Create visualization with predicted trajectories.
        
        Args:
            prediction_data: Dictionary with prediction results
            timestamp: Timestamp for the prediction
            output_dir: Output directory
        
        Returns:
            Path to saved visualization
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter red forces by timestamp if provided
        if timestamp is not None and 'datetime' in self.red_forces.columns:
            current_red = self.red_forces[self.red_forces['datetime'] == timestamp].copy()
            title_time = f" at {timestamp}"
        else:
            # Use latest timestamp if not specified
            if 'datetime' in self.red_forces.columns:
                timestamp = self.red_forces['datetime'].max()
                current_red = self.red_forces[self.red_forces['datetime'] == timestamp].copy()
                title_time = f" at {timestamp}"
            else:
                current_red = self.red_forces.copy()
                title_time = ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot elevation background
        if self.elevation is not None:
            # Create proper normalization
            elev_min = np.min(self.elevation)
            elev_max = np.max(self.elevation)
            
            # Make water areas blue
            if elev_min < 0:
                from matplotlib.colors import TwoSlopeNorm
                elev_norm = TwoSlopeNorm(vmin=elev_min, vcenter=0, vmax=elev_max)
            else:
                from matplotlib.colors import Normalize
                elev_norm = Normalize(vmin=elev_min, vmax=elev_max)
            
            # Plot elevation
            elev_im = ax.imshow(
                self.elevation,
                cmap=self.elevation_cmap,
                norm=elev_norm,
                extent=[self.bounds['lon_min'], self.bounds['lon_max'], 
                       self.bounds['lat_min'], self.bounds['lat_max']],
                aspect='auto',
                origin='lower'
            )
            
            # Add colorbar
            cbar = plt.colorbar(elev_im, ax=ax, shrink=0.7)
            cbar.set_label('Elevation (m)')
        
        # Plot blue forces
        ax.scatter(
            self.blue_forces['longitude'],
            self.blue_forces['latitude'],
            s=120,
            c='blue',
            marker='^',
            label='Blue Forces',
            edgecolor='black',
            zorder=10
        )
        
        # Plot current red positions
        if 'target_class' in current_red.columns:
            for target_class, group in current_red.groupby('target_class'):
                color = self.target_colors.get(target_class, 'red')
                
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
                current_red['longitude'],
                current_red['latitude'],
                s=80,
                c='red',
                marker='o',
                label='Red Forces',
                edgecolor='black',
                zorder=10
            )
        
        # Plot prediction trajectories if available
        if prediction_data is not None:
            for target_id, prediction in prediction_data.items():
                # Get target info
                target_info = current_red[current_red['target_id'] == target_id]
                
                if len(target_info) > 0:
                    # Get target class for color
                    if 'target_class' in target_info.columns:
                        target_class = target_info.iloc[0]['target_class']
                        color = self.target_colors.get(target_class, 'red')
                    else:
                        color = 'red'
                    
                    # Extract prediction components
                    mean_traj = prediction['mean']
                    lower_ci = prediction['lower_ci']
                    upper_ci = prediction['upper_ci']
                    
                    # Plot trajectory
                    ax.plot(
                        mean_traj[:, 0], mean_traj[:, 1],
                        '--',
                        color=color,
                        linewidth=2,
                        alpha=0.8,
                        zorder=5
                    )
                    
                    # Plot prediction confidence area
                    from matplotlib.patches import Ellipse
                    
                    for i in range(len(mean_traj)):
                        ellipse = Ellipse(
                            (mean_traj[i, 0], mean_traj[i, 1]),
                            width=upper_ci[i, 0] - lower_ci[i, 0],
                            height=upper_ci[i, 1] - lower_ci[i, 1],
                            color=color,
                            alpha=0.2,
                            zorder=3
                        )
                        ax.add_patch(ellipse)
                    
                    # Add final position marker
                    ax.scatter(
                        mean_traj[-1, 0], mean_traj[-1, 1],
                        color=color,
                        s=80,
                        marker='x',
                        zorder=8,
                        label=f"Predicted {target_id}" if i == 0 else "",
                        edgecolor='black'
                    )
        
        # Add grid and labels
        ax.grid(alpha=0.3)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Target Movement Prediction{title_time}', fontsize=14)
        ax.legend(loc='upper right')
        
        # Add info box
        info_text = (
            f"Red Forces: {len(current_red)}\n"
            f"Blue Forces: {len(self.blue_forces)}\n"
            f"Trajectory Prediction: {5} minutes ahead"
        )
        if timestamp is not None:
            info_text = f"Time: {timestamp}\n" + info_text
            
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
        
        # Save visualization
        timestamp_str = ""
        if timestamp is not None:
            # Convert timestamp to string for filename
            if isinstance(timestamp, pd.Timestamp):
                timestamp_str = f"_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            else:
                timestamp_str = f"_{timestamp}"
        
        output_path = os.path.join(output_dir, f"trajectory_prediction{timestamp_str}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"Prediction visualization saved to {output_path}")
        
        return output_path

def main():
    """Create terrain and force visualizations."""
    # Initialize visualizer
    visualizer = TerrainForceVisualizer(data_dir="data")
    
    # Create interactive terrain map
    visualizer.create_interactive_map(output_dir="output/terrain_analysis")
    
    # If you have prediction data, you could also create a prediction visualization
    # Example:
    # from target_movement_prediction import TargetMovementPredictor
    # predictor = TargetMovementPredictor()
    # predictor.load_model("models/target_predictor_model.pt")
    # 
    # target_data = visualizer.red_forces
    # timestamp = target_data['datetime'].max()
    # 
    # predictions = {}
    # for target_id in target_data['target_id'].unique():
    #     pred = predictor.predict_out_of_view(
    #         target_data, target_id, timestamp, 300
    #     )
    #     if pred is not None:
    #         predictions[target_id] = pred
    # 
    # visualizer.create_trajectory_prediction_visualization(
    #     prediction_data=predictions,
    #     timestamp=timestamp
    # )

if __name__ == "__main__":
    main()
