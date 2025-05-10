"""
Enhanced Target Visualization Module

This module provides streamlined functions for visualizing target movement predictions,
with proper terrain visualization and customizable styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, Patch
import matplotlib.dates as mdates
from datetime import timedelta, datetime
import os
from tqdm import tqdm
import torch


class TargetVisualizer:
    """
    A class for visualizing target predictions with terrain awareness.
    Provides standardized visualization functions with consistent styling.
    """
    
    def __init__(self, terrain_data=None, elevation_data=None):
        """
        Initialize the visualizer with terrain and elevation data.
        
        Args:
            terrain_data: Numpy array of terrain classification
            elevation_data: Numpy array of elevation values
        """
        self.terrain_data = terrain_data
        self.elevation_data = elevation_data
        
        # Standard color schemes
        self.terrain_colors = self._create_terrain_colormap()
        self.target_class_colors = {
            'tank': '#8B0000',             # Dark red
            'armoured personnel carrier': '#FF4500',  # Orange red
            'light vehicle': '#FFA07A',     # Light salmon
            'unknown': '#DC143C'            # Crimson
        }
        
        # Setup colormaps
        self.terrain_cmap = ListedColormap(self.terrain_colors)
        self.elevation_cmap = plt.cm.terrain
        
    def _create_terrain_colormap(self):
        """Create a standardized colormap for terrain visualization."""
        return [
            '#FFFFFF',  # 0: No data or out of bounds
            '#1A5BAB',  # 1: Broadleaf Evergreen Forest - dark blue-green
            '#358221',  # 2: Broadleaf Deciduous Forest - green
            '#2E8B57',  # 3: Needleleaf Evergreen Forest - sea green
            '#52A72D',  # 4: Needleleaf Deciduous Forest - light green
            '#76B349',  # 5: Mixed Forest - medium green
            '#90EE90',  # 6: Tree Open - light green
            '#D2B48C',  # 7: Shrub - tan
            '#9ACD32',  # 8: Herbaceous - yellow-green
            '#ADFF2F',  # 9: Herbaceous with Sparse Tree/Shrub - green-yellow
            '#F5DEB3',  # 10: Sparse vegetation - wheat
            '#FFD700',  # 11: Cropland - gold
            '#F4A460',  # 12: Paddy field - sandy brown
            '#DAA520',  # 13: Cropland / Other Vegetation Mosaic - goldenrod
            '#2F4F4F',  # 14: Mangrove - dark slate gray
            '#00FFFF',  # 15: Wetland - cyan
            '#A0522D',  # 16: Bare area, consolidated (gravel, rock) - sienna
            '#DEB887',  # 17: Bare area, unconsolidated (sand) - burlywood
            '#808080',  # 18: Urban - gray
            '#FFFFFF',  # 19: Snow / Ice - white
            '#0000FF',  # 20: Water bodies - blue
        ]
    
    def _get_target_color(self, target_class):
        """Get color for a target based on its class."""
        return self.target_class_colors.get(target_class.lower(), self.target_class_colors['unknown'])
    
    def _add_terrain_background(self, ax, lon_min, lon_max, lat_min, lat_max):
        """Add terrain background to the given axes."""
        if self.terrain_data is not None:
            # Ensure correct orientation with origin='upper'
            ax.imshow(
                self.terrain_data, 
                cmap=self.terrain_cmap, 
                extent=[lon_min, lon_max, lat_max, lat_min],
                aspect='auto', 
                origin='upper', 
                alpha=0.7, 
                zorder=0
            )
            
            # Add legend for terrain
            terrain_legend = [
                Patch(facecolor='#0000FF', label='Water'),
                Patch(facecolor='#808080', label='Urban'),
                Patch(facecolor='#FFD700', label='Agricultural'),
                Patch(facecolor='#358221', label='Forest'),
                Patch(facecolor='#90EE90', label='Grassland'),
                Patch(facecolor='#A0522D', label='Barren'),
                Patch(facecolor='#00FFFF', label='Wetland')
            ]
            
            # Create second legend for terrain
            legend2 = ax.legend(
                handles=terrain_legend, 
                loc='lower right',
                title='Terrain Type',
                framealpha=0.7
            )
            ax.add_artist(legend2)
            
        elif self.elevation_data is not None:
            # If we only have elevation data, show that
            ax.imshow(
                self.elevation_data, 
                cmap=self.elevation_cmap, 
                extent=[lon_min, lon_max, lat_max, lat_min],
                aspect='auto', 
                origin='upper', 
                alpha=0.6, 
                zorder=0
            )
    
    def _add_confidence_ellipses(self, ax, mean_positions, lower_ci, upper_ci, color, alpha=0.2, zorder=2):
        """Add confidence ellipses to the plot."""
        ellipses = []
        for i in range(len(mean_positions)):
            # Calculate width and height from confidence intervals
            width = upper_ci[i, 0] - lower_ci[i, 0]
            height = upper_ci[i, 1] - lower_ci[i, 1]
            
            ellipse = Ellipse(
                (mean_positions[i, 0], mean_positions[i, 1]),
                width=width,
                height=height,
                color=color, 
                alpha=alpha, 
                zorder=zorder
            )
            ax.add_patch(ellipse)
            ellipses.append(ellipse)
        
        return ellipses
    
    def _set_common_plot_elements(self, ax, title, show_grid=True):
        """Set common plotting elements like title, labels, grid."""
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        if show_grid:
            ax.grid(True, alpha=0.3, zorder=1)
    
    def visualize_single_target(self, target_data, target_id, timestamp, prediction_result, 
                               blue_force_data=None, show_confidence=True, output_path=None):
        """
        Visualize prediction for a single target.
        
        Args:
            target_data: DataFrame with target observations
            target_id: ID of target to visualize
            timestamp: Timestamp when prediction was made
            prediction_result: Dictionary with prediction results
            blue_force_data: Optional DataFrame with blue force positions
            show_confidence: Whether to show confidence ellipses
            output_path: Path to save the visualization
        
        Returns:
            matplotlib figure
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Get target history and metadata
        target_df = target_data[target_data['target_id'] == target_id].copy()
        if 'datetime' in target_df.columns and not pd.api.types.is_datetime64_any_dtype(target_df['datetime']):
            target_df['datetime'] = pd.to_datetime(target_df['datetime'])
        
        # Get target class
        target_class = target_df['target_class'].iloc[0] if 'target_class' in target_df.columns else 'Unknown'
        color = self._get_target_color(target_class)
        
        # Get history up to timestamp
        history = target_df[target_df['datetime'] <= timestamp]
        
        # Extract prediction components
        mean_traj = prediction_result['mean']
        lower_ci = prediction_result['lower_ci']
        upper_ci = prediction_result['upper_ci']
        time_points = prediction_result['time_points']
        
        # Calculate plot bounds with padding
        lon_points = np.concatenate([history['longitude'].values, mean_traj[:, 0]])
        lat_points = np.concatenate([history['latitude'].values, mean_traj[:, 1]])
        
        lon_min, lon_max = lon_points.min(), lon_points.max()
        lat_min, lat_max = lat_points.min(), lat_points.max()
        
        # Add padding
        padding = 0.05
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        lon_min -= lon_range * padding
        lon_max += lon_range * padding
        lat_min -= lat_range * padding
        lat_max += lat_range * padding
        
        # PLOT 1: Basic prediction view
        # -----------------------------------------
        # Plot history trajectory
        ax1.plot(history['longitude'], history['latitude'], 'b-', 
                linewidth=2, label='Past Trajectory')
        ax1.scatter(
            history['longitude'].iloc[-1], history['latitude'].iloc[-1], 
            c='blue', s=100, marker='o', label='Last Seen Position',
            zorder=10, edgecolor='black'
        )
        
        # Plot predicted trajectory
        ax1.plot(
            mean_traj[:, 0], mean_traj[:, 1], '-', 
            color=color, linewidth=2.5, 
            label=f'Predicted {target_class} Trajectory',
            zorder=5
        )
        ax1.scatter(
            mean_traj[-1, 0], mean_traj[-1, 1], 
            c=color, s=100, marker='x', label='Predicted Final Position',
            zorder=10, edgecolor='black'
        )
        
        # Add confidence ellipses
        if show_confidence:
            self._add_confidence_ellipses(
                ax1, mean_traj, lower_ci, upper_ci, color, alpha=0.2, zorder=3
            )
        
        # Add time labels to predicted points
        for i, (x, y, t) in enumerate(zip(mean_traj[:, 0], mean_traj[:, 1], time_points)):
            if i % 2 == 0 or i == len(mean_traj) - 1:
                ax1.annotate(
                    t.strftime('%H:%M:%S'), 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
                )
        
        # Check for actual future data to validate prediction
        future = target_df[(target_df['datetime'] > timestamp) & 
                           (target_df['datetime'] <= time_points[-1])]
        
        if len(future) > 0:
            ax1.plot(
                future['longitude'], future['latitude'], 'g-', 
                linewidth=2, label='Actual Future Trajectory'
            )
            ax1.scatter(
                future['longitude'].iloc[-1], future['latitude'].iloc[-1],
                c='green', s=100, marker='*', label='Actual Final Position',
                zorder=10, edgecolor='black'
            )
        
        # Set title and legend
        self._set_common_plot_elements(
            ax1, f'Target {target_id} ({target_class}) Prediction'
        )
        ax1.legend(loc='upper left')
        ax1.set_xlim(lon_min, lon_max)
        ax1.set_ylim(lat_min, lat_max)
        
        # PLOT 2: Terrain view with prediction
        # --------------------------------------------
        # Set the axis extent for the global view
        full_lon_min, full_lon_max = target_data['longitude'].min(), target_data['longitude'].max()
        full_lat_min, full_lat_max = target_data['latitude'].min(), target_data['latitude'].max()
        
        # Add padding to full extent
        full_lon_padding = (full_lon_max - full_lon_min) * 0.05
        full_lat_padding = (full_lat_max - full_lat_min) * 0.05
        full_lon_min -= full_lon_padding
        full_lon_max += full_lon_padding
        full_lat_min -= full_lat_padding
        full_lat_max += full_lat_padding
        
        # Add terrain background
        self._add_terrain_background(ax2, full_lon_min, full_lon_max, full_lat_min, full_lat_max)
        
        # Plot the trajectory on terrain
        ax2.plot(
            history['longitude'], history['latitude'], 
            'b-', linewidth=2, zorder=5
        )
        ax2.plot(
            mean_traj[:, 0], mean_traj[:, 1], 
            '--', color=color, linewidth=2, zorder=5
        )
        
        # Show the prediction area as a rectangle on the terrain map
        rect = plt.Rectangle(
            (lon_min, lat_min), lon_max-lon_min, lat_max-lat_min, 
            fill=False, edgecolor='black', linestyle='--', linewidth=2,
            zorder=6
        )
        ax2.add_patch(rect)
        
        # Add markers for start and end points
        ax2.scatter(
            history['longitude'].iloc[-1], history['latitude'].iloc[-1], 
            c='blue', s=100, marker='o', zorder=10
        )
        ax2.scatter(
            mean_traj[-1, 0], mean_traj[-1, 1], 
            c=color, s=100, marker='x', zorder=10
        )
        
        # Add blue forces if available
        if blue_force_data is not None:
            ax2.scatter(
                blue_force_data['longitude'], blue_force_data['latitude'],
                c='blue', s=100, marker='^', label='Blue Forces',
                zorder=10, edgecolor='black'
            )
        
        self._set_common_plot_elements(ax2, 'Terrain Analysis with Prediction Context')
        
        # Add legend for start/end points and blue forces
        pos_legend = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Last Seen Position'),
            plt.Line2D([0], [0], marker='x', color='w', markerfacecolor=color, 
                      markersize=10, label='Predicted Final Position')
        ]
        
        if blue_force_data is not None:
            pos_legend.append(
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                          markersize=10, label='Blue Forces')
            )
        
        ax2.legend(handles=pos_legend, loc='upper left')
        
        # Add information box
        prediction_duration = (time_points[-1] - timestamp).total_seconds()
        info_text = (
            f"Target: {target_id} ({target_class})\n"
            f"Last seen: {timestamp}\n"
            f"Prediction duration: {prediction_duration:.0f} seconds\n"
            f"95% confidence interval shown"
        )
        
        ax1.text(0.02, 0.02, info_text, transform=ax1.transAxes, 
               bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        return fig
    
    def visualize_all_targets(self, target_data, timestamp, prediction_results, 
                             blue_force_data=None, output_path=None):
        """
        Visualize predictions for all targets on a single map.
        
        Args:
            target_data: DataFrame with target observations
            timestamp: Timestamp when prediction was made
            prediction_results: Dictionary of prediction results, keyed by target_id
            blue_force_data: Optional DataFrame with blue force positions
            output_path: Path to save visualization
        
        Returns:
            matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Get coordinate bounds for the map
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
        
        # Add terrain background
        self._add_terrain_background(ax, lon_min, lon_max, lat_min, lat_max)
        
        # Plot blue force positions if available
        if blue_force_data is not None:
            ax.scatter(
                blue_force_data['longitude'], blue_force_data['latitude'],
                c='blue', s=120, marker='^', label='Blue Forces', 
                zorder=10, edgecolor='black'
            )
        
        # Add legend handles
        legend_handles = []
        
        if blue_force_data is not None:
            legend_handles.append(
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                          markersize=10, label='Blue Forces')
            )
        
        # Track which target classes we've already added to the legend
        added_target_types = set()
        
        # Counter for successful predictions
        successful_predictions = 0
        total_targets = len(prediction_results)
        
        # Process each target
        for target_id, prediction in prediction_results.items():
            # Filter data for this target
            target_df = target_data[target_data['target_id'] == target_id]
            
            # Get target class if available
            target_class = target_df['target_class'].iloc[0] if 'target_class' in target_df.columns else 'Unknown'
            color = self._get_target_color(target_class)
            
            # Filter history for this target up to the timestamp
            if 'datetime' in target_df.columns and not pd.api.types.is_datetime64_any_dtype(target_df['datetime']):
                target_df['datetime'] = pd.to_datetime(target_df['datetime'])
                
            history = target_df[target_df['datetime'] <= timestamp]
            
            # Plot history trajectory as a thin line
            ax.plot(
                history['longitude'], history['latitude'], '-', 
                color=color, alpha=0.3, linewidth=1, zorder=3
            )
            
            # Plot last known position
            ax.scatter(
                history['longitude'].iloc[-1], history['latitude'].iloc[-1], 
                color=color, s=50, marker='o', zorder=5
            )
            
            # Plot predicted trajectory
            mean_traj = prediction['mean']
            lower_ci = prediction['lower_ci']
            upper_ci = prediction['upper_ci']
            
            ax.plot(
                mean_traj[:, 0], mean_traj[:, 1], '--', 
                color=color, linewidth=2, alpha=0.8, zorder=4
            )
            
            # Plot final predicted position
            ax.scatter(
                mean_traj[-1, 0], mean_traj[-1, 1], 
                color=color, s=80, marker='x', zorder=5, edgecolor='black'
            )
            
            # Add target ID label at the end of prediction
            ax.annotate(
                f"ID: {target_id}", 
                (mean_traj[-1, 0], mean_traj[-1, 1]), 
                textcoords="offset points", 
                xytext=(5, 5), 
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
            )
            
            # Add confidence ellipse for the final prediction
            ellipse = Ellipse(
                (mean_traj[-1, 0], mean_traj[-1, 1]),
                width=upper_ci[-1, 0] - lower_ci[-1, 0],
                height=upper_ci[-1, 1] - lower_ci[-1, 1],
                color=color, alpha=0.2, zorder=2
            )
            ax.add_patch(ellipse)
            
            # Add to legend if not already added for this target type
            if target_class not in added_target_types:
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, lw=2, linestyle='--',
                              marker='x', markeredgecolor='black',
                              label=f'{target_class} Target')
                )
                added_target_types.add(target_class)
            
            successful_predictions += 1
        
        # Set title and labels
        title = f'All Target Predictions at {timestamp}\nPrediction Duration: {prediction_results[list(prediction_results.keys())[0]]["time_points"][-1] - timestamp} seconds'
        self._set_common_plot_elements(ax, title)
        
        # Set axis limits
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        
        # Add legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
        
        # Add info text
        info_text = (
            f"Timestamp: {timestamp}\n"
            f"Targets with predictions: {successful_predictions} of {total_targets}\n"
            f"95% confidence ellipses shown"
        )
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
               bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
        
        # Save figure if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        return fig
    
    def create_animation(self, predictor, target_data, blue_force_data=None, 
                        output_filename='target_prediction_animation.mp4', frame_interval=10):
        """
        Create an animation of target predictions over time.
        
        Args:
            predictor: Trained TargetMovementPredictor model
            target_data: DataFrame with target observations
            blue_force_data: Optional DataFrame with blue force positions
            output_filename: Path to save the animation
            frame_interval: Interval between frames in the animation
            
        Returns:
            Animation object
        """
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        
        # Ensure datetime is properly parsed
        if 'datetime' in target_data.columns and not pd.api.types.is_datetime64_any_dtype(target_data['datetime']):
            target_data['datetime'] = pd.to_datetime(target_data['datetime'])
        
        # Get all unique timestamps sorted
        all_timestamps = sorted(pd.unique(target_data['datetime']))
        
        # Only use a subset of frames to keep the animation manageable
        frame_skip = max(1, len(all_timestamps) // 300)
        selected_frames = all_timestamps[::frame_skip]
        
        print(f"Creating animation with {len(selected_frames)} frames...")
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get coordinate bounds for the map
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
        
        # Add terrain background
        self._add_terrain_background(ax, lon_min, lon_max, lat_min, lat_max)
        
        # Create empty collections for visualization
        blue_scatter = ax.scatter([], [], c='blue', s=120, marker='^', 
                                 label='Blue Forces', zorder=10, edgecolor='black')
        red_scatter = ax.scatter([], [], c='red', s=80, marker='o', 
                                label='Red Forces', zorder=10, edgecolor='black')
        
        # Dictionaries to store lines and patches
        prediction_lines = {}
        confidence_ellipses = {}
        target_trails = {}
        
        # Time display
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                          bbox=dict(facecolor='white', alpha=0.8), zorder=20)
        
        # Add title
        ax.set_title('Nova Scotia Battlefield Visualization', fontsize=16)
        
        # Set axis limits
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, zorder=1)
        
        # Progress bar
        progress_bar = tqdm(total=len(selected_frames), desc="Generating frames")
        
        def init():
            blue_scatter.set_offsets(np.empty((0, 2)))
            red_scatter.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return [blue_scatter, red_scatter, time_text]
        
        def update(frame_idx):
            # Get current timestamp
            current_time = selected_frames[frame_idx]
            
            # Filter data for this timestamp
            current_red = target_data[target_data['datetime'] == current_time]
            
            # Update blue & red positions
            if blue_force_data is not None:
                blue_scatter.set_offsets(blue_force_data[['longitude', 'latitude']].values)
            red_scatter.set_offsets(current_red[['longitude', 'latitude']].values)
            
            # Update time display
            time_text.set_text(f'Time: {pd.Timestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")}')
            
            # Clean up previous predictions
            for target_id in list(prediction_lines.keys()):
                if prediction_lines[target_id] in ax.lines:
                    prediction_lines[target_id].remove()
                del prediction_lines[target_id]
            
            for target_id in list(confidence_ellipses.keys()):
                for ellipse in confidence_ellipses[target_id]:
                    if ellipse in ax.patches:
                        ellipse.remove()
                confidence_ellipses[target_id] = []
            
            # Update target trails
            for target_id in list(target_trails.keys()):
                if target_trails[target_id] in ax.lines:
                    target_trails[target_id].remove()
                del target_trails[target_id]
            
            # Make predictions for each target and color by target class
            for target_id, group in current_red.groupby('target_id'):
                target_class = group['target_class'].iloc[0] if 'target_class' in group.columns else 'Unknown'
                
                # Choose color based on target class
                color = self._get_target_color(target_class)
                
                # Get history data for this target
                target_history = target_data[(target_data['target_id'] == target_id) & 
                                          (target_data['datetime'] <= current_time)].sort_values('datetime')
                
                # Update target trail
                target_points = target_history[['longitude', 'latitude']].values[-10:]
                if len(target_points) >= 2:
                    line, = ax.plot(target_points[:, 0], target_points[:, 1], '-', 
                                   color=color, alpha=0.4, linewidth=1.5, zorder=5)
                    target_trails[target_id] = line
                
                # Generate prediction if enough history
                if len(target_history) >= predictor.config['sequence_length']:
                    try:
                        # Make prediction
                        prediction = predictor.predict_out_of_view(
                            target_data, 
                            target_id, 
                            current_time,
                            300  # 5-minute prediction
                        )
                        
                        if prediction is not None:
                            # Plot predicted trajectory
                            mean_traj = prediction['mean']
                            lower_ci = prediction['lower_ci']
                            upper_ci = prediction['upper_ci']
                            
                            line, = ax.plot(mean_traj[:, 0], mean_traj[:, 1], '--', 
                                           color=color, linewidth=2, alpha=0.8, zorder=6)
                            prediction_lines[target_id] = line
                            
                            # Add confidence ellipses
                            ellipses = []
                            for i in range(len(mean_traj)):
                                ellipse = Ellipse(
                                    (mean_traj[i, 0], mean_traj[i, 1]),
                                    width=upper_ci[i, 0] - lower_ci[i, 0],
                                    height=upper_ci[i, 1] - lower_ci[i, 1],
                                    color=color, alpha=0.2, zorder=4
                                )
                                ax.add_patch(ellipse)
                                ellipses.append(ellipse)
                            
                            confidence_ellipses[target_id] = ellipses
                    except Exception as e:
                        print(f"Error predicting for target {target_id}: {e}")
            
            # Update progress bar
            progress_bar.update(1)
            
            # Return updated artists
            artists = [blue_scatter, red_scatter, time_text]
            artists.extend(list(prediction_lines.values()))
            artists.extend([e for ellipses in confidence_ellipses.values() for e in ellipses])
            artists.extend(list(target_trails.values()))
            return artists
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=len(selected_frames),
                          init_func=init, blit=True, interval=frame_interval)
        
        # Save animation
        print("\nSaving animation...")
        writer = FFMpegWriter(fps=10, metadata=dict(artist='Target Prediction'), bitrate=3600)
        
        with tqdm(total=100, desc="Encoding video") as pbar:
            ani.save(output_filename, writer=writer, dpi=150,
                   progress_callback=lambda i, n: pbar.update(100/n))
        
        progress_bar.close()
        print(f"Animation saved to {output_filename}")
        return ani
    
    def plot_training_history(self, history, output_path=None):
        """Create a standardized plot of training history metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        
        if 'best_epoch' in history:
            ax1.axvline(x=history['best_epoch'] + 1, color='g', linestyle='--', label='Best Model')
            
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot NLL losses if available
        if 'train_nll_loss' in history and 'val_nll_loss' in history:
            ax2.plot(epochs, history['train_nll_loss'], 'b-', label='Training NLL Loss')
            ax2.plot(epochs, history['val_nll_loss'], 'r-', label='Validation NLL Loss')
            
            if 'best_epoch' in history:
                ax2.axvline(x=history['best_epoch'] + 1, color='g', linestyle='--', label='Best Model')
                
            ax2.set_title('Training and Validation NLL Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('NLL Loss')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {output_path}")
        
        return fig

    def visualize_predictions_on_terrain(self, target_data, predictions_by_target_id, timestamp, 
                                        blue_force_data=None, output_path=None):
        """
        Create a high-quality terrain visualization with all target predictions.
        
        Args:
            target_data: DataFrame with target observations
            predictions_by_target_id: Dictionary of prediction results by target_id
            timestamp: Time point for visualization
            blue_force_data: Optional DataFrame with blue force positions
            output_path: Path to save visualization
            
        Returns:
            matplotlib figure
        """
        # Create figure with a single large plot
        fig, ax = plt.subplots(figsize=(16, 12))
        
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
        
        # Add high-quality terrain background with proper colormapping
        self._add_terrain_background(ax, lon_min, lon_max, lat_min, lat_max)
        
        # Add blue force positions
        if blue_force_data is not None:
            ax.scatter(
                blue_force_data['longitude'], blue_force_data['latitude'],
                c='blue', s=150, marker='^', label='Blue Forces',
                zorder=10, edgecolor='black', alpha=0.9
            )
        
        # Process each target
        for target_id, prediction in predictions_by_target_id.items():
            # Get target data
            target_df = target_data[target_data['target_id'] == target_id]
            
            # Skip if no data before timestamp
            if 'datetime' in target_df.columns and not pd.api.types.is_datetime64_any_dtype(target_df['datetime']):
                target_df['datetime'] = pd.to_datetime(target_df['datetime'])
                
            if target_df[target_df['datetime'] <= timestamp].empty:
                continue
                
            # Get target class
            target_class = target_df['target_class'].iloc[0] if 'target_class' in target_df.columns else 'Unknown'
            color = self._get_target_color(target_class)
            
            # Get history
            history = target_df[target_df['datetime'] <= timestamp]
            
            # Plot history
            ax.plot(
                history['longitude'], history['latitude'], '-',
                color=color, alpha=0.5, linewidth=1.5, zorder=4
            )
            
            # Plot last position
            ax.scatter(
                history['longitude'].iloc[-1], history['latitude'].iloc[-1],
                color=color, s=80, marker='o', zorder=6, edgecolor='black'
            )
            
            # Extract prediction components
            mean_traj = prediction['mean']
            lower_ci = prediction['lower_ci']
            upper_ci = prediction['upper_ci']
            
            # Plot prediction
            ax.plot(
                mean_traj[:, 0], mean_traj[:, 1], '--',
                color=color, linewidth=2.5, alpha=0.8, zorder=5
            )
            
            # Add final position marker
            ax.scatter(
                mean_traj[-1, 0], mean_traj[-1, 1],
                color=color, s=100, marker='x', zorder=7, 
                edgecolor='black', linewidth=1.5
            )
            
            # Add confidence ellipse only for final position for clarity
            ellipse = Ellipse(
                (mean_traj[-1, 0], mean_traj[-1, 1]),
                width=upper_ci[-1, 0] - lower_ci[-1, 0],
                height=upper_ci[-1, 1] - lower_ci[-1, 1],
                color=color, alpha=0.3, zorder=3
            )
            ax.add_patch(ellipse)
            
            # Add target ID and class label
            ax.annotate(
                f"{target_id}: {target_class}",
                (mean_traj[-1, 0], mean_traj[-1, 1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        # Add legend by target class
        legend_elements = []
        
        if blue_force_data is not None:
            legend_elements.append(
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
                          markersize=12, label='Blue Forces')
            )
        
        # Add target type legend entries
        for target_class, color in self.target_class_colors.items():
            if target_class != 'unknown' and any(
                target_data['target_class'].str.lower() == target_class 
                if 'target_class' in target_data.columns else False
            ):
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                              markersize=10, label=f'{target_class.title()}')
                )
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Set title and labels
        title = f'Nova Scotia Battlefield - Target Predictions at {timestamp}'
        self._set_common_plot_elements(ax, title)
        
        # Add timestamp and info text
        info_text = (
            f"Time: {timestamp}\n"
            f"Prediction Horizon: 5 minutes\n"
            f"Battlefield Visualization"
        )
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.7), fontsize=12)
        
        # Save figure if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        return fig


# Example usage
def visualize_target_movement(predictor, target_data_path, blue_force_path, 
                             terrain_path=None, elevation_path=None):
    """
    Create comprehensive visualizations for target movement predictions.
    
    Args:
        predictor: Trained TargetMovementPredictor model
        target_data_path: Path to target data CSV
        blue_force_path: Path to blue force data CSV
        terrain_path: Path to terrain data NPY file (optional)
        elevation_path: Path to elevation data NPY file (optional)
        
    Returns:
        Dictionary of generated figures
    """
    import pandas as pd
    import numpy as np
    import os
    from datetime import timedelta
    
    # Load data
    target_data = pd.read_csv(target_data_path)
    blue_force_data = pd.read_csv(blue_force_path)
    
    # Ensure datetime is properly parsed
    if 'datetime' in target_data.columns:
        target_data['datetime'] = pd.to_datetime(target_data['datetime'])
    
    # Load terrain data if available
    terrain_data = None
    elevation_data = None
    
    if terrain_path and os.path.exists(terrain_path):
        terrain_data = np.load(terrain_path)
    
    if elevation_path and os.path.exists(elevation_path):
        elevation_data = np.load(elevation_path)
    
    # Create visualizer
    visualizer = TargetVisualizer(terrain_data, elevation_data)
    
    # Select a timestamp for predictions
    timestamps = sorted(target_data['datetime'].unique())
    selected_time = timestamps[len(timestamps) // 3]  # Use timestamp from first third
    print(f"Selected timestamp: {selected_time}")
    
    # Get all target IDs
    target_ids = target_data['target_id'].unique()
    
    # Make predictions for all targets
    predictions = {}
    
    for target_id in target_ids:
        # Get history for this target
        target_df = target_data[target_data['target_id'] == target_id]
        
        # Skip if no data before selected_time
        if target_df[target_df['datetime'] <= selected_time].empty:
            continue
        
        # Generate prediction
        try:
            prediction = predictor.predict_out_of_view(
                target_data, 
                target_id, 
                selected_time,
                300  # 5-minute prediction
            )
            
            if prediction is not None:
                predictions[target_id] = prediction
        except Exception as e:
            print(f"Error predicting for target {target_id}: {e}")
    
    # Create visualizations
    figures = {}
    
    # 1. Create visualization for a single target
    if predictions:
        # Select first target with valid prediction
        target_id = list(predictions.keys())[0]
        
        figures['single_target'] = visualizer.visualize_single_target(
            target_data, 
            target_id, 
            selected_time, 
            predictions[target_id],
            blue_force_data=blue_force_data,
            output_path="output/single_target_prediction.png"
        )
    
    # 2. Create visualization for all targets
    figures['all_targets'] = visualizer.visualize_all_targets(
        target_data,
        selected_time,
        predictions,
        blue_force_data=blue_force_data,
        output_path="output/all_targets_prediction.png"
    )
    
    # 3. Create high-quality terrain visualization
    figures['terrain'] = visualizer.visualize_predictions_on_terrain(
        target_data,
        predictions,
        selected_time,
        blue_force_data=blue_force_data,
        output_path="output/terrain_visualization.png"
    )
    
    # 4. Optionally create animation
    if len(predictions) > 0:
        try:
            figures['animation'] = visualizer.create_animation(
                predictor,
                target_data,
                blue_force_data,
                output_filename="output/target_prediction_animation.mp4"
            )
        except Exception as e:
            print(f"Error creating animation: {e}")
    
    return figures


# Run visualization if executed directly
if __name__ == "__main__":
    import argparse
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Target Movement Visualization')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data files')
    parser.add_argument('--terrain_path', type=str, default='adapted_data/terrain_map.npy',
                        help='Path to terrain data')
    parser.add_argument('--elevation_path', type=str, default='adapted_data/elevation_map.npy',
                        help='Path to elevation data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--target_id', type=str, default=None,
                        help='Target ID for single-target visualization (optional)')
    parser.add_argument('--create_animation', action='store_true',
                        help='Create animation of predictions')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Import target movement predictor
    from target_movement_prediction import TargetMovementPredictor
    
    # Load model
    predictor = TargetMovementPredictor()
    if not predictor.load_model(args.model_path):
        print(f"Error: Could not load model from {args.model_path}")
        exit(1)
    
    # Generate visualizations
    visualize_target_movement(
        predictor,
        os.path.join(args.data_dir, "red_sightings.csv"),
        os.path.join(args.data_dir, "blue_locations.csv"),
        args.terrain_path,
        args.elevation_path
    )
