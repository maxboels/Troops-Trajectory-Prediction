# target_movement_animation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap, Normalize
from tqdm import tqdm

def create_target_movement_animation(target_data_path, blue_force_path,
                                     terrain_path=None, elevation_path=None,
                                     output_file="target_movement_animation.mp4",
                                     fps=10, dpi=150, duration_seconds=30):
    """
    Create an animation of target movements over time.

    Args:
        target_data_path: Path to target data CSV
        blue_force_path: Path to blue force data CSV
        terrain_path: Path to terrain data NPY file (optional)
        elevation_path: Path to elevation data NPY file (optional)
        output_file: Path to save the animation
        fps: Frames per second
        dpi: Resolution of the animation
        duration_seconds: Duration of the animation in seconds
    """
    # Load data
    target_data = pd.read_csv(target_data_path)
    blue_force_data = pd.read_csv(blue_force_path)

    # Ensure datetime is properly parsed
    if 'datetime' in target_data.columns:
        target_data['datetime'] = pd.to_datetime(target_data['datetime'])

    # Load terrain and elevation data if available
    terrain_data_array = None
    elevation_data_array = None

    # Define geographical bounds for the map data if available
    # These should correspond to the actual bounds of your .npy files
    # For Nova Scotia, as an example:
    map_actual_lon_min, map_actual_lon_max = -66.5, -59.5 # Example, adjust to your data
    map_actual_lat_min, map_actual_lat_max = 43.0, 47.5  # Example, adjust to your data

    if terrain_path and os.path.exists(terrain_path):
        terrain_data_array = np.load(terrain_path)
        print(f"Loaded terrain data with shape {terrain_data_array.shape}")
        print(f"Terrain data min: {terrain_data_array.min()}, max: {terrain_data_array.max()}")
        # Assuming terrain_data_array.shape[0] is latitude dimension, terrain_data_array.shape[1] is longitude

    if elevation_path and os.path.exists(elevation_path):
        elevation_data_array = np.load(elevation_path)
        print(f"Loaded elevation data with shape {elevation_data_array.shape}")
        print(f"Elevation data min: {elevation_data_array.min()}, max: {elevation_data_array.max()}")

    # Get all unique timestamps from target data
    timestamps = sorted(pd.unique(target_data['datetime']))
    if not timestamps:
        print("No timestamps found in target data. Cannot create animation.")
        return

    # Determine how many timestamps to use based on desired duration
    total_frames = fps * duration_seconds
    if len(timestamps) > total_frames :
        frame_skip = max(1, len(timestamps) // total_frames)
        selected_timestamps = timestamps[::frame_skip]
    else: # If fewer timestamps than total_frames, use all of them and adjust fps or duration notionally
        selected_timestamps = timestamps
        # Consider adjusting fps if len(selected_timestamps) is very small
        # fps = max(1, len(selected_timestamps) // duration_seconds) if duration_seconds > 0 else 1


    print(f"Creating animation with {len(selected_timestamps)} frames...")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Determine coordinate bounds for the plot from target data
    plot_lon_min, plot_lon_max = target_data['longitude'].min(), target_data['longitude'].max()
    plot_lat_min, plot_lat_max = target_data['latitude'].min(), target_data['latitude'].max()

    # Add padding to plot bounds
    padding_factor = 0.05
    lon_range = plot_lon_max - plot_lon_min
    lat_range = plot_lat_max - plot_lat_min
    
    # Ensure range is not zero to avoid issues with padding
    if lon_range == 0: lon_range = 1.0 
    if lat_range == 0: lat_range = 1.0

    plot_lon_min -= lon_range * padding_factor
    plot_lon_max += lon_range * padding_factor
    plot_lat_min -= lat_range * padding_factor
    plot_lat_max += lat_range * padding_factor

    # Define terrain colormap
    # (Using a simplified example, ensure your actual land_use_colors match your data's class indices)
    land_use_colors_map = {
        0: '#ADD8E6',  # Water (example, adjust)
        1: '#228B22',  # Forest (example, adjust)
        2: '#808080',  # Urban (example, adjust)
        3: '#9ACD32',  # Herbaceous (example, adjust)
        4: '#FFD700',  # Cropland (example, adjust)
        # ... add all your classes ...
    }
    # Create a colormap from the highest index down to 0
    max_terrain_val = int(terrain_data_array.max()) if terrain_data_array is not None else 0
    cmap_colors = [land_use_colors_map.get(i, '#FFFFFF') for i in range(max_terrain_val + 1)]
    terrain_cmap = ListedColormap(cmap_colors)


    # Use the actual extent of your map data for imshow
    # If your .npy files cover a specific region, use those bounds.
    # For this example, we'll use the padded plot bounds if map specific bounds aren't set.
    # It's better to define these based on your map_*.npy files' actual geographic coverage.
    # For demonstration, using plot_lon_min/max, but ideally these should be the actual geo-reference of the NPYs.
    # If your NPY covers the entire target_data area, then plot_lon_min etc. might be okay.
    # However, the script used hardcoded values for Nova Scotia in the previous version.
    # Let's assume the NPY files are georeferenced to these extents:
    # These should be the true extents of your terrain_map.npy and elevation_map.npy
    # If they are derived from the target_data extents, then plot_lon_min etc. is fine.
    # For now, using the calculated plot extents for the imshow
    imshow_extent = [plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max]


    # Plot terrain data if available
    if terrain_data_array is not None:
        ax.imshow(terrain_data_array, cmap=terrain_cmap, alpha=0.7,
                  extent=imshow_extent, # Use the correct geographical extent of the terrain_data_array
                  aspect='auto', origin='upper', zorder=0) # CHANGED to origin='upper'

        # Add elevation overlay if available
        if elevation_data_array is not None:
            elev_min_val = np.min(elevation_data_array)
            elev_max_val = np.max(elevation_data_array)
            elev_norm = Normalize(vmin=elev_min_val, vmax=elev_max_val)

            ax.imshow(elevation_data_array, cmap='terrain', norm=elev_norm, alpha=0.3,
                      extent=imshow_extent, # Use the correct geographical extent
                      aspect='auto', origin='upper', zorder=1) # CHANGED to origin='upper'


    # Define colors for target classes
    target_colors = {
        'tank': 'darkred',
        'armoured personnel carrier': 'orangered',
        'light vehicle': 'coral',
        'unknown': 'red' # Default for any other class
    }

    # Set up plot elements
    ax.set_xlim(plot_lon_min, plot_lon_max)
    ax.set_ylim(plot_lat_min, plot_lat_max) # lat_min at bottom, lat_max at top
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Target Movement Animation')
    ax.grid(True, alpha=0.3)

    # Add terrain legend (simplified, ensure colors match your cmap_colors and terrain classes)
    terrain_legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=land_use_colors_map.get(0, '#ADD8E6'), markersize=10, label='Water'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=land_use_colors_map.get(2, '#808080'), markersize=10, label='Urban'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=land_use_colors_map.get(1, '#228B22'), markersize=10, label='Forest'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=land_use_colors_map.get(3, '#9ACD32'), markersize=10, label='Herbaceous'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=land_use_colors_map.get(4, '#FFD700'), markersize=10, label='Cropland')
    ]

    # Target legend
    target_legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=target_colors['tank'], markersize=10, label='Tank'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=target_colors['armoured personnel carrier'], markersize=10, label='APC'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=target_colors['light vehicle'], markersize=10, label='Light Vehicle'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Blue Forces')
    ]

    # Add legend
    first_legend = ax.legend(handles=terrain_legend_handles, loc='upper left', title='Terrain', fontsize='small')
    ax.add_artist(first_legend)
    ax.legend(handles=target_legend_handles, loc='upper right', title='Forces', fontsize='small')

    # Initialize scatter plots for blue and red forces
    blue_scatter = ax.scatter([], [], c='blue', s=120, marker='^',
                              label='Blue Forces', zorder=10, edgecolor='black')

    # Create a scatter plot for each target class
    target_scatters = {}
    for target_class_name, color_val in target_colors.items():
        target_scatters[target_class_name] = ax.scatter([], [], c=color_val, s=80, marker='o',
                                                      zorder=8, edgecolor='black')

    # Track lines for each target's trail
    target_lines_dict = {}

    # Text for timestamp display
    time_text_obj = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           bbox=dict(facecolor='white', alpha=0.7), zorder=20)

    # Initialize function for animation
    def init():
        blue_scatter.set_offsets(np.empty((0, 2)))
        for scatter_obj in target_scatters.values():
            scatter_obj.set_offsets(np.empty((0, 2)))
        for line_obj in target_lines_dict.values(): # Clear existing lines if any
            line_obj.remove()
        target_lines_dict.clear()
        time_text_obj.set_text('')
        return [blue_scatter] + list(target_scatters.values()) + [time_text_obj] # Initial artists

    # Update function for animation
    def update(frame_idx):
        timestamp = selected_timestamps[frame_idx]
        current_frame_data = target_data[target_data['datetime'] <= timestamp]

        # Get the most recent positions for each target
        # FIXED to avoid DeprecationWarning and be more efficient
        if not current_frame_data.empty:
            idx = current_frame_data.groupby('target_id')['datetime'].idxmax()
            latest_positions_df = current_frame_data.loc[idx]
        else:
            latest_positions_df = pd.DataFrame(columns=target_data.columns)


        # Update blue forces (assuming they are static or pre-filtered for the animation)
        if blue_force_data is not None and not blue_force_data.empty:
            blue_scatter.set_offsets(blue_force_data[['longitude', 'latitude']].values)

        # Update target positions by class
        for target_class_name, scatter_obj in target_scatters.items():
            if 'target_class' in latest_positions_df.columns:
                class_targets_df = latest_positions_df[
                    latest_positions_df['target_class'].astype(str).str.lower() == target_class_name
                ]
            elif target_class_name == 'unknown': # If no class info, treat all as unknown
                 class_targets_df = latest_positions_df
            else:
                class_targets_df = pd.DataFrame()

            if not class_targets_df.empty:
                scatter_obj.set_offsets(class_targets_df[['longitude', 'latitude']].values)
            else:
                scatter_obj.set_offsets(np.empty((0, 2)))

        # Update trail lines
        # Remove old lines
        for target_id_key in list(target_lines_dict.keys()): # Iterate over a copy of keys
            line = target_lines_dict.pop(target_id_key)
            line.remove()

        # Add new trails for each target
        if not current_frame_data.empty:
            for target_id_val, group_df in current_frame_data.groupby('target_id'):
                if len(group_df) >= 2:
                    target_class_val = group_df['target_class'].iloc[-1].lower() if 'target_class' in group_df.columns else 'unknown'
                    color_val = target_colors.get(target_class_val, target_colors['unknown'])
                    group_df = group_df.sort_values('datetime')
                    trail_points_arr = group_df.tail(10)[['longitude', 'latitude']].values
                    line, = ax.plot(trail_points_arr[:, 0], trail_points_arr[:, 1], '-',
                                    color=color_val, alpha=0.6, linewidth=1.5, zorder=5)
                    target_lines_dict[target_id_val] = line

        time_text_obj.set_text(f'Time: {pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")}')

        artists = [blue_scatter] + list(target_scatters.values()) + list(target_lines_dict.values()) + [time_text_obj]
        return artists

    # Create animation
    animation = FuncAnimation(
        fig, update, frames=len(selected_timestamps),
        init_func=init, blit=True, interval=1000 / fps if fps > 0 else 200
    )

    # Save animation
    print("Saving animation...")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    writer = FFMpegWriter(fps=fps if fps > 0 else 5, metadata=dict(artist='Target Movement Animation'), bitrate=3600)

    with tqdm(total=len(selected_timestamps), desc="Encoding video") as pbar:
        animation.save(output_file, writer=writer, dpi=dpi,
                       progress_callback=lambda i, n: pbar.update(1))

    print(f"Animation saved to {output_file}")
    plt.close(fig) # Close the figure to free memory
    return animation

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create target movement animation')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    # Adjusted default paths to be more generic, assuming they are in data_dir or a subdir like 'adapted_data'
    parser.add_argument('--terrain_path', type=str, default='data/adapted_data/terrain_map.npy', help='Path to terrain data')
    parser.add_argument('--elevation_path', type=str, default='data/adapted_data/elevation_map.npy', help='Path to elevation data')
    parser.add_argument('--output_file', type=str, default='visualizations/target_movement_animation.mp4', help='Output video file path')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--duration', type=int, default=30, help='Duration of animation in seconds')
    parser.add_argument('--dpi', type=int, default=150, help='Resolution of the animation')

    args = parser.parse_args()

    # Construct full paths for data files if they are relative to data_dir
    target_data_full_path = os.path.join(args.data_dir, "red_sightings.csv")
    blue_force_full_path = os.path.join(args.data_dir, "blue_locations.csv")
    
    # For terrain and elevation, it might be an absolute path or relative to project root/data_dir
    # The default argparse paths are 'adapted_data/terrain_map.npy', etc.
    # If these are meant to be inside args.data_dir, they should be joined.
    # For this example, assuming args.terrain_path and args.elevation_path are correct as provided.

    create_target_movement_animation(
        target_data_full_path,
        blue_force_full_path,
        args.terrain_path,
        args.elevation_path,
        args.output_file,
        fps=args.fps,
        duration_seconds=args.duration,
        dpi=args.dpi
    )