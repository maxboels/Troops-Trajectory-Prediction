"""
Optimized animation script for target trajectories with terrain background and progress bars
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from matplotlib.colors import ListedColormap

# Output settings
OUTPUT_DIR = "adapted_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "battlefield_animation.mp4")

def create_animation():
    # Load data
    print("Loading data...")
    blue_df = pd.read_csv(os.path.join(OUTPUT_DIR, "blue_force_observations.csv"))
    red_df = pd.read_csv(os.path.join(OUTPUT_DIR, "target_observations.csv"))
    
    # Convert timestamps to datetime
    print("Processing timestamps...")
    blue_df['timestamp'] = pd.to_datetime(blue_df['timestamp'])
    red_df['timestamp'] = pd.to_datetime(red_df['timestamp'])
    
    # Load terrain and elevation data
    try:
        print("Loading terrain data...")
        terrain_map = np.load(os.path.join(OUTPUT_DIR, "terrain_map.npy"))
        elevation_map = np.load(os.path.join(OUTPUT_DIR, "elevation_map.npy"))
        print(f"Loaded terrain map with shape {terrain_map.shape}")
        print(f"Loaded elevation map with shape {elevation_map.shape}")
        
        # Create terrain colormap
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
        terrain_cmap = ListedColormap(terrain_colors)
        has_terrain = True
    except Exception as e:
        print(f"Error loading terrain data: {e}")
        has_terrain = False
    
    # Get all unique timestamps sorted
    print("Processing unique timestamps...")
    all_timestamps = sorted(pd.unique(red_df['timestamp']))
    
    # Set up the figure with a larger size
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
    
    # Plot terrain background if available
    if has_terrain:
        print("Plotting terrain background...")
        # Since we're using lat/lon directly as x, y coordinates, we need to create a 
        # simple background. The exact mapping isn't critical for visualization
        terrain_img = ax.imshow(terrain_map, cmap=terrain_cmap, vmin=0, vmax=7, 
                              alpha=0.5, extent=[min_x, max_x, min_y, max_y], 
                              zorder=0, aspect='auto')
    
    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Create scatter plots with larger markers for better visibility
    blue_scatter = ax.scatter([], [], c='blue', s=120, marker='^', label='Blue Forces', zorder=3, edgecolor='black')
    red_scatter = ax.scatter([], [], c='red', s=100, marker='o', label='Red Forces', zorder=3, edgecolor='black')
    
    # Create time text display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                        bbox=dict(facecolor='white', alpha=0.8), zorder=5)
    
    # Pre-process target data to improve performance
    print("Pre-processing target data...")
    target_data_by_id = {}
    for target_id in tqdm(red_df['id'].unique(), desc="Processing targets"):
        target_data_by_id[target_id] = red_df[red_df['id'] == target_id].sort_values('timestamp')
    
    # Pre-compute timestamp mappings for faster lookup
    print("Building timestamp lookup tables...")
    timestamp_to_red = {}
    for ts in tqdm(all_timestamps, desc="Processing red timestamps"):
        timestamp_to_red[ts] = red_df[red_df['timestamp'] == ts]
    
    # Create paths for each red target (to show where they've been)
    paths = {}
    path_history = {}  # Store path history to avoid recomputing
    
    def init():
        blue_scatter.set_offsets(np.empty((0, 2)))
        red_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return blue_scatter, red_scatter, time_text
    
    def update(frame):
        # Get current timestamp
        current_time = all_timestamps[frame]
        
        # Convert numpy.datetime64 to Python datetime for strftime
        if isinstance(current_time, np.datetime64):
            current_time = pd.Timestamp(current_time).to_pydatetime()
        
        # Filter data for this timestamp - use pre-computed dataframes for speed
        current_blue = blue_df[blue_df['timestamp'] <= current_time].drop_duplicates('id', keep='last')
        
        if current_time in timestamp_to_red:
            current_red = timestamp_to_red[current_time]
        else:
            current_red = pd.DataFrame(columns=red_df.columns)
        
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
            
            # Initialize path history if needed
            if target_id not in path_history:
                path_history[target_id] = []
            
            # Add current position to history
            path_history[target_id].append(pos)
            
            # Limit trail length to improve performance
            max_trail_length = 20
            if len(path_history[target_id]) > max_trail_length:
                path_history[target_id] = path_history[target_id][-max_trail_length:]
            
            if target_id not in paths:
                # Create new path
                paths[target_id], = ax.plot([], [], 'r-', alpha=0.3, linewidth=1, zorder=2)
            
            # Update path with history
            x_data, y_data = zip(*path_history[target_id]) if path_history[target_id] else ([], [])
            paths[target_id].set_data(x_data, y_data)
        
        # Return updated artists
        artists = [blue_scatter, red_scatter, time_text]
        artists.extend(list(paths.values()))
        return artists
    
    # Determine number of frames (use fewer frames to make it manageable)
    n_frames = len(all_timestamps)
    frame_skip = max(1, n_frames // 300)  # Aim for around 300 frames
    selected_frames = list(range(0, n_frames, frame_skip))
    
    print(f"Creating animation with {len(selected_frames)} frames (sampling every {frame_skip} of {n_frames} total)...")
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=selected_frames,
                      init_func=init, blit=True, interval=100)
    
    # Add grid and labels
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_title('Nova Scotia Battlefield Visualization', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    
    # Make legend more prominent
    ax.legend(loc='upper right', fontsize=12)
    
    # Save animation
    print("Saving animation...")
    
    # Create a custom writer with progress bar
    writer = FFMpegWriter(fps=10, metadata=dict(artist='RF Visualization'), bitrate=3600)
    
    # Setup progress callback for saving
    with tqdm(total=len(selected_frames), desc="Saving frames") as progress_bar:
        def progress_callback(current_frame, total_frames):
            progress_bar.update(1)
        
        ani.save(OUTPUT_FILE, writer=writer, dpi=150, progress_callback=progress_callback)
    
    print(f"Animation saved to {OUTPUT_FILE}")
    plt.close()

if __name__ == "__main__":
    create_animation()