import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from battlefield_simulation import BattlefieldSimulation
from tqdm import tqdm
import time
import matplotlib as mpl
from matplotlib.animation import FFMpegWriter

def create_trajectory_animation(output_filename='battlefield_animation.mp4'):
    print("Starting animation creation process...")
    
    # Create progress bar for data loading
    loading_pbar = tqdm(total=4, desc="Loading data")
    
    # Load simulation data
    sim = BattlefieldSimulation()
    loading_pbar.update(1)
    
    sim.load_terrain_data(
        terrain_data_path="simulation_data/terrain_map.npy",
        elevation_data_path="simulation_data/elevation_map.npy"
    )
    loading_pbar.update(1)
    
    sim.load_observation_data(
        target_csv="synthetic_data/target_observations.csv",
        blue_force_csv="synthetic_data/blue_force_observations.csv"
    )
    loading_pbar.update(1)
    
    # Get all unique timestamps
    target_df = pd.read_csv("synthetic_data/target_observations.csv")
    target_df['timestamp'] = pd.to_datetime(target_df['timestamp'])
    timestamps = sorted(target_df['timestamp'].unique())
    frame_count = min(len(timestamps), 100)  # Limit to 100 frames
    loading_pbar.update(1)
    loading_pbar.close()
    
    print(f"Processing {frame_count} frames...")
    
    # Create figure and background
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot terrain background once
    terrain_colors = [sim.TERRAIN_TYPES[i]['color'] for i in range(len(sim.TERRAIN_TYPES))]
    from matplotlib.colors import ListedColormap
    terrain_cmap = ListedColormap(terrain_colors)
    ax.imshow(sim.terrain_map.T, origin='lower', cmap=terrain_cmap, 
              vmin=0, vmax=len(sim.TERRAIN_TYPES)-1, alpha=0.7)
    
    # Create empty scatter plots for different target types
    infantry_scatter = ax.scatter([], [], color='red', s=50, label='Infantry')
    light_vehicle_scatter = ax.scatter([], [], color='orange', s=50, label='Light Vehicle')
    heavy_vehicle_scatter = ax.scatter([], [], color='darkred', s=50, label='Heavy Vehicle')
    uav_scatter = ax.scatter([], [], color='purple', s=50, label='UAV')
    civilian_scatter = ax.scatter([], [], color='pink', s=50, label='Civilian')
    
    # Create empty scatter plots for blue forces
    blue_force_scatter = ax.scatter([], [], color='blue', s=50, label='Blue Forces')
    
    # Time indicator text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title('Battlefield Simulation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    
    # Initialize the progress bar for frame updates
    progress_bar = tqdm(total=frame_count, desc="Generating frames")
    
    def update(frame):
        # Get data for this timestamp
        current_time = timestamps[frame]
        current_targets = target_df[target_df['timestamp'] == current_time]
        
        # Update time text
        time_text.set_text(f'Time: {current_time}')
        
        # Update target positions by type
        infantry = current_targets[current_targets['target_class'] == 'infantry']
        light_vehicle = current_targets[current_targets['target_class'] == 'light_vehicle']
        heavy_vehicle = current_targets[current_targets['target_class'] == 'heavy_vehicle']
        uav = current_targets[current_targets['target_class'] == 'uav']
        civilian = current_targets[current_targets['target_class'] == 'civilian']
        
        # Convert coordinates to grid indices
        if not infantry.empty:
            x, y = zip(*[sim.data_to_grid(row['x_coord'], row['y_coord']) for _, row in infantry.iterrows()])
            infantry_scatter.set_offsets(np.column_stack([x, y]))
        else:
            infantry_scatter.set_offsets(np.empty((0, 2)))
        
        if not light_vehicle.empty:
            x, y = zip(*[sim.data_to_grid(row['x_coord'], row['y_coord']) for _, row in light_vehicle.iterrows()])
            light_vehicle_scatter.set_offsets(np.column_stack([x, y]))
        else:
            light_vehicle_scatter.set_offsets(np.empty((0, 2)))
        
        if not heavy_vehicle.empty:
            x, y = zip(*[sim.data_to_grid(row['x_coord'], row['y_coord']) for _, row in heavy_vehicle.iterrows()])
            heavy_vehicle_scatter.set_offsets(np.column_stack([x, y]))
        else:
            heavy_vehicle_scatter.set_offsets(np.empty((0, 2)))
        
        if not uav.empty:
            x, y = zip(*[sim.data_to_grid(row['x_coord'], row['y_coord']) for _, row in uav.iterrows()])
            uav_scatter.set_offsets(np.column_stack([x, y]))
        else:
            uav_scatter.set_offsets(np.empty((0, 2)))
        
        if not civilian.empty:
            x, y = zip(*[sim.data_to_grid(row['x_coord'], row['y_coord']) for _, row in civilian.iterrows()])
            civilian_scatter.set_offsets(np.column_stack([x, y]))
        else:
            civilian_scatter.set_offsets(np.empty((0, 2)))
        
        # Update blue forces (this would need to be implemented similarly)
        blue_force_scatter.set_offsets(np.empty((0, 2)))  # Placeholder
        
        # Update progress bar
        progress_bar.update(1)
        
        return [infantry_scatter, light_vehicle_scatter, heavy_vehicle_scatter, 
                uav_scatter, civilian_scatter, blue_force_scatter, time_text]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=frame_count, 
                       blit=True, interval=200, repeat=False)
    
    # Create a custom writer with progress bar for saving
    print("\nSaving animation...")
    
    # Setup FFmpeg writer with progress output
    writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    
    # Save the animation with progress tracking
    with tqdm(total=100, desc="Encoding video") as pbar:
        # Open the writer manually
        writer.setup(fig, output_filename, dpi=100)
        
        # Write frames one by one with progress updates
        for i in range(frame_count):
            # Manually update the figure for each frame
            update(i)
            writer.grab_frame()
            
            # Update encoding progress bar (approximate)
            pbar.update(100 // frame_count)
        
        # Close the writer
        writer.finish()
    
    progress_bar.close()
    print(f"Animation saved to {output_filename}")
    
    return ani

# Call the function to create the animation
animation = create_trajectory_animation()
plt.show()