def create_prediction_animation(predictor, output_filename='target_prediction_animation.mp4'):
    """
    Create animation of target predictions with terrain and blue forces.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import os
    from tqdm import tqdm
    from matplotlib.colors import ListedColormap
    from datetime import timedelta
    import torch
    
    # Load data
    print("Loading data...")
    blue_df = pd.read_csv("data/blue_locations.csv")
    red_df = pd.read_csv("data/red_sightings.csv")
    red_df['datetime'] = pd.to_datetime(red_df['datetime'])
    
    # Load terrain data
    terrain_map = np.load("adapted_data/terrain_map.npy")
    elevation_map = np.load("adapted_data/elevation_map.npy")
    
    # Create terrain colormap
    terrain_colors = [
        'blue',       # 0: Water
        'gray',       # 1: Urban
        'yellow',     # 2: Agricultural
        'darkgreen',  # 3: Forest
        'lightgreen', # 4: Grassland
        'brown',      # 5: Barren
        'cyan',       # 6: Wetland
        'white'       # 7: Snow/Ice
    ]
    terrain_cmap = ListedColormap(terrain_colors)
    
    # Get coordinate bounds
    lon_min, lon_max = red_df['longitude'].min(), red_df['longitude'].max()
    lat_min, lat_max = red_df['latitude'].min(), red_df['latitude'].max()
    
    # Add some padding
    lon_padding = (lon_max - lon_min) * 0.05
    lat_padding = (lat_max - lat_min) * 0.05
    lon_min -= lon_padding
    lon_max += lon_padding
    lat_min -= lat_padding
    lat_max += lat_padding
    
    # Get all unique timestamps sorted
    all_timestamps = sorted(pd.unique(red_df['datetime']))
    
    # Only use a subset of frames to keep the animation manageable
    frame_skip = max(1, len(all_timestamps) // 300)
    selected_frames = all_timestamps[::frame_skip]
    
    print(f"Creating animation with {len(selected_frames)} frames...")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create empty collections for visualization
    blue_scatter = ax.scatter([], [], c='blue', s=120, marker='^', label='Blue Forces', zorder=10, edgecolor='black')
    red_scatter = ax.scatter([], [], c='red', s=80, marker='o', label='Red Forces', zorder=10, edgecolor='black')
    prediction_lines = {}
    confidence_ellipses = {}
    target_trails = {}
    
    # Add terrain background (adjust extent to match coordinate system)
    terrain_img = ax.imshow(terrain_map, cmap=terrain_cmap, alpha=0.5, 
                          extent=[lon_min, lon_max, lat_min, lat_max], 
                          aspect='auto', zorder=0)
    
    # Time display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                      bbox=dict(facecolor='white', alpha=0.8), zorder=20)
    
    # Set axis limits
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title('Target Movement Prediction')
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
        current_red = red_df[red_df['datetime'] == current_time]
        
        # Update blue & red positions
        blue_scatter.set_offsets(blue_df[['longitude', 'latitude']].values)
        red_scatter.set_offsets(current_red[['longitude', 'latitude']].values)
        
        # Update time display
        time_text.set_text(f'Time: {current_time.strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Clean up previous predictions
        for target_id in list(prediction_lines.keys()):
            if prediction_lines[target_id] in ax.lines:
                ax.lines.remove(prediction_lines[target_id])
            del prediction_lines[target_id]
        
        for target_id in list(confidence_ellipses.keys()):
            for ellipse in confidence_ellipses[target_id]:
                if ellipse in ax.patches:
                    ellipse.remove()
            confidence_ellipses[target_id] = []
        
        # Update target trails
        for target_id in list(target_trails.keys()):
            if target_trails[target_id] in ax.lines:
                ax.lines.remove(target_trails[target_id])
            del target_trails[target_id]
        
        # Make predictions for each target
        for _, row in current_red.iterrows():
            target_id = row['target_id']
            
            # Get history data for this target
            target_history = red_df[(red_df['target_id'] == target_id) & 
                                  (red_df['datetime'] <= current_time)].sort_values('datetime')
            
            # Update target trail
            target_points = target_history[['longitude', 'latitude']].values[-10:]
            if len(target_points) >= 2:
                line, = ax.plot(target_points[:, 0], target_points[:, 1], 'r-', alpha=0.4, linewidth=1.5, zorder=5)
                target_trails[target_id] = line
            
            # Generate prediction if enough history
            if len(target_history) >= predictor.config['sequence_length']:
                # Prepare input sequence
                history = target_history.tail(predictor.config['sequence_length'])
                
                # Extract features for prediction
                input_sequence = []
                for _, hist_row in history.iterrows():
                    # Base features: x, y, altitude
                    features = [hist_row['longitude'], hist_row['latitude']]
                    
                    # Add altitude if available
                    if 'altitude_m' in hist_row:
                        features.append(hist_row['altitude_m'])
                    else:
                        features.append(0)  # Default altitude
                    
                    # Add target class one-hot encoding
                    if 'target_class' in hist_row:
                        classes = red_df['target_class'].unique()
                        for cls in classes:
                            features.append(1.0 if hist_row['target_class'] == cls else 0.0)
                    
                    # Add time features
                    dt = hist_row['datetime']
                    hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                    hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                    minute_sin = np.sin(2 * np.pi * dt.minute / 60)
                    minute_cos = np.cos(2 * np.pi * dt.minute / 60)
                    day_sin = np.sin(2 * np.pi * dt.dayofweek / 7)
                    day_cos = np.cos(2 * np.pi * dt.dayofweek / 7)
                    is_morning = 1.0 if (6 <= dt.hour < 12) else 0.0
                    is_afternoon = 1.0 if (12 <= dt.hour < 18) else 0.0
                    is_evening = 1.0 if (18 <= dt.hour < 22) else 0.0
                    is_night = 1.0 if (dt.hour >= 22 or dt.hour < 6) else 0.0
                    
                    features.extend([
                        hour_sin, hour_cos, 
                        minute_sin, minute_cos, 
                        day_sin, day_cos,
                        is_morning, is_afternoon, is_evening, is_night
                    ])
                    
                    input_sequence.append(features)
                
                # Predict
                input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
                predictions = predictor.predict(input_tensor)
                
                # Add prediction line
                mean_traj = predictions['mean']
                lower_ci = predictions['lower_ci']
                upper_ci = predictions['upper_ci']
                
                # Plot predicted trajectory
                line, = ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r--', linewidth=2, alpha=0.8, zorder=6)
                prediction_lines[target_id] = line
                
                # Plot confidence ellipses
                confidence_ellipses[target_id] = []
                for i in range(len(mean_traj)):
                    ellipse = plt.matplotlib.patches.Ellipse(
                        (mean_traj[i, 0], mean_traj[i, 1]),
                        width=upper_ci[i, 0] - lower_ci[i, 0],
                        height=upper_ci[i, 1] - lower_ci[i, 1],
                        color='red', alpha=0.2, zorder=4
                    )
                    ax.add_patch(ellipse)
                    confidence_ellipses[target_id].append(ellipse)
                
                # Check if we have ground truth for the prediction
                future_time = current_time + timedelta(seconds=300)  # 5-minute prediction
                future_point = red_df[(red_df['target_id'] == target_id) & 
                                    (red_df['datetime'] <= future_time)].sort_values('datetime').tail(1)
                
                if len(future_point) > 0:
                    # Plot ground truth point
                    ax.scatter(future_point['longitude'].values[0], future_point['latitude'].values[0],
                              marker='*', color='green', s=100, zorder=11, edgecolor='black')
        
        # Update progress bar
        progress_bar.update(1)
        
        # Return updated artists (needed for blitting)
        artists = [blue_scatter, red_scatter, time_text]
        artists.extend(list(prediction_lines.values()))
        artists.extend([e for ellipses in confidence_ellipses.values() for e in ellipses])
        artists.extend(list(target_trails.values()))
        return artists
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(selected_frames),
                      init_func=init, blit=True, interval=100)
    
    # Save animation
    print("\nSaving animation...")
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Target Prediction'), bitrate=3600)
    
    with tqdm(total=100, desc="Encoding video") as pbar:
        ani.save(output_filename, writer=writer, dpi=150,
               progress_callback=lambda i, n: pbar.update(100/n))
    
    progress_bar.close()
    print(f"Animation saved to {output_filename}")
    return ani