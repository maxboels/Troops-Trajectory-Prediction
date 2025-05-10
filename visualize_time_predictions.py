def visualize_time_of_day_predictions(predictor, target_data, target_id, base_history=None):
    """
    Visualize how predictions for the same target change based on time of day.
    
    Args:
        predictor: Trained TargetMovementPredictor model
        target_data: DataFrame with target observations
        target_id: ID of the target to predict
        base_history: Optional dataframe with specific history to use for all predictions
                     (if None, uses the actual history)
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    
    # Filter data for this target
    target_df = target_data[target_data['target_id'] == target_id].copy()
    
    # Convert datetime column if necessary
    if target_df['datetime'].dtype == object:
        target_df['datetime'] = pd.to_datetime(target_df['datetime'])
    
    # Get a representative date from the data
    base_date = target_df['datetime'].iloc[0].date()
    
    # Sample timestamps from different parts of the day
    times = [
        datetime.combine(base_date, datetime.strptime("08:00:00", "%H:%M:%S").time()),
        datetime.combine(base_date, datetime.strptime("14:00:00", "%H:%M:%S").time()),
        datetime.combine(base_date, datetime.strptime("19:00:00", "%H:%M:%S").time()),
        datetime.combine(base_date, datetime.strptime("23:00:00", "%H:%M:%S").time())
    ]
    
    labels = ["Morning (8:00 AM)", "Afternoon (2:00 PM)", "Evening (7:00 PM)", "Night (11:00 PM)"]
    colors = ['#FFA500', '#32CD32', '#6A5ACD', '#191970']  # orange, green, purple, navy
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # If base_history is not provided, use a consistent segment from the data
    if base_history is None:
        # Find a segment with enough points for prediction
        seq_length = predictor.config['sequence_length']
        mid_idx = len(target_df) // 3  # Use first third to avoid running out of data
        base_history = target_df.iloc[mid_idx:mid_idx+seq_length].copy()
    
    # Get the base trajectory for reference
    base_x = base_history['longitude'].values
    base_y = base_history['latitude'].values
    
    # Create predictions for each time of day
    for i, (time, label, color, ax) in enumerate(zip(times, labels, colors, axes)):
        # Make a copy of the history with modified timestamps
        modified_history = base_history.copy()
        
        # Calculate time differences to maintain the same intervals
        original_start = modified_history['datetime'].iloc[0]
        time_diffs = [(t - original_start).total_seconds() for t in modified_history['datetime']]
        
        # Update timestamps to the new time of day while preserving intervals
        modified_history['datetime'] = [time + timedelta(seconds=diff) for diff in time_diffs]
        
        # Generate features for prediction
        input_sequence = []
        for _, row in modified_history.iterrows():
            # Base features: x, y, altitude
            features = [row['longitude'], row['latitude']]
            
            # Add altitude if available
            if 'altitude_m' in row:
                features.append(row['altitude_m'])
            else:
                features.append(0)
            
            # Add target class one-hot if needed
            if 'target_class' in row:
                classes = target_data['target_class'].unique()
                for cls in classes:
                    features.append(1.0 if row['target_class'] == cls else 0.0)
            
            # Add time features
            dt = row['datetime']
            hour_sin = np.sin(2 * np.pi * dt.hour / 24)
            hour_cos = np.cos(2 * np.pi * dt.hour / 24)
            minute_sin = np.sin(2 * np.pi * dt.minute / 60)
            minute_cos = np.cos(2 * np.pi * dt.minute / 60)
            day_sin = np.sin(2 * np.pi * dt.dayofweek / 7)
            day_cos = np.cos(2 * np.pi * dt.dayofweek / 7)
            
            # Time of day indicators
            is_morning = 1.0 if (6 <= dt.hour < 12) else 0.0
            is_afternoon = 1.0 if (12 <= dt.hour < 18) else 0.0
            is_evening = 1.0 if (18 <= dt.hour < 22) else 0.0
            is_night = 1.0 if (dt.hour >= 22 or dt.hour < 6) else 0.0
            
            # Add all temporal features
            features.extend([
                hour_sin, hour_cos, 
                minute_sin, minute_cos, 
                day_sin, day_cos,
                is_morning, is_afternoon, is_evening, is_night
            ])
            
            input_sequence.append(features)
        
        # Convert to tensor and predict
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
        predictions = predictor.predict(input_tensor)
        
        # Plot on this subplot
        ax.plot(base_x, base_y, 'b-', linewidth=2, label='Base Trajectory')
        ax.scatter(base_x[-1], base_y[-1], c='blue', s=100, marker='o', label='Last Seen Position')
        
        # Plot prediction
        pred_x = predictions['mean'][:, 0]
        pred_y = predictions['mean'][:, 1]
        lower_x = predictions['lower_ci'][:, 0]
        lower_y = predictions['lower_ci'][:, 1]
        upper_x = predictions['upper_ci'][:, 0]
        upper_y = predictions['upper_ci'][:, 1]
        
        ax.plot(pred_x, pred_y, '-', color=color, linewidth=2, label='Predicted Path')
        ax.scatter(pred_x[-1], pred_y[-1], c=color, s=100, marker='x', label='Final Position')
        
        # Plot confidence regions
        for j in range(len(pred_x)):
            ellipse = plt.matplotlib.patches.Ellipse(
                (pred_x[j], pred_y[j]),
                width=upper_x[j] - lower_x[j],
                height=upper_y[j] - lower_y[j],
                color=color, alpha=0.2
            )
            ax.add_patch(ellipse)
        
        # Add time of day info
        ax.set_title(f"{label}", fontsize=14)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        if i == 0:
            ax.legend(loc='upper left', fontsize=10)
        
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Impact of Time of Day on Target {target_id} Movement Prediction', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    return fig


def visualize_weekly_pattern_predictions(predictor, target_data, target_id, base_history=None):
    """
    Visualize how predictions change based on day of the week.
    
    Args:
        predictor: Trained TargetMovementPredictor model
        target_data: DataFrame with target observations
        target_id: ID of the target to predict
        base_history: Optional dataframe with specific history to use for all predictions
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    
    # Filter data for this target
    target_df = target_data[target_data['target_id'] == target_id].copy()
    
    # Convert datetime column if necessary
    if target_df['datetime'].dtype == object:
        target_df['datetime'] = pd.to_datetime(target_df['datetime'])
    
    # Get a representative date from the data
    base_date = target_df['datetime'].iloc[0].date()
    
    # Sample different days of the week (keeping time constant)
    base_time = datetime.combine(base_date, datetime.strptime("14:00:00", "%H:%M:%S").time())
    
    # Calculate dates for different days of the week
    weekday_offsets = []
    for i in range(7):
        # Find the date for this day of week
        target_weekday = i  # 0=Monday, 6=Sunday
        current_weekday = base_time.weekday()
        days_to_add = (target_weekday - current_weekday) % 7
        weekday_offsets.append(days_to_add)
    
    days = [base_time + timedelta(days=offset) for offset in weekday_offsets]
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    colors = plt.cm.rainbow(np.linspace(0, 1, 7))
    
    # Create figure with 7 subplots (for each day)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Remove the last subplot (we only need 7)
    axes[-1].axis('off')
    axes = axes[:-1]
    
    # If base_history is not provided, use a consistent segment from the data
    if base_history is None:
        # Find a segment with enough points for prediction
        seq_length = predictor.config['sequence_length']
        mid_idx = len(target_df) // 3  # Use first third to avoid running out of data
        base_history = target_df.iloc[mid_idx:mid_idx+seq_length].copy()
    
    # Get the base trajectory for reference
    base_x = base_history['longitude'].values
    base_y = base_history['latitude'].values
    
    # Create predictions for each day of the week
    for i, (day, day_name, color, ax) in enumerate(zip(days, day_names, colors, axes)):
        # Make a copy of the history with modified timestamps
        modified_history = base_history.copy()
        
        # Calculate time differences to maintain the same intervals
        original_start = modified_history['datetime'].iloc[0]
        time_diffs = [(t - original_start).total_seconds() for t in modified_history['datetime']]
        
        # Update timestamps to the new day while preserving intervals
        modified_history['datetime'] = [day + timedelta(seconds=diff) for diff in time_diffs]
        
        # Generate features for prediction
        input_sequence = []
        for _, row in modified_history.iterrows():
            # Base features: x, y, altitude
            features = [row['longitude'], row['latitude']]
            
            # Add altitude if available
            if 'altitude_m' in row:
                features.append(row['altitude_m'])
            else:
                features.append(0)
            
            # Add target class one-hot if needed
            if 'target_class' in row:
                classes = target_data['target_class'].unique()
                for cls in classes:
                    features.append(1.0 if row['target_class'] == cls else 0.0)
            
            # Add time features
            dt = row['datetime']
            hour_sin = np.sin(2 * np.pi * dt.hour / 24)
            hour_cos = np.cos(2 * np.pi * dt.hour / 24)
            minute_sin = np.sin(2 * np.pi * dt.minute / 60)
            minute_cos = np.cos(2 * np.pi * dt.minute / 60)
            day_sin = np.sin(2 * np.pi * dt.dayofweek / 7)
            day_cos = np.cos(2 * np.pi * dt.dayofweek / 7)
            
            # Time of day indicators
            is_morning = 1.0 if (6 <= dt.hour < 12) else 0.0
            is_afternoon = 1.0 if (12 <= dt.hour < 18) else 0.0
            is_evening = 1.0 if (18 <= dt.hour < 22) else 0.0
            is_night = 1.0 if (dt.hour >= 22 or dt.hour < 6) else 0.0
            
            # Add all temporal features
            features.extend([
                hour_sin, hour_cos, 
                minute_sin, minute_cos, 
                day_sin, day_cos,
                is_morning, is_afternoon, is_evening, is_night
            ])
            
            input_sequence.append(features)
        
        # Convert to tensor and predict
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
        predictions = predictor.predict(input_tensor)
        
        # Plot on this subplot
        ax.plot(base_x, base_y, 'b-', linewidth=2, label='Base Trajectory')
        ax.scatter(base_x[-1], base_y[-1], c='blue', s=100, marker='o')
        
        # Plot prediction
        pred_x = predictions['mean'][:, 0]
        pred_y = predictions['mean'][:, 1]
        
        ax.plot(pred_x, pred_y, '-', color=color, linewidth=2, label='Predicted Path')
        ax.scatter(pred_x[-1], pred_y[-1], c=color, s=100, marker='x')
        
        # Add day info
        ax.set_title(f"{day_name}", fontsize=14)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        if i == 0:
            ax.legend(loc='upper left', fontsize=10)
        
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Weekly Movement Patterns for Target {target_id}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    return fig


def plot_temporal_feature_importance(predictor, test_loader):
    """
    Analyze and visualize the importance of different temporal features.
    
    This function uses a technique similar to permutation importance:
    1. Get baseline performance on test data
    2. For each temporal feature, randomly permute its values and measure performance drop
    3. Features that cause larger performance drops when permuted are more important
    
    Args:
        predictor: Trained TargetMovementPredictor model
        test_loader: DataLoader for test data
        
    Returns:
        matplotlib figure with feature importance visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from tqdm import tqdm
    
    # Temporal feature indices (adjust based on your actual feature ordering)
    # Assuming the temporal features start after base + class features
    # This is an approximation - you'll need to adjust based on your actual feature indices
    feature_indices = {
        'Hour (sin/cos)': [7, 8],  # Indices for hour_sin, hour_cos
        'Minute (sin/cos)': [9, 10],  # Indices for minute_sin, minute_cos
        'Day of week (sin/cos)': [11, 12],  # Indices for day_sin, day_cos
        'Morning indicator': [13],  # Index for is_morning
        'Afternoon indicator': [14],  # Index for is_afternoon
        'Evening indicator': [15],  # Index for is_evening
        'Night indicator': [16],  # Index for is_night
    }
    
    # Get baseline performance
    baseline_mse = 0.0
    baseline_count = 0
    
    predictor.transformer_model.eval()
    if predictor.terrain_model is not None:
        predictor.terrain_model.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing baseline performance"):
            inputs = batch['input'].to(predictor.config['device'])
            targets = batch['target'].to(predictor.config['device'])
            
            # Normalize
            inputs_norm, targets_norm = predictor.normalize_features(inputs, targets)
            
            # Process terrain if available
            terrain_features = None
            if predictor.terrain_model is not None and 'terrain' in batch and batch['terrain'] is not None:
                terrain = batch['terrain'].to(predictor.config['device'])
                terrain_features = predictor.terrain_model(terrain)
                vis_seq = terrain_features.unsqueeze(1).expand(-1, 1, -1)
            else:
                vis_dim = predictor.config.get('terrain_feature_dim', 32)
                vis_seq = torch.zeros(
                    (inputs_norm.shape[0], 1, vis_dim), 
                    device=predictor.config['device']
                )
            
            # Forward pass
            pred_mean, _ = predictor.transformer_model(inputs_norm, vis_seq)
            
            # Extract predictions for forecast horizon
            pred_mean = pred_mean[:, -predictor.config['prediction_horizon']:, :]
            
            # Compute MSE
            mse = ((pred_mean - targets_norm) ** 2).mean().item()
            baseline_mse += mse * inputs.size(0)
            baseline_count += inputs.size(0)
    
    baseline_mse /= baseline_count
    print(f"Baseline MSE: {baseline_mse:.4f}")
    
    # Compute importance for each feature group
    feature_importance = {}
    
    for feature_name, feature_idxs in feature_indices.items():
        # Compute performance with this feature permuted
        permuted_mse = 0.0
        permuted_count = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Processing {feature_name}"):
                inputs = batch['input'].to(predictor.config['device'])
                targets = batch['target'].to(predictor.config['device'])
                
                # Create a permuted version of inputs
                permuted_inputs = inputs.clone()
                
                # Permute the feature values across the batch
                for idx in feature_idxs:
                    # For each sequence in the batch
                    for t in range(inputs.size(1)):  # For each time step
                        # Get values for this feature at this timestep
                        values = permuted_inputs[:, t, idx].clone()
                        # Shuffle these values
                        shuffled = values[torch.randperm(values.size(0))]
                        # Replace with shuffled values
                        permuted_inputs[:, t, idx] = shuffled
                
                # Normalize
                permuted_inputs_norm, targets_norm = predictor.normalize_features(permuted_inputs, targets)
                
                # Process terrain if available
                terrain_features = None
                if predictor.terrain_model is not None and 'terrain' in batch and batch['terrain'] is not None:
                    terrain = batch['terrain'].to(predictor.config['device'])
                    terrain_features = predictor.terrain_model(terrain)
                    vis_seq = terrain_features.unsqueeze(1).expand(-1, 1, -1)
                else:
                    vis_dim = predictor.config.get('terrain_feature_dim', 32)
                    vis_seq = torch.zeros(
                        (permuted_inputs_norm.shape[0], 1, vis_dim), 
                        device=predictor.config['device']
                    )
                
                # Forward pass with permuted inputs
                pred_mean, _ = predictor.transformer_model(permuted_inputs_norm, vis_seq)
                
                # Extract predictions for forecast horizon
                pred_mean = pred_mean[:, -predictor.config['prediction_horizon']:, :]
                
                # Compute MSE
                mse = ((pred_mean - targets_norm) ** 2).mean().item()
                permuted_mse += mse * inputs.size(0)
                permuted_count += inputs.size(0)
        
        permuted_mse /= permuted_count
        print(f"{feature_name} permuted MSE: {permuted_mse:.4f}")
        
        # Compute importance as performance drop
        importance = permuted_mse - baseline_mse
        feature_importance[feature_name] = importance
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importances)
    features = [features[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]
    
    # Plot
    bars = ax.barh(features, importances, color='skyblue')
    
    # Add values to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{importances[i]:.4f}', ha='left', va='center')
    
    ax.set_xlabel('Feature Importance (increase in MSE when permuted)')
    ax.set_title('Temporal Feature Importance Analysis')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig
