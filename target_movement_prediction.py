"""
Target Movement Prediction - Predicting where targets will move when out of view
Based on transformer architecture with terrain awareness
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import math
from datetime import timedelta, datetime
import time

import torch.serialization
from sklearn.preprocessing import StandardScaler

# Add StandardScaler and numpy scalar to the safe globals list
torch.serialization.add_safe_globals([StandardScaler, np.core.multiarray.scalar])
torch.serialization.add_safe_globals([StandardScaler])

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TargetTransformerPredictor(nn.Module):
    """
    Transformer model that fuses positional history via causal self-attention
    and terrain/elevation embeddings via cross-attention.
    """
    def __init__(self, pos_dim, vis_dim, hidden_dim, output_dim, n_layers, n_heads, dropout=0.1):
        super().__init__()
        # Project position features
        self.pos_proj = nn.Linear(pos_dim, hidden_dim)
        # Project visual features to hidden for memory
        self.vis_proj = nn.Linear(vis_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out_mean = nn.Linear(hidden_dim, output_dim)
        self.out_logvar = nn.Linear(hidden_dim, output_dim)

    def _gen_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)

    def forward(self, pos_seq, vis_seq):
        """
        Args:
            pos_seq: [B, T, pos_dim]  positional history features
            vis_seq: [B, M, vis_dim]  spatial/vision embeddings
        Returns:
            mean:   [B, T, output_dim]
            logvar: [B, T, output_dim]
        """
        B, T, _ = pos_seq.shape
        # Project
        Q = self.pos_proj(pos_seq)      # [B, T, H]
        K = self.vis_proj(vis_seq)      # [B, M, H]
        V = K                           # share for cross-attn
        # Add positional encoding to queries
        Q = self.pos_enc(Q)

        # Causal self-attention mask
        tgt_mask = self._gen_causal_mask(T, pos_seq.device)

        # Decode: self-attend on Q (causal) and cross-attend on vis embeddings
        H_dec = self.decoder(
            tgt=Q,
            memory=K,
            tgt_mask=tgt_mask
        )  # [B, T, H]

        # Output projections
        mean = self.out_mean(H_dec)
        logvar = self.out_logvar(H_dec)
        return mean, logvar


class TerrainFeatureExtractor(nn.Module):
    """
    CNN module to extract features from terrain and elevation data.
    """
    def __init__(self, input_channels, output_dim):
        super(TerrainFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TargetTrajectoryDataset(Dataset):
    """
    Dataset for training the target movement predictor with enhanced temporal features.
    """
    def __init__(self, 
                target_data,
                terrain_data=None, 
                elevation_data=None,
                blue_force_data=None,
                sequence_length=10, 
                prediction_horizon=5, 
                stride=1,
                terrain_window_size=32):
        """
        Initialize the dataset.
        
        Args:
            target_data: DataFrame with target observations (red sightings)
            terrain_data: Terrain map as numpy array
            elevation_data: Elevation map as numpy array
            blue_force_data: DataFrame with blue force locations
            sequence_length: Number of timesteps to use as input
            prediction_horizon: Number of future timesteps to predict
            stride: Stride between sequences
            terrain_window_size: Size of terrain patch to extract
        """
        self.target_data = target_data
        self.terrain_data = terrain_data
        self.elevation_data = elevation_data
        self.blue_force_data = blue_force_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.terrain_window_size = terrain_window_size
        
        # Process data
        self.process_data()
        
    def process_data(self):
        # Group by target ID
        target_groups = self.target_data.groupby('target_id')
        
        self.sequences = []
        
        for target_id, group in target_groups:
            # Sort by timestamp
            group = group.sort_values('datetime')
            
            # Skip if not enough data points
            if len(group) < self.sequence_length + self.prediction_horizon:
                continue
            
            # Extract features
            x_coords = group['longitude'].values  # Note: using lon for x
            y_coords = group['latitude'].values   # Note: using lat for y
            
            # Add altitude if available
            has_altitude = 'altitude_m' in group.columns
            if has_altitude:
                altitude = group['altitude_m'].values
            else:
                altitude = np.zeros_like(x_coords)
                
            # Extract target class as one-hot encoding
            target_classes = None
            if 'target_class' in group.columns:
                # Get unique target classes
                all_classes = self.target_data['target_class'].unique()
                
                # Create one-hot encoding
                target_classes = np.zeros((len(group), len(all_classes)))
                for i, tclass in enumerate(all_classes):
                    target_classes[:, i] = (group['target_class'] == tclass).astype(int)
            
            # Create sequences
            for i in range(0, len(group) - self.sequence_length - self.prediction_horizon + 1, self.stride):
                # Input sequence
                input_sequence = []
                
                for j in range(i, i + self.sequence_length):
                    # Base features: x, y, altitude
                    features = [x_coords[j], y_coords[j]]
                    
                    if has_altitude:
                        features.append(altitude[j])
                    else:
                        features.append(0)  # Default altitude
                    
                    # Add target class if available
                    if target_classes is not None:
                        features.extend(target_classes[j])
                    
                    # Extract rich temporal features from timestamp
                    dt = group['datetime'].iloc[j]
                    
                    # 1. Cyclical encoding of hour (captures day/night patterns)
                    hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                    hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                    
                    # 2. Cyclical encoding of minutes
                    minute_sin = np.sin(2 * np.pi * dt.minute / 60)
                    minute_cos = np.cos(2 * np.pi * dt.minute / 60)
                    
                    # 3. Day of week (for weekly patterns)
                    day_sin = np.sin(2 * np.pi * dt.dayofweek / 7)
                    day_cos = np.cos(2 * np.pi * dt.dayofweek / 7)
                    
                    # 4. Time of day indicators
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
                
                # Target sequence - just x, y coordinates
                target_sequence = []
                
                for j in range(i + self.sequence_length, i + self.sequence_length + self.prediction_horizon):
                    target_sequence.append([x_coords[j], y_coords[j]])
                
                # Extract terrain patch if available
                terrain_patch = None
                if self.terrain_data is not None and self.elevation_data is not None:
                    # Get the latest position
                    latest_y = y_coords[i + self.sequence_length - 1] 
                    latest_x = x_coords[i + self.sequence_length - 1]
                    
                    # Convert lat/lon to terrain map indices
                    try:
                        # Get bounds of coordinate system
                        min_lat = np.min(self.target_data['latitude'])
                        max_lat = np.max(self.target_data['latitude'])
                        min_lon = np.min(self.target_data['longitude'])
                        max_lon = np.max(self.target_data['longitude'])
                        
                        # Map to terrain indices
                        terrain_height, terrain_width = self.terrain_data.shape
                        x_idx = int((latest_x - min_lon) / (max_lon - min_lon) * (terrain_width - 1))
                        y_idx = int((max_lat - latest_y) / (max_lat - min_lat) * (terrain_height - 1))
                        
                        # Ensure indices are within bounds
                        x_idx = max(0, min(x_idx, terrain_width - 1))
                        y_idx = max(0, min(y_idx, terrain_height - 1))
                        
                        # Extract terrain patch centered on latest position
                        half_window = self.terrain_window_size // 2
                        
                        # Ensure within bounds
                        x_min = max(0, x_idx - half_window)
                        x_max = min(terrain_width, x_idx + half_window)
                        y_min = max(0, y_idx - half_window)
                        y_max = min(terrain_height, y_idx + half_window)
                        
                        # Extract patches
                        terrain_patch = self.terrain_data[y_min:y_max, x_min:x_max]
                        elevation_patch = self.elevation_data[y_min:y_max, x_min:x_max]
                        
                        # Pad if necessary
                        if terrain_patch.shape[0] < self.terrain_window_size or terrain_patch.shape[1] < self.terrain_window_size:
                            padded_terrain = np.zeros((self.terrain_window_size, self.terrain_window_size))
                            padded_elevation = np.zeros((self.terrain_window_size, self.terrain_window_size))
                            
                            # Copy available data
                            h, w = terrain_patch.shape
                            padded_terrain[:h, :w] = terrain_patch
                            padded_elevation[:h, :w] = elevation_patch
                            
                            terrain_patch = padded_terrain
                            elevation_patch = padded_elevation
                        
                        # Stack terrain and elevation
                        terrain_patch = np.stack([terrain_patch, elevation_patch], axis=0)
                    except Exception as e:
                        print(f"Error extracting terrain patch: {e}")
                        # Create a dummy terrain patch if extraction fails
                        terrain_patch = np.zeros((2, self.terrain_window_size, self.terrain_window_size))
                
                # Store metadata as additional information (but not used in training)
                metadata = {
                    'target_id': target_id,
                    'start_idx': i,
                    'start_time_str': str(group['datetime'].iloc[i + self.sequence_length - 1]),
                    'end_time_str': str(group['datetime'].iloc[i + self.sequence_length + self.prediction_horizon - 1])
                }
                
                # Add to sequences
                self.sequences.append({
                    'input': np.array(input_sequence, dtype=np.float32),
                    'target': np.array(target_sequence, dtype=np.float32),
                    'terrain': terrain_patch,
                    'metadata': metadata
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Prepare input - ensure it has the right shape
        input_data = sequence['input']
        
        # Check if it's already in sequence format
        if len(input_data.shape) < 2:
            # If it's flat, reshape it to [1, features]
            input_seq = torch.tensor(input_data).unsqueeze(0)
        else:
            # Already has sequence dimension
            input_seq = torch.tensor(input_data, dtype=torch.float32)
        
        # Prepare target
        target_seq = torch.tensor(sequence['target'], dtype=torch.float32)
        
        # Prepare terrain if available
        terrain = None
        if sequence['terrain'] is not None:
            terrain = torch.tensor(sequence['terrain'], dtype=torch.float32)
        
        # Extract metadata that's safe for DataLoader (no timestamps)
        metadata = sequence['metadata']
        
        # Return only the essential elements needed for training
        return {
            'input': input_seq,
            'target': target_seq,
            'terrain': terrain,
            'target_id': metadata['target_id'],
            'start_idx': metadata['start_idx']
        }

class TargetMovementPredictor:
    """
    Main class for predicting target movements.
    """
    def __init__(self, 
                config=None,
                terrain_data_path=None, 
                elevation_data_path=None):
        """
        Initialize the predictor.
        
        Args:
            config: Model configuration dictionary
            terrain_data_path: Path to terrain map file
            elevation_data_path: Path to elevation map file
        """
        # Default configuration
        default_config = {
            'hidden_dim': 128,
            'n_layers': 3,
            'n_heads': 4,
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_epochs': 50, # default is 50, use 2 for debugging
            'sequence_length': 10,
            'prediction_horizon': 5,
            'terrain_feature_dim': 32,
            'use_terrain': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Update with provided config
        self.config = default_config
        if config is not None:
            self.config.update(config)
        
        # Load terrain and elevation data if paths provided
        self.terrain_data = None
        self.elevation_data = None
        
        if terrain_data_path is not None and os.path.exists(terrain_data_path):
            self.terrain_data = np.load(terrain_data_path)
            print(f"Loaded terrain data with shape: {self.terrain_data.shape}")
        
        if elevation_data_path is not None and os.path.exists(elevation_data_path):
            self.elevation_data = np.load(elevation_data_path)
            print(f"Loaded elevation data with shape: {self.elevation_data.shape}")
        
        # Initialize models
        self.transformer_model = None
        self.terrain_model = None
        self.input_scaler = None
        self.output_scaler = None
        
        # Loss function for position prediction
        self.mse_loss = nn.MSELoss()
    
    def setup_feature_scaling(self, train_dataset):
        """
        Set up feature scaling based on training data.
        
        Args:
            train_dataset: Training dataset to fit scalers on
        """
        # Initialize scalers
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        # Collect all input and output features from the training set
        all_inputs = []
        all_outputs = []
        
        # Use a subset of the training data for fitting scalers
        max_samples = min(len(train_dataset), 1000)  # Limit to 1000 samples for efficiency
        indices = np.random.choice(len(train_dataset), max_samples, replace=False)
        
        for idx in indices:
            sample = train_dataset[idx]
            inputs = sample['input']
            outputs = sample['target']
            
            # Handle different tensor shapes
            if len(inputs.shape) == 3:  # [batch, seq, features]
                inputs = inputs.reshape(-1, inputs.shape[-1])
            elif len(inputs.shape) == 2:  # [seq, features]
                pass  # Already in the right shape
            else:  # Single feature vector
                inputs = inputs.unsqueeze(0)
                
            if len(outputs.shape) == 3:  # [batch, seq, features]
                outputs = outputs.reshape(-1, outputs.shape[-1])
            elif len(outputs.shape) == 2:  # [seq, features]
                pass  # Already in the right shape
            else:  # Single feature vector
                outputs = outputs.unsqueeze(0)
            
            # Convert to numpy
            inputs_np = inputs.numpy()
            outputs_np = outputs.numpy()
            
            all_inputs.append(inputs_np)
            all_outputs.append(outputs_np)
        
        # Concatenate and fit scalers
        all_inputs = np.vstack(all_inputs)
        all_outputs = np.vstack(all_outputs)
        
        # Fit the scalers
        self.input_scaler.fit(all_inputs)
        self.output_scaler.fit(all_outputs)
        
        print(f"Feature scaling set up based on {all_inputs.shape[0]} input samples")
        print(f"Input feature mean: {self.input_scaler.mean_}")
        print(f"Input feature scale: {self.input_scaler.scale_}")
        
    def build_models(self, input_dim, output_dim=2):
        """
        Build the models.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features (default: 2 for x, y coordinates)
        """
        # Calculate dimensions for the transformer model
        pos_dim = input_dim  # Positional features dimension
        
        # For the visual/terrain features dimension
        if self.config['use_terrain'] and self.terrain_data is not None and self.elevation_data is not None:
            vis_dim = self.config['terrain_feature_dim']
        else:
            vis_dim = input_dim  # Default to same as pos_dim if no terrain
        
        # Transformer model for sequence prediction
        self.transformer_model = TargetTransformerPredictor(
            pos_dim=pos_dim,
            vis_dim=vis_dim,
            hidden_dim=self.config['hidden_dim'],
            output_dim=output_dim,
            n_layers=self.config['n_layers'],
            n_heads=self.config['n_heads'],
            dropout=self.config['dropout']
        )
        
        # Terrain feature extractor
        if self.config['use_terrain'] and self.terrain_data is not None and self.elevation_data is not None:
            self.terrain_model = TerrainFeatureExtractor(
                input_channels=2,  # Terrain + elevation
                output_dim=self.config['terrain_feature_dim']
            )
        
        # Move models to device
        self.transformer_model.to(self.config['device'])
        if self.terrain_model is not None:
            self.terrain_model.to(self.config['device'])
    
    def prepare_data(self, target_data, blue_force_data=None, train_ratio=0.8):
        """
        Prepare data for training and validation.
        
        Args:
            target_data: DataFrame with target observations
            blue_force_data: DataFrame with blue force locations
            train_ratio: Ratio of data to use for training
            
        Returns:
            Training and validation data loaders
        """
        # Ensure datetime is properly parsed
        if 'datetime' in target_data.columns:
            if target_data['datetime'].dtype == object:
                target_data['datetime'] = pd.to_datetime(target_data['datetime'])
        
        # Create dataset
        dataset = TargetTrajectoryDataset(
            target_data=target_data,
            terrain_data=self.terrain_data,
            elevation_data=self.elevation_data,
            blue_force_data=blue_force_data,
            sequence_length=self.config['sequence_length'],
            prediction_horizon=self.config['prediction_horizon'],
            stride=1
        )

        # Split into training and validation sets
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        # Set up feature normalization
        self.setup_feature_scaling(train_dataset)

        # Determine input dimension by getting a sample from the dataset
        sample = dataset[0]
        
        # Check the shape of the input tensor
        input_tensor = sample['input']
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Number of features: {input_tensor.shape[-1]}")

        # Determine the feature dimension based on shape
        if len(input_tensor.shape) == 1:
            # 1D tensor: Just features
            input_dim = input_tensor.shape[0]
        elif len(input_tensor.shape) == 2:
            if input_tensor.shape[0] == 1:
                # Shape [1, features] - missing sequence dimension
                input_dim = input_tensor.shape[1]
            else:
                # Shape [sequence_length, features]
                input_dim = input_tensor.shape[1]
        else:
            # 3D tensor: [batch, sequence_length, features]
            input_dim = input_tensor.shape[2]
        
        print(f"Using input dimension: {input_dim}")
        
        # Build models with the correct input dimension
        self.build_models(input_dim)

        return train_loader, val_loader
    
    def normalize_features(self, inputs, outputs=None):
        """
        Normalize features using fitted scalers.
        
        Args:
            inputs: Input features
            outputs: Output features (optional)
            
        Returns:
            Normalized inputs and outputs
        """
        # Convert to numpy if tensors
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.cpu().numpy()
            input_shape = inputs.shape
            
            # Reshape for scaling
            inputs_flat = inputs_np.reshape(-1, inputs_np.shape[-1])
            
            # Scale
            inputs_scaled = self.input_scaler.transform(inputs_flat)
            
            # Reshape back
            inputs_scaled = inputs_scaled.reshape(input_shape)
            
            # Convert back to tensor
            inputs_scaled = torch.tensor(inputs_scaled, dtype=inputs.dtype, device=inputs.device)
        else:
            inputs_scaled = self.input_scaler.transform(inputs)
        
        # Scale outputs if provided
        outputs_scaled = None
        if outputs is not None:
            if isinstance(outputs, torch.Tensor):
                outputs_np = outputs.cpu().numpy()
                output_shape = outputs.shape
                
                # Reshape for scaling
                outputs_flat = outputs_np.reshape(-1, outputs_np.shape[-1])
                
                # Scale
                outputs_scaled = self.output_scaler.transform(outputs_flat)
                
                # Reshape back
                outputs_scaled = outputs_scaled.reshape(output_shape)
                
                # Convert back to tensor
                outputs_scaled = torch.tensor(outputs_scaled, dtype=outputs.dtype, device=outputs.device)
            else:
                outputs_scaled = self.output_scaler.transform(outputs)
        
        return inputs_scaled, outputs_scaled
    
    def denormalize_features(self, outputs):
        """
        Denormalize features using fitted scalers.
        
        Args:
            outputs: Output features
            
        Returns:
            Denormalized outputs
        """
        # Convert to numpy if tensor
        if isinstance(outputs, torch.Tensor):
            outputs_np = outputs.cpu().numpy()
            output_shape = outputs.shape
            
            # Reshape for scaling
            outputs_flat = outputs_np.reshape(-1, outputs_np.shape[-1])
            
            # Inverse transform
            outputs_denorm = self.output_scaler.inverse_transform(outputs_flat)
            
            # Reshape back
            outputs_denorm = outputs_denorm.reshape(output_shape)
            
            # Convert back to tensor
            outputs_denorm = torch.tensor(outputs_denorm, dtype=outputs.dtype, device=outputs.device)
        else:
            outputs_denorm = self.output_scaler.inverse_transform(outputs)
        
        return outputs_denorm
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for (default: config value)
            
        Returns:
            Dictionary of training history
        """
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        # Optimizer
        optimizer = optim.Adam(
            list(self.transformer_model.parameters()) + 
            (list(self.terrain_model.parameters()) if self.terrain_model is not None else []),
            lr=self.config['learning_rate']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_nll_loss': [],
            'val_nll_loss': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
        # Training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            # Training
            self.transformer_model.train()
            if self.terrain_model is not None:
                self.terrain_model.train()
            
            train_loss = 0.0
            train_nll_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch in train_pbar:
                # Get batch
                inputs = batch['input'].to(self.config['device'])
                targets = batch['target'].to(self.config['device'])
                
                # Normalize
                inputs_norm, targets_norm = self.normalize_features(inputs, targets)
                
                # Process terrain if available
                terrain_features = None
                if self.terrain_model is not None and batch['terrain'] is not None:
                    terrain = batch['terrain'].to(self.config['device'])
                    terrain_features = self.terrain_model(terrain)
                    
                    # Create dummy visual sequence if needed 
                    # (1, sequence length, visual feature dimension)
                    vis_seq = terrain_features.unsqueeze(1).expand(-1, 1, -1)
                else:
                    # Create dummy visual features as default
                    vis_seq = torch.zeros(
                        (inputs_norm.shape[0], 1, inputs_norm.shape[2]), 
                        device=self.config['device']
                    )
                
                # Forward pass
                optimizer.zero_grad()
                
                # Forward pass through transformer
                pred_mean, pred_logvar = self.transformer_model(inputs_norm, vis_seq)
                
                # Extract predictions for the forecast horizon
                pred_mean = pred_mean[:, -self.config['prediction_horizon']:, :]
                pred_logvar = pred_logvar[:, -self.config['prediction_horizon']:, :]
                
                # Calculate MSE loss
                mse_loss = self.mse_loss(pred_mean, targets_norm)
                
                # Calculate negative log likelihood loss
                nll_loss = self.gaussian_nll_loss(pred_mean, pred_logvar, targets_norm)
                
                # Total loss
                loss = mse_loss + 0.1 * nll_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                train_loss += loss.item()
                train_nll_loss += nll_loss.item()
                train_pbar.set_postfix({'loss': f"{train_loss/(train_pbar.n+1):.4f}"})
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            train_nll_loss /= len(train_loader)
            
            # Validation
            self.transformer_model.eval()
            if self.terrain_model is not None:
                self.terrain_model.eval()
            
            val_loss = 0.0
            val_nll_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for batch in val_pbar:
                    # Get batch
                    inputs = batch['input'].to(self.config['device'])
                    targets = batch['target'].to(self.config['device'])
                    
                    # Normalize
                    inputs_norm, targets_norm = self.normalize_features(inputs, targets)
                    
                    # Process terrain if available
                    terrain_features = None
                    if self.terrain_model is not None and batch['terrain'] is not None:
                        terrain = batch['terrain'].to(self.config['device'])
                        terrain_features = self.terrain_model(terrain)
                        
                        # Create dummy visual sequence
                        # (1, sequence length, visual feature dimension)
                        vis_seq = terrain_features.unsqueeze(1).expand(-1, 1, -1)
                    else:
                        # Create dummy visual features as default
                        vis_seq = torch.zeros(
                            (inputs_norm.shape[0], 1, inputs_norm.shape[2]), 
                            device=self.config['device']
                        )
                    
                    # Forward pass through transformer
                    pred_mean, pred_logvar = self.transformer_model(inputs_norm, vis_seq)
                    
                    # Extract predictions for the forecast horizon
                    pred_mean = pred_mean[:, -self.config['prediction_horizon']:, :]
                    pred_logvar = pred_logvar[:, -self.config['prediction_horizon']:, :]
                    
                    # Calculate MSE loss
                    mse_loss = self.mse_loss(pred_mean, targets_norm)
                    
                    # Calculate negative log likelihood loss
                    nll_loss = self.gaussian_nll_loss(pred_mean, pred_logvar, targets_norm)
                    
                    # Total loss
                    loss = mse_loss + 0.1 * nll_loss
                    
                    # Update progress bar
                    val_loss += loss.item()
                    val_nll_loss += nll_loss.item()
                    val_pbar.set_postfix({'loss': f"{val_loss/(val_pbar.n+1):.4f}"})
            
            # Calculate average validation loss
            val_loss /= len(val_loader)
            val_nll_loss /= len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_nll_loss'].append(train_nll_loss)
            history['val_nll_loss'].append(val_nll_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Train NLL: {train_nll_loss:.4f}, "
                  f"Val NLL: {val_nll_loss:.4f}")
            
            # Save best model
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                self.save_model('best_target_model.pt')
        
        # Calculate total training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model
        self.load_model('best_target_model.pt')
        
        return history
    
    def gaussian_nll_loss(self, mean, logvar, target):
        """
        Calculate negative log likelihood loss for Gaussian distribution.
        
        Args:
            mean: Predicted mean
            logvar: Predicted log variance
            target: Target values
            
        Returns:
            NLL loss
        """
        # Calculate precision (inverse variance)
        precision = torch.exp(-logvar)
        
        # Calculate squared error
        squared_error = (target - mean) ** 2
        
        # Calculate NLL loss
        nll_loss = 0.5 * (logvar + squared_error * precision)
        
        return torch.mean(nll_loss)

    def predict(self, input_sequence, terrain_patch=None, num_samples=100):
        """
        Predict future positions with confidence intervals.
        
        Args:
            input_sequence: Input sequence tensor [sequence_length, features]
            terrain_patch: Terrain patch tensor (optional)
            num_samples: Number of samples to draw for confidence intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Set model to evaluation mode
        self.transformer_model.eval()
        if self.terrain_model is not None:
            self.terrain_model.eval()
        
        # Add batch dimension if necessary
        if len(input_sequence.shape) == 2:
            input_sequence = input_sequence.unsqueeze(0)
        
        if terrain_patch is not None and len(terrain_patch.shape) == 3:
            terrain_patch = terrain_patch.unsqueeze(0)
        
        # Move to device
        input_sequence = input_sequence.to(self.config['device'])
        if terrain_patch is not None:
            terrain_patch = terrain_patch.to(self.config['device'])
        
        # Normalize input
        input_norm, _ = self.normalize_features(input_sequence)
        
        # Process terrain if available
        terrain_features = None
        if self.terrain_model is not None and terrain_patch is not None:
            terrain_features = self.terrain_model(terrain_patch)
            
            # Create visual sequence
            vis_seq = terrain_features.unsqueeze(1).expand(-1, 1, -1)
        else:
            # Create dummy visual features with the correct dimension from the config
            vis_dim = self.config.get('terrain_feature_dim', 32)  # Default to 32 if not found
            vis_seq = torch.zeros(
                (input_norm.shape[0], 1, vis_dim), 
                device=self.config['device']
            )
        
        # Predict with the model
        with torch.no_grad():
            pred_mean, pred_logvar = self.transformer_model(input_norm, vis_seq)
            
            # Extract predictions for the forecast horizon
            pred_mean = pred_mean[:, -self.config['prediction_horizon']:, :]
            pred_logvar = pred_logvar[:, -self.config['prediction_horizon']:, :]
            
            # Convert log variance to standard deviation
            pred_std = torch.exp(0.5 * pred_logvar)
            
            # Generate samples for confidence intervals
            samples = []
            for _ in range(num_samples):
                # Sample from normal distribution
                noise = torch.randn_like(pred_mean)
                sample = pred_mean + noise * pred_std
                samples.append(sample)
            
            # Stack samples
            samples = torch.stack(samples, dim=0)
            
            # Calculate confidence intervals
            lower_ci = torch.quantile(samples, 0.025, dim=0)
            upper_ci = torch.quantile(samples, 0.975, dim=0)
            
            # Denormalize predictions
            pred_mean_denorm = self.denormalize_features(pred_mean)
            lower_ci_denorm = self.denormalize_features(lower_ci)
            upper_ci_denorm = self.denormalize_features(upper_ci)
            
            # Convert to numpy
            pred_mean_np = pred_mean_denorm.cpu().numpy()
            lower_ci_np = lower_ci_denorm.cpu().numpy()
            upper_ci_np = upper_ci_denorm.cpu().numpy()
            
            # Remove batch dimension if only one sample
            if pred_mean_np.shape[0] == 1:
                pred_mean_np = pred_mean_np[0]
                lower_ci_np = lower_ci_np[0]
                upper_ci_np = upper_ci_np[0]
        
        return {
            'mean': pred_mean_np,
            'lower_ci': lower_ci_np,
            'upper_ci': upper_ci_np
        }
    
    def save_model(self, filename):
        """
        Save the model.
        
        Args:
            filename: Filename to save the model to
        """
        save_dict = {
            'transformer_state': self.transformer_model.state_dict(),
            'config': self.config,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler
        }
        
        if self.terrain_model is not None:
            save_dict['terrain_state'] = self.terrain_model.state_dict()
        
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load the model.
        
        Args:
            filename: Filename to load the model from
        """
        if not os.path.exists(filename):
            print(f"Model file {filename} not found")
            return False
        
        try:
            # Load checkpoint
            checkpoint = torch.load(filename, map_location=self.config['device'], weights_only=False)
            
            # Update config
            self.config.update(checkpoint['config'])
            
            # Load scalers
            self.input_scaler = checkpoint['input_scaler']
            self.output_scaler = checkpoint['output_scaler']
            
            # Initialize models if not already initialized
            if self.transformer_model is None:
                # Determine input dimension
                input_dim = next(iter(checkpoint['transformer_state'].values())).shape[1]
                
                self.build_models(input_dim)
            
            # Load model weights
            self.transformer_model.load_state_dict(checkpoint['transformer_state'])
            
            if 'terrain_state' in checkpoint and self.terrain_model is not None:
                self.terrain_model.load_state_dict(checkpoint['terrain_state'])
            
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.transformer_model.eval()
        if self.terrain_model is not None:
            self.terrain_model.eval()
        
        all_targets = []
        all_predictions = []
        all_lower_ci = []
        all_upper_ci = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Get batch
                inputs = batch['input'].to(self.config['device'])
                targets = batch['target'].to(self.config['device'])
                
                # Process terrain if available
                terrain_patch = None
                if 'terrain' in batch and batch['terrain'] is not None:
                    terrain_patch = batch['terrain'].to(self.config['device'])
                
                # Make predictions
                predictions = self.predict(inputs, terrain_patch)
                
                # Ensure consistent shape for concatenation
                target_np = targets.cpu().numpy()
                
                # Collect results - ensure they have the same number of dimensions
                all_targets.append(target_np)
                
                # Ensure prediction arrays have same shape as target_np
                if len(predictions['mean'].shape) != len(target_np.shape):
                    # Add batch dimension if missing
                    mean = predictions['mean'][np.newaxis, ...]
                    lower_ci = predictions['lower_ci'][np.newaxis, ...]
                    upper_ci = predictions['upper_ci'][np.newaxis, ...]
                else:
                    mean = predictions['mean']
                    lower_ci = predictions['lower_ci']
                    upper_ci = predictions['upper_ci']
                
                all_predictions.append(mean)
                all_lower_ci.append(lower_ci)
                all_upper_ci.append(upper_ci)
        
        # Concatenate results - now with consistent shapes
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_lower_ci = np.concatenate(all_lower_ci, axis=0)
        all_upper_ci = np.concatenate(all_upper_ci, axis=0)
        
        # Calculate metrics
        mse = np.mean((all_targets - all_predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_targets - all_predictions))
        
        # Calculate coverage of confidence intervals
        in_ci = np.logical_and(
            all_targets >= all_lower_ci,
            all_targets <= all_upper_ci
        )
        coverage = np.mean(in_ci)
        
        # Calculate average CI width
        ci_width = np.mean(all_upper_ci - all_lower_ci)
        
        # Calculate error by time step
        errors_by_step = []
        for t in range(all_targets.shape[1]):
            step_mse = np.mean((all_targets[:, t, :] - all_predictions[:, t, :]) ** 2)
            step_rmse = np.sqrt(step_mse)
            errors_by_step.append(step_rmse)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'ci_width': ci_width,
            'errors_by_step': errors_by_step
        }
    
    def visualize_predictions(self, input_sequence, true_future=None, terrain_patch=None, 
                             title="Target Movement Prediction", save_path=None):
        """
        Visualize predictions with confidence intervals.
        
        Args:
            input_sequence: Input sequence tensor
            true_future: True future positions (optional)
            terrain_patch: Terrain patch tensor (optional)
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Make predictions
        predictions = self.predict(input_sequence, terrain_patch)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot input sequence
        input_np = input_sequence.cpu().numpy()
        if len(input_np.shape) == 3:
            input_np = input_np[0]  # Remove batch dimension
            
        ax.plot(input_np[:, 0], input_np[:, 1], 'b-', label='Past Trajectory')
        ax.scatter(input_np[-1, 0], input_np[-1, 1], c='blue', s=100, marker='o', label='Current Position')
        
        # Plot predictions
        mean = predictions['mean']
        lower = predictions['lower_ci']
        upper = predictions['upper_ci']
        
        if len(mean.shape) == 3:
            mean = mean[0]  # Remove batch dimension
            lower = lower[0]
            upper = upper[0]
            
        ax.plot(mean[:, 0], mean[:, 1], 'r-', label='Predicted Trajectory')
        ax.scatter(mean[-1, 0], mean[-1, 1], c='red', s=100, marker='x', label='Final Predicted Position')
        
        # Plot confidence regions
        for i in range(len(mean)):
            ellipse = plt.matplotlib.patches.Ellipse(
                (mean[i, 0], mean[i, 1]),
                width=upper[i, 0] - lower[i, 0],
                height=upper[i, 1] - lower[i, 1],
                color='red', alpha=0.2
            )
            ax.add_patch(ellipse)
        
        # Plot true future if provided
        if true_future is not None:
            true_np = true_future.cpu().numpy()
            if len(true_np.shape) == 3:
                true_np = true_np[0]  # Remove batch dimension
                
            ax.plot(true_np[:, 0], true_np[:, 1], 'g-', label='True Future Trajectory')
            ax.scatter(true_np[-1, 0], true_np[-1, 1], c='green', s=100, marker='*', label='True Final Position')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        ax.grid(True)
        
        # Add annotations for confidence levels
        ax.text(0.05, 0.05, "95% Confidence Region", transform=ax.transAxes, 
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Save plot if path provided
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_training_history(self, history, save_path=None):
        """
        Visualize training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.axvline(x=history['best_epoch'] + 1, color='g', linestyle='--', label='Best Model')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot NLL losses
        ax2.plot(epochs, history['train_nll_loss'], 'b-', label='Training NLL Loss')
        ax2.plot(epochs, history['val_nll_loss'], 'r-', label='Validation NLL Loss')
        ax2.axvline(x=history['best_epoch'] + 1, color='g', linestyle='--', label='Best Model')
        ax2.set_title('Training and Validation NLL Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('NLL Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def predict_out_of_view(self, target_data, target_id, last_seen_time, prediction_duration,
                        terrain_patch=None, blue_force_data=None):
        """Predict the target's movement after it goes out of view."""
        # Convert last_seen_time to datetime if it's a string
        if isinstance(last_seen_time, str):
            last_seen_time = pd.to_datetime(last_seen_time)
        
        # Filter data for this target
        target_df = target_data[target_data['target_id'] == target_id]
        target_df = target_df.sort_values('datetime')
        
        # Convert datetime column if necessary
        if target_df['datetime'].dtype == object:
            target_df['datetime'] = pd.to_datetime(target_df['datetime'])
        
        # Get data up to last_seen_time
        history_df = target_df[target_df['datetime'] <= last_seen_time]
        
        # Check if we have enough history
        if len(history_df) < self.config['sequence_length']:
            print(f"Warning: Not enough history for target {target_id}. Need at least {self.config['sequence_length']} points.")
            return None
        
        # Prepare input sequence - take the last sequence_length points
        history_df = history_df.iloc[-self.config['sequence_length']:]
        
        # Extract features
        input_sequence = []
        
        for _, row in history_df.iterrows():
            # Base features: x, y, altitude
            features = [row['longitude'], row['latitude']]
            
            if 'altitude_m' in row:
                features.append(row['altitude_m'])
            else:
                features.append(0)  # Default altitude
            
            # Add target class one-hot if available
            if 'target_class' in row:
                classes = target_data['target_class'].unique()
                for cls in classes:
                    features.append(1.0 if row['target_class'] == cls else 0.0)
            
            # Extract rich temporal features from timestamp
            dt = row['datetime']
            
            # 1. Cyclical encoding of hour (captures day/night patterns)
            hour_sin = np.sin(2 * np.pi * dt.hour / 24)
            hour_cos = np.cos(2 * np.pi * dt.hour / 24)
            
            # 2. Cyclical encoding of minutes
            minute_sin = np.sin(2 * np.pi * dt.minute / 60)
            minute_cos = np.cos(2 * np.pi * dt.minute / 60)
            
            # 3. Day of week (for weekly patterns)
            day_sin = np.sin(2 * np.pi * dt.dayofweek / 7)
            day_cos = np.cos(2 * np.pi * dt.dayofweek / 7)
            
            # 4. Time of day indicators
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
        
        # Print the shape for debugging
        print(f"Input sequence shape for prediction: {np.array(input_sequence).shape}")
        
        # Convert to tensor
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        
        # Make predictions
        predictions = self.predict(input_sequence, terrain_patch)
        
        # Calculate time points for predictions
        if isinstance(prediction_duration, (int, float)):
            # Duration in seconds
            time_delta = timedelta(seconds=prediction_duration)
        else:
            # Assume it's already a timedelta
            time_delta = prediction_duration
        
        # Calculate total prediction time and divide into steps
        step_delta = time_delta / self.config['prediction_horizon']
        
        time_points = [last_seen_time + step_delta * (i + 1) for i in range(self.config['prediction_horizon'])]
        
        # Add time information to predictions
        predictions['time_points'] = time_points
        
        return predictions

# Function to load and process data
def load_and_process_data(target_csv="data/red_sightings.csv", 
                         blue_csv="data/blue_locations.csv",
                         terrain_path="adapted_data/terrain_map.npy",
                         elevation_path="adapted_data/elevation_map.npy"):
    """
    Load and process data files.
    
    Args:
        target_csv: Path to red target sightings CSV
        blue_csv: Path to blue force locations CSV
        terrain_path: Path to terrain map NPY file
        elevation_path: Path to elevation map NPY file
        
    Returns:
        Dictionary with processed data
    """
    # Load target data
    target_df = pd.read_csv(target_csv)
    
    # Convert datetime column
    if 'datetime' in target_df.columns:
        target_df['datetime'] = pd.to_datetime(target_df['datetime'])
    
    # Load blue force data
    blue_df = pd.read_csv(blue_csv)
    
    # Load terrain and elevation if available
    terrain_data = None
    elevation_data = None
    
    if os.path.exists(terrain_path):
        terrain_data = np.load(terrain_path)
        print(f"Loaded terrain data with shape {terrain_data.shape}")
    
    if os.path.exists(elevation_path):
        elevation_data = np.load(elevation_path)
        print(f"Loaded elevation data with shape {elevation_data.shape}")
    
    # Return all data
    return {
        'target_data': target_df,
        'blue_force_data': blue_df,
        'terrain_data': terrain_data,
        'elevation_data': elevation_data
    }

# Sample usage function
def run_training_pipeline(
    target_csv="data/red_sightings.csv",
    blue_csv="data/blue_locations.csv",
    terrain_path="adapted_data/terrain_map.npy",
    elevation_path="adapted_data/elevation_map.npy",
    output_dir="models/target_prediction",
    config=None
):
    """
    Run the full training pipeline for target movement prediction.
    
    Args:
        target_csv: Path to target observations CSV
        blue_csv: Path to blue force locations CSV
        terrain_path: Path to terrain map file
        elevation_path: Path to elevation map file
        output_dir: Directory to save models and plots
        config: Model configuration dictionary (optional)
        
    Returns:
        Trained predictor object
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_and_process_data(
        target_csv=target_csv,
        blue_csv=blue_csv,
        terrain_path=terrain_path,
        elevation_path=elevation_path
    )
    
    target_df = data['target_data']
    blue_df = data['blue_force_data']
    
    # Initialize predictor
    predictor = TargetMovementPredictor(
        config=config,
        terrain_data_path=terrain_path,
        elevation_data_path=elevation_path
    )
    
    # Prepare data
    train_loader, val_loader = predictor.prepare_data(target_df, blue_df)
    
    # Train model
    history = predictor.train(train_loader, val_loader)
    
    # Save training history plot
    predictor.visualize_training_history(
        history,
        save_path=os.path.join(output_dir, "training_history.png")
    )
    
    # Evaluate on validation set
    metrics = predictor.evaluate(val_loader)
    print("\nValidation Metrics:")
    for key, value in metrics.items():
        if key != 'errors_by_step':
            print(f"  {key}: {value}")
    
    # Plot errors by prediction step
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics['errors_by_step']) + 1), metrics['errors_by_step'], 'o-')
    plt.title('RMSE by Prediction Step')
    plt.xlabel('Prediction Step')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "errors_by_step.png"), dpi=300)
    
    # Visualize sample predictions
    for i, batch in enumerate(val_loader):
        if i >= 3:  # Show 3 examples
            break
            
        inputs = batch['input']
        targets = batch['target']
        terrain = batch['terrain'] if 'terrain' in batch else None
        
        predictor.visualize_predictions(
            inputs, targets, terrain,
            title=f"Target {batch['target_id'][0]} Predictions",
            save_path=os.path.join(output_dir, f"prediction_sample_{i}.png")
        )
    
    # Save model
    predictor.save_model(os.path.join(output_dir, "target_predictor_model.pt"))
    
    print(f"\nTraining pipeline completed. Model and visualizations saved to {output_dir}")
    
    return predictor

def visualize_out_of_view_prediction(
    predictor, 
    target_data, 
    target_id, 
    last_seen_time,
    prediction_duration,
    terrain_data=None,
    blue_force_data=None,
    output_path=None
):
    """
    Visualize predictions for a target after it goes out of view.
    
    Args:
        predictor: Trained TargetMovementPredictor model
        target_data: DataFrame with target observations
        target_id: ID of the target to predict
        last_seen_time: Time when target was last seen
        prediction_duration: How long to predict ahead (seconds or timedelta)
        terrain_data: Terrain map (optional)
        blue_force_data: Blue force locations (optional)
        output_path: Path to save visualization (optional)
        
    Returns:
        Matplotlib figure
    """
    # Make prediction
    prediction = predictor.predict_out_of_view(
        target_data, 
        target_id, 
        last_seen_time,
        prediction_duration
    )
    
    if prediction is None:
        print(f"Could not generate prediction for target {target_id}")
        return None
    
    # Get history for this target
    target_df = target_data[target_data['target_id'] == target_id].copy()
    if 'datetime' in target_df.columns and target_df['datetime'].dtype == object:
        target_df['datetime'] = pd.to_datetime(target_df['datetime'])
    
    # Filter to get only data before last_seen_time
    history = target_df[target_df['datetime'] <= last_seen_time]
    
    # Check if we have any future data to validate against
    if isinstance(last_seen_time, str):
        last_seen_time = pd.to_datetime(last_seen_time)
        
    # Get the end time of prediction
    if isinstance(prediction_duration, (int, float)):
        end_time = last_seen_time + timedelta(seconds=prediction_duration)
    else:
        end_time = last_seen_time + prediction_duration
    
    # Get validation data if available
    future = target_df[(target_df['datetime'] > last_seen_time) & (target_df['datetime'] <= end_time)]
    has_validation = len(future) > 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot history trajectory
    ax.plot(history['longitude'], history['latitude'], 'b-', label='Past Trajectory')
    ax.scatter(history['longitude'].iloc[-1], history['latitude'].iloc[-1], 
              c='blue', s=100, marker='o', label='Last Seen Position')
    
    # Plot predicted trajectory
    mean_traj = prediction['mean']
    lower_ci = prediction['lower_ci']
    upper_ci = prediction['upper_ci']
    time_points = prediction['time_points']
    
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'r-', label='Predicted Trajectory')
    ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], 
              c='red', s=100, marker='x', label='Predicted Final Position')
    
    # Plot confidence intervals
    for i in range(len(mean_traj)):
        ellipse = plt.matplotlib.patches.Ellipse(
            (mean_traj[i, 0], mean_traj[i, 1]),
            width=upper_ci[i, 0] - lower_ci[i, 0],
            height=upper_ci[i, 1] - lower_ci[i, 1],
            color='red', alpha=0.2
        )
        ax.add_patch(ellipse)
    
    # Plot actual future trajectory if available
    if has_validation:
        ax.plot(future['longitude'], future['latitude'], 'g-', label='Actual Future Trajectory')
        ax.scatter(future['longitude'].iloc[-1], future['latitude'].iloc[-1],
                 c='green', s=100, marker='*', label='Actual Final Position')
    
    # Plot blue force positions if available
    if blue_force_data is not None:
        ax.scatter(blue_force_data['longitude'], blue_force_data['latitude'],
                 c='blue', s=100, marker='^', label='Blue Forces')
    
    # Add time labels to predicted points
    for i, (x, y, t) in enumerate(zip(mean_traj[:, 0], mean_traj[:, 1], time_points)):
        # Only label some points to avoid clutter
        if i % 2 == 0 or i == len(mean_traj) - 1:
            ax.annotate(t.strftime('%H:%M:%S'), 
                       (x, y), 
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='center')
    
    # Set title and labels
    target_class = target_df['target_class'].iloc[0] if 'target_class' in target_df.columns else 'Unknown'
    ax.set_title(f'Target {target_id} ({target_class}) Prediction After Going Out of View')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)
    
    # Add information box
    info_text = (
        f"Target: {target_id} ({target_class})\n"
        f"Last seen: {last_seen_time}\n"
        f"Prediction duration: {prediction_duration} seconds\n"
        f"95% confidence interval shown"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    # Save figure if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_prediction_animation(predictor, output_filename='target_prediction_animation.mp4'):
    """
    Create animation of target predictions with correctly mapped terrain and blue forces.
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
    
    # Add terrain background with proper extent and orientation
    # The critical fix: correctly map the terrain to lat/lon with origin='upper'
    # Note that we reverse lat_min and lat_max because imshow uses origin='upper'
    terrain_img = ax.imshow(terrain_map, cmap=terrain_cmap, norm=norm, alpha=0.5, 
                      extent=[lon_min, lon_max, lat_max, lat_min],
                      aspect='auto', zorder=0, origin='upper')
    
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
        current_time_np = selected_frames[frame_idx]
        # Convert numpy.datetime64 to pandas Timestamp for strftime
        current_time = pd.Timestamp(current_time_np)
        
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
        for _, row in current_red.iterrows():
            target_id = row['target_id']
            target_class = row['target_class']
            
            # Choose color based on target class
            if target_class == 'tank':
                color = 'darkred'
            elif target_class == 'armoured personnel carrier':
                color = 'orangered'
            elif target_class == 'light vehicle':
                color = 'coral'
            else:
                color = 'red'
            
            # Get history data for this target
            target_history = red_df[(red_df['target_id'] == target_id) & 
                                  (red_df['datetime'] <= current_time)].sort_values('datetime')
            
            # Update target trail
            target_points = target_history[['longitude', 'latitude']].values[-10:]
            if len(target_points) >= 2:
                line, = ax.plot(target_points[:, 0], target_points[:, 1], '-', 
                               color=color, alpha=0.4, linewidth=1.5, zorder=5)
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
                line, = ax.plot(mean_traj[:, 0], mean_traj[:, 1], '--', 
                               color=color, linewidth=2, alpha=0.8, zorder=6)
                prediction_lines[target_id] = line
                
                # Plot confidence ellipses
                confidence_ellipses[target_id] = []
                for i in range(len(mean_traj)):
                    ellipse = plt.matplotlib.patches.Ellipse(
                        (mean_traj[i, 0], mean_traj[i, 1]),
                        width=upper_ci[i, 0] - lower_ci[i, 0],
                        height=upper_ci[i, 1] - lower_ci[i, 1],
                        color=color, alpha=0.2, zorder=4
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

def visualize_target_prediction_with_terrain(
    predictor, 
    target_data, 
    target_id, 
    timestamp,
    prediction_duration=300,
    terrain_data=None,
    blue_force_data=None,
    output_path=None,
    show_confidence=True
):
    """
    Enhanced visualization function that properly displays target prediction with terrain data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import pandas as pd
    import numpy as np
    from matplotlib.patches import Patch
    
    # Convert timestamp to datetime if it's a string
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    
    # Make prediction
    prediction = predictor.predict_out_of_view(
        target_data, 
        target_id, 
        timestamp,
        prediction_duration
    )
    
    if prediction is None:
        print(f"Could not generate prediction for target {target_id}")
        return None
    
    # Create figure with two subplots - one for prediction, one for terrain analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Get coordinate bounds for the prediction
    mean_traj = prediction['mean']
    lon_min, lon_max = mean_traj[:, 0].min() - 0.05, mean_traj[:, 0].max() + 0.05
    lat_min, lat_max = mean_traj[:, 1].min() - 0.05, mean_traj[:, 1].max() + 0.05
    
    # 1. Plot prediction on the first subplot
    # Filter data for this target
    target_df = target_data[target_data['target_id'] == target_id].copy()
    if 'datetime' in target_df.columns and target_df['datetime'].dtype == object:
        target_df['datetime'] = pd.to_datetime(target_df['datetime'])
    
    # Get history leading up to timestamp
    history = target_df[target_df['datetime'] <= timestamp]
    
    # Get target class for styling
    target_class = target_df['target_class'].iloc[0] if 'target_class' in target_df.columns else 'Unknown'
    
    # Choose color based on target class
    if target_class == 'tank':
        color = 'darkred'
    elif target_class == 'armoured personnel carrier':
        color = 'orangered'
    elif target_class == 'light vehicle':
        color = 'coral'
    else:
        color = 'red'
    
    # Plot history trajectory
    ax1.plot(history['longitude'], history['latitude'], 'b-', label='Past Trajectory')
    ax1.scatter(history['longitude'].iloc[-1], history['latitude'].iloc[-1], 
              c='blue', s=100, marker='o', label='Last Seen Position')
    
    # Plot predicted trajectory
    lower_ci = prediction['lower_ci']
    upper_ci = prediction['upper_ci']
    time_points = prediction['time_points']
    
    ax1.plot(mean_traj[:, 0], mean_traj[:, 1], '-', color=color, linewidth=2.5,
             label=f'Predicted {target_class} Trajectory')
    ax1.scatter(mean_traj[-1, 0], mean_traj[-1, 1], 
              c=color, s=100, marker='x', label='Predicted Final Position')
    
    # Plot confidence intervals
    if show_confidence:
        for i in range(len(mean_traj)):
            ellipse = plt.matplotlib.patches.Ellipse(
                (mean_traj[i, 0], mean_traj[i, 1]),
                width=upper_ci[i, 0] - lower_ci[i, 0],
                height=upper_ci[i, 1] - lower_ci[i, 1],
                color=color, alpha=0.2
            )
            ax1.add_patch(ellipse)
    
    # Check for actual future data to validate prediction
    future = target_df[(target_df['datetime'] > timestamp) & 
                      (target_df['datetime'] <= time_points[-1])]
    
    if len(future) > 0:
        ax1.plot(future['longitude'], future['latitude'], 'g-', 
                label='Actual Future Trajectory')
        ax1.scatter(future['longitude'].iloc[-1], future['latitude'].iloc[-1],
                  c='green', s=100, marker='*', label='Actual Final Position')
    
    # Add time labels to predicted points
    for i, (x, y, t) in enumerate(zip(mean_traj[:, 0], mean_traj[:, 1], time_points)):
        if i % 2 == 0 or i == len(mean_traj) - 1:
            ax1.annotate(t.strftime('%H:%M:%S'), 
                      (x, y), 
                      textcoords="offset points", 
                      xytext=(0, 10), 
                      ha='center')
    
    # Set title and labels
    ax1.set_title(f'Target {target_id} ({target_class}) Prediction')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(lon_min, lon_max)
    ax1.set_ylim(lat_min, lat_max)
    
    # 2. Plot terrain analysis on the second subplot
    if terrain_data is not None:
        # Define colors for each land use category from the LandUseCategory enum
        land_use_colors = [
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

        terrain_cmap = ListedColormap(land_use_colors)

        # Add normalization for the correct range (0-20)
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(0, 20)
        # Get the full extent of the target data for the terrain map
        full_lon_min, full_lon_max = target_data['longitude'].min(), target_data['longitude'].max()
        full_lat_min, full_lat_max = target_data['latitude'].min(), target_data['latitude'].max()
        
        # Plot terrain - note the use of origin='upper' and reversed lat_min/lat_max
        im = ax2.imshow(terrain_data, cmap=terrain_cmap, 
                      extent=[full_lon_min, full_lon_max, full_lat_max, full_lat_min],
                      aspect='auto', origin='upper', alpha=0.7)
        
        # Plot the trajectory on terrain
        ax2.plot(history['longitude'], history['latitude'], 'b-', linewidth=2)
        ax2.plot(mean_traj[:, 0], mean_traj[:, 1], '--', color=color, linewidth=2)
        
        # Show the prediction area as a rectangle on the terrain map
        rect = plt.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min, 
                           fill=False, edgecolor='black', linestyle='--', linewidth=2)
        ax2.add_patch(rect)
        
        # Add markers for start and end points
        ax2.scatter(history['longitude'].iloc[-1], history['latitude'].iloc[-1], 
                  c='blue', s=100, marker='o')
        ax2.scatter(mean_traj[-1, 0], mean_traj[-1, 1], 
                  c=color, s=100, marker='x')
        
        # Add legend for terrain types
        legend_elements = [
            Patch(facecolor='blue', label='Water'),
            Patch(facecolor='gray', label='Urban'),
            Patch(facecolor='yellow', label='Agricultural'),
            Patch(facecolor='darkgreen', label='Forest'),
            Patch(facecolor='lightgreen', label='Grassland'),
            Patch(facecolor='brown', label='Barren'),
            Patch(facecolor='cyan', label='Wetland'),
            Patch(facecolor='white', label='Snow/Ice')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        ax2.set_title('Terrain Analysis')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_all_targets_predictions(
    predictor, 
    target_data, 
    timestamp,
    prediction_duration=300,
    terrain_data=None,
    blue_force_data=None,
    output_path=None,
    use_different_colors=True
):
    """
    Visualize predictions for all targets on the global terrain map.
    
    Args:
        predictor: Trained TargetMovementPredictor model
        target_data: DataFrame with target observations
        timestamp: Time to make predictions from
        prediction_duration: How long to predict ahead (seconds or timedelta)
        terrain_data: Terrain map (optional)
        blue_force_data: Blue force locations (optional)
        output_path: Path to save visualization (optional)
        use_different_colors: Whether to use different colors for different targets
        
    Returns:
        Matplotlib figure
    """
    # Ensure timestamp is datetime object
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Get coordinate bounds for the map
    lon_min, lon_max = target_data['longitude'].min(), target_data['longitude'].max()
    lat_min, lat_max = target_data['latitude'].min(), target_data['latitude'].max()
    
    # Add padding
    lon_padding = (lon_max - lon_min) * 0.05
    lat_padding = (lat_max - lat_min) * 0.05
    lon_min -= lon_padding
    lon_max += lon_padding
    lat_min -= lat_padding
    lat_max += lat_padding
    
    # Load and plot terrain data if available
    if terrain_data is not None or os.path.exists("adapted_data/terrain_map.npy"):
        if terrain_data is None:
            terrain_data = np.load("adapted_data/terrain_map.npy")
        
            # Define colors for each land use category from the LandUseCategory enum
            land_use_colors = [
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

            terrain_cmap = ListedColormap(land_use_colors)

            # Add normalization for the correct range (0-20)
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(0, 20)
        
        # Plot terrain
        ax.imshow(terrain_data, cmap=terrain_cmap, alpha=0.5, 
                  extent=[lon_min, lon_max, lat_min, lat_max], 
                  aspect='auto', zorder=0)
    
    # Plot blue force positions if available
    if blue_force_data is not None:
        ax.scatter(blue_force_data['longitude'], blue_force_data['latitude'],
                  c='blue', s=100, marker='^', label='Blue Forces', zorder=10, edgecolor='black')
    
    # Generate color map for targets if using different colors
    unique_targets = target_data['target_id'].unique()
    num_targets = len(unique_targets)
    
    if use_different_colors:
        # Create a colormap for different targets
        target_colors = plt.cm.rainbow(np.linspace(0, 1, num_targets))
        target_color_map = {target_id: target_colors[i] for i, target_id in enumerate(unique_targets)}
    else:
        # Use red for all targets
        target_color_map = {target_id: 'red' for target_id in unique_targets}
    
    # Add a legend entry for predictions
    legend_handles = []
    if blue_force_data is not None:
        legend_handles.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                                        markersize=10, label='Blue Forces'))
    
    # Track whether we've added legend entries for target types
    added_target_types = set()
    
    # Counter for successful predictions
    successful_predictions = 0
    
    # Process each target
    for target_id in tqdm(unique_targets, desc="Generating predictions"):
        # Filter data for this target
        target_df = target_data[target_data['target_id'] == target_id]
        
        # Skip if no data before timestamp
        if target_df[target_df['datetime'] <= timestamp].empty:
            continue
        
        # Get target class if available
        target_class = target_df['target_class'].iloc[0] if 'target_class' in target_df.columns else 'Unknown'
        
        # Make prediction
        prediction = predictor.predict_out_of_view(
            target_data, 
            target_id, 
            timestamp,
            prediction_duration
        )
        
        if prediction is None:
            continue
        
        successful_predictions += 1
        
        # Get the color for this target
        color = target_color_map[target_id]
        
        # Filter history for this target
        history = target_df[target_df['datetime'] <= timestamp]
        
        # Plot history trajectory as a thin line
        ax.plot(history['longitude'], history['latitude'], '-', color=color, 
                alpha=0.3, linewidth=1, zorder=3)
        
        # Plot last known position
        ax.scatter(history['longitude'].iloc[-1], history['latitude'].iloc[-1], 
                  color=color, s=50, marker='o', zorder=5)
        
        # Plot predicted trajectory
        mean_traj = prediction['mean']
        lower_ci = prediction['lower_ci']
        upper_ci = prediction['upper_ci']
        
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], '--', color=color, 
                linewidth=2, alpha=0.8, zorder=4)
        
        # Plot final predicted position
        ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], color=color, 
                  s=80, marker='x', zorder=5, edgecolor='black')
        
        # Add target ID label at the end of prediction
        ax.annotate(f"ID: {target_id}", 
                   (mean_traj[-1, 0], mean_traj[-1, 1]), 
                   textcoords="offset points", 
                   xytext=(5, 5), 
                   fontsize=8,
                   color=color)
        
        # Add confidence ellipse for the final prediction
        ellipse = plt.matplotlib.patches.Ellipse(
            (mean_traj[-1, 0], mean_traj[-1, 1]),
            width=upper_ci[-1, 0] - lower_ci[-1, 0],
            height=upper_ci[-1, 1] - lower_ci[-1, 1],
            color=color, alpha=0.2, zorder=2
        )
        ax.add_patch(ellipse)
        
        # Add to legend if not already added for this target type
        if target_class not in added_target_types:
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, linestyle='--',
                                           marker='x', markeredgecolor='black',
                                           label=f'{target_class} Target'))
            added_target_types.add(target_class)
    
    # Set title and labels
    ax.set_title(f'All Target Predictions at {timestamp}\nPrediction Duration: {prediction_duration} seconds', 
                fontsize=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Set axis limits
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    
    # Add legend
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add info text
    info_text = (
        f"Timestamp: {timestamp}\n"
        f"Prediction duration: {prediction_duration} seconds\n"
        f"Targets with predictions: {successful_predictions} of {num_targets}\n"
        f"95% confidence ellipses shown"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    return fig

def main():
    """
    Main entry point with improved structure for training, validation, inference, and test set prediction.
    """
    import argparse
    import os
    import torch
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Target Movement Prediction')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'eval', 'infer', 'visualize', 'test'],
                        help='Mode to run: train, eval, infer, visualize, or test')
    parser.add_argument('--config_path', type=str, default='config/default_config.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data files')
    parser.add_argument('--terrain_path', type=str, default='adapted_data/terrain_map.npy',
                        help='Path to terrain data')
    parser.add_argument('--elevation_path', type=str, default='adapted_data/elevation_map.npy',
                        help='Path to elevation data')
    parser.add_argument('--model_path', type=str, default='models/target_prediction/target_predictor_model.pt',
                        help='Path to save/load model')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--target_id', type=str, default=None,
                        help='Target ID for inference/visualization (optional)')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp for inference (optional)')
    parser.add_argument('--prediction_duration', type=int, default=300,
                        help='Duration to predict ahead in seconds')
    parser.add_argument('--create_animation', action='store_true',
                        help='Create animation of predictions')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data file (for test mode)')
    parser.add_argument('--test_output', type=str, default='test_predictions.csv',
                        help='Path to save test predictions')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    if os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            import json
            config = json.load(f)
        print(f"Loaded configuration from {args.config_path}")
    else:
        # Default configuration
        config = {
            'hidden_dim': 128,
            'n_layers': 3,
            'n_heads': 4,
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_epochs': 50,
            'sequence_length': 10,
            'prediction_horizon': 5,
            'terrain_feature_dim': 32,
            'use_terrain': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        print("Using default configuration")
    
    # Load data
    data = load_and_process_data(
        target_csv=os.path.join(args.data_dir, "red_sightings.csv"),
        blue_csv=os.path.join(args.data_dir, "blue_locations.csv"),
        terrain_path=args.terrain_path,
        elevation_path=args.elevation_path
    )
    
    target_data = data['target_data']
    blue_force_data = data['blue_force_data']
    terrain_data = data['terrain_data']
    elevation_data = data['elevation_data']
    
    # Initialize predictor
    predictor = TargetMovementPredictor(
        config=config,
        terrain_data_path=args.terrain_path,
        elevation_data_path=args.elevation_path
    )
    
    # Execute the selected mode
    if args.mode == 'train':
        # Train mode
        train_loader, val_loader = predictor.prepare_data(target_data, blue_force_data)
        
        # Train model
        history = predictor.train(train_loader, val_loader)
        
        # Save training history plot
        predictor.visualize_training_history(
            history,
            save_path=os.path.join(args.output_dir, "training_history.png")
        )
        
        # Evaluate on validation set
        metrics = predictor.evaluate(val_loader)
        print("\nValidation Metrics:")
        for key, value in metrics.items():
            if key != 'errors_by_step':
                print(f"  {key}: {value}")
        
        # Plot errors by prediction step
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(metrics['errors_by_step']) + 1), metrics['errors_by_step'], 'o-')
        plt.title('RMSE by Prediction Step')
        plt.xlabel('Prediction Step')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "errors_by_step.png"), dpi=300)
        
        # Save model
        predictor.save_model(args.model_path)
        print(f"Model saved to {args.model_path}")
        
    elif args.mode == 'eval':
        # Evaluation mode
        # Load model
        if not predictor.load_model(args.model_path):
            print(f"Error: Could not load model from {args.model_path}")
            return
        
        # Prepare data
        _, val_loader = predictor.prepare_data(target_data, blue_force_data)
        
        # Evaluate
        metrics = predictor.evaluate(val_loader)
        print("\nEvaluation Metrics:")
        for key, value in metrics.items():
            if key != 'errors_by_step':
                print(f"  {key}: {value}")
        
        # Visualize sample predictions
        for i, batch in enumerate(val_loader):
            if i >= 3:  # Show 3 examples
                break
                
            inputs = batch['input']
            targets = batch['target']
            terrain = batch['terrain'] if 'terrain' in batch else None
            
            predictor.visualize_predictions(
                inputs, targets, terrain,
                title=f"Target {batch['target_id'][0]} Predictions",
                save_path=os.path.join(args.output_dir, f"prediction_sample_{i}.png")
            )
        
    elif args.mode == 'infer':
        # Inference mode
        # Load model
        if not predictor.load_model(args.model_path):
            print(f"Error: Could not load model from {args.model_path}")
            return
        
        # Parse timestamp if provided
        timestamp = None
        if args.timestamp:
            timestamp = pd.to_datetime(args.timestamp)
        else:
            # Use a timestamp from the middle of the dataset
            timestamps = sorted(target_data['datetime'].unique())
            mid_idx = len(timestamps) // 2
            timestamp = timestamps[mid_idx]
        
        print(f"Using timestamp: {timestamp}")
        
        if args.target_id:
            # Predict for a specific target
            target_id = args.target_id
            
            # Make prediction
            prediction = predictor.predict_out_of_view(
                target_data, 
                target_id, 
                timestamp,
                args.prediction_duration
            )
            
            if prediction is None:
                print(f"Could not generate prediction for target {target_id}")
                return
            
            # Standard visualization
            visualize_out_of_view_prediction(
                predictor,
                target_data,
                target_id,
                timestamp,
                prediction_duration=args.prediction_duration,
                blue_force_data=blue_force_data,
                output_path=os.path.join(args.output_dir, f"prediction_{target_id}.png")
            )
            
            # Enhanced visualization with terrain
            visualize_target_prediction_with_terrain(
                predictor,
                target_data,
                target_id,
                timestamp,
                prediction_duration=args.prediction_duration,
                terrain_data=terrain_data,
                blue_force_data=blue_force_data,
                output_path=os.path.join(args.output_dir, f"prediction_terrain_{target_id}.png")
            )
            
            print(f"Prediction visualizations saved to {args.output_dir}")
        else:
            # Predict for all targets
            visualize_all_targets_predictions(
                predictor,
                target_data,
                timestamp,
                prediction_duration=args.prediction_duration,
                terrain_data=terrain_data,
                blue_force_data=blue_force_data,
                output_path=os.path.join(args.output_dir, "all_targets_prediction.png"),
                use_different_colors=True
            )
            
            print(f"All targets prediction saved to {os.path.join(args.output_dir, 'all_targets_prediction.png')}")
        
        # Create animation if requested
        if args.create_animation:
            create_prediction_animation(
                predictor,
                output_filename=os.path.join(args.output_dir, "target_prediction_animation.mp4")
            )
            
    elif args.mode == 'visualize':
        # Visualization mode (just for terrain/elevation data)
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        if terrain_data is not None:
            # Create terrain colormap
            # Define colors for each land use category from the LandUseCategory enum
            land_use_colors = [
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

            terrain_cmap = ListedColormap(land_use_colors)

            # Add normalization for the correct range (0-20)
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(0, 20)
            
            # Visualize terrain
            plt.figure(figsize=(12, 10))
            plt.imshow(terrain_data, cmap=terrain_cmap, origin='upper')
            plt.colorbar(ticks=range(8), label='Terrain Type')
            plt.title('Terrain Map')
            plt.savefig(os.path.join(args.output_dir, "terrain_map.png"), dpi=300)
            
        if elevation_data is not None:
            # Visualize elevation
            plt.figure(figsize=(12, 10))
            plt.imshow(elevation_data, cmap='terrain', origin='upper')
            plt.colorbar(label='Elevation (m)')
            plt.title('Elevation Map')
            plt.savefig(os.path.join(args.output_dir, "elevation_map.png"), dpi=300)
            
        # Display target and blue force positions
        plt.figure(figsize=(12, 10))
        
        # Get coordinate bounds
        lon_min, lon_max = target_data['longitude'].min(), target_data['longitude'].max()
        lat_min, lat_max = target_data['latitude'].min(), target_data['latitude'].max()
        
        # Plot terrain with correct extent
        if terrain_data is not None:
            plt.imshow(terrain_data, cmap=terrain_cmap, alpha=0.5,
                      extent=[lon_min, lon_max, lat_max, lat_min],
                      aspect='auto', origin='upper')
        
        # Plot targets
        plt.scatter(target_data['longitude'], target_data['latitude'],
                  c='red', s=50, marker='o', label='Red Forces', alpha=0.6)
        
        # Plot blue forces
        plt.scatter(blue_force_data['longitude'], blue_force_data['latitude'],
                  c='blue', s=80, marker='^', label='Blue Forces')
        
        plt.title('Target and Blue Force Positions')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, "positions_map.png"), dpi=300)
        
        print(f"Visualizations saved to {args.output_dir}")
        
    elif args.mode == 'test':
        # Test mode - process the test set
        if args.test_data is None:
            print("Error: Test data path is required for test mode")
            return
            
        # Load model
        if not predictor.load_model(args.model_path):
            print(f"Error: Could not load model from {args.model_path}")
            return
        
        # Run inference on test set
        predict_test_set(
            predictor, 
            args.test_data, 
            os.path.join(args.output_dir, args.test_output),
            prediction_duration=args.prediction_duration
        )

def predict_test_set(predictor, test_data_path, output_path, prediction_duration=300):
    """
    Run inference on a test set and save predictions to a file.
    
    Args:
        predictor: Trained TargetMovementPredictor model
        test_data_path: Path to test data CSV file
        output_path: Path to save predictions
        prediction_duration: Duration to predict ahead in seconds
    
    Returns:
        DataFrame with predictions
    """
    import pandas as pd
    from tqdm import tqdm
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Ensure datetime column is parsed
    if 'datetime' in test_df.columns:
        test_df['datetime'] = pd.to_datetime(test_df['datetime'])
    
    # Get unique target IDs
    target_ids = test_df['target_id'].unique()
    
    # Create results dataframe
    results = []
    
    print(f"Running inference on {len(target_ids)} targets...")
    
    # Process each target
    for target_id in tqdm(target_ids):
        # Get the latest observation for this target
        target_data = test_df[test_df['target_id'] == target_id].sort_values('datetime')
        
        if len(target_data) < predictor.config['sequence_length']:
            print(f"Warning: Not enough history for target {target_id}. Skipping.")
            continue
        
        # Get the last observation time
        last_seen_time = target_data['datetime'].iloc[-1]
        
        # Make prediction
        prediction = predictor.predict_out_of_view(
            test_df, 
            target_id, 
            last_seen_time,
            prediction_duration
        )
        
        if prediction is None:
            print(f"Could not generate prediction for target {target_id}")
            continue
        
        # Extract predicted coordinates and time points
        for i, (time_point, mean_pos, lower_ci, upper_ci) in enumerate(zip(
            prediction['time_points'], 
            prediction['mean'], 
            prediction['lower_ci'], 
            prediction['upper_ci']
        )):
            results.append({
                'target_id': target_id,
                'prediction_step': i + 1,
                'timestamp': time_point,
                'predicted_longitude': mean_pos[0],
                'predicted_latitude': mean_pos[1],
                'longitude_lower_ci': lower_ci[0],
                'longitude_upper_ci': upper_ci[0],
                'latitude_lower_ci': lower_ci[1],
                'latitude_upper_ci': upper_ci[1],
                'last_seen_time': last_seen_time
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    
    return results_df

if __name__ == "__main__":
    main()