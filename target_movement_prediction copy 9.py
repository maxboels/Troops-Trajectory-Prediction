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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, Patch

# Ensure StandardScaler is compatible with serialization
import torch.serialization
torch.serialization.add_safe_globals([StandardScaler, np.core.multiarray.scalar])

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
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

class TargetTrajectoryModel(nn.Module):
    """Simplified target trajectory prediction model with terrain awareness."""
    def __init__(self, input_dim, terrain_dim=32, hidden_dim=128, n_layers=3, n_heads=4, dropout=0.1):
        super().__init__()
        # Position encoder
        self.pos_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Terrain encoder (simplified CNN)
        self.terrain_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, terrain_dim)
        )
        
        # Positional encoding for transformer
        self.pos_enc = PositionalEncoding(hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output heads
        self.out_mean = nn.Linear(hidden_dim, 2)  # x, y coordinates
        self.out_logvar = nn.Linear(hidden_dim, 2)  # uncertainty
        
    def _gen_causal_mask(self, seq_len, device):
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)
    
    def forward(self, pos_seq, terrain_patch):
        """
        Forward pass through the model.
        
        Args:
            pos_seq: Position sequence [B, T, features]
            terrain_patch: Terrain patch [B, 2, H, W]
            
        Returns:
            mean: Predicted mean coordinates [B, T, 2]
            logvar: Predicted log variance [B, T, 2]
        """
        # Encode position sequence
        pos_embed = self.pos_encoder(pos_seq)
        pos_embed = self.pos_enc(pos_embed)
        
        # Encode terrain
        terrain_embed = self.terrain_encoder(terrain_patch)
        terrain_embed = terrain_embed.unsqueeze(1)  # [B, 1, terrain_dim]
        
        # Create causal mask
        seq_len = pos_seq.shape[1]
        causal_mask = self._gen_causal_mask(seq_len, pos_seq.device)
        
        # Transformer decoding
        decoded = self.decoder(
            tgt=pos_embed,
            memory=terrain_embed,
            tgt_mask=causal_mask
        )
        
        # Predict mean and variance
        mean = self.out_mean(decoded)
        logvar = self.out_logvar(decoded)
        
        return mean, logvar

class TargetTrajectoryDataset(Dataset):
    """Dataset for training the target movement predictor with terrain awareness."""
    def __init__(self, 
                target_data,
                terrain_data=None, 
                elevation_data=None,
                blue_force_data=None,
                sequence_length=10, 
                prediction_horizon=5, 
                stride=1,
                terrain_window_size=32):
        
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
            x_coords = group['longitude'].values
            y_coords = group['latitude'].values
            
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
                    
                    # Cyclical encoding of time features
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
                    
                    terrain_patch = self.extract_terrain_patch(latest_x, latest_y)
                
                # Store metadata
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
    
    def extract_terrain_patch(self, lon, lat):
        """
        Extract terrain patch at given coordinates.
        
        Args:
            lon: Longitude
            lat: Latitude
            
        Returns:
            terrain_patch: Terrain patch as numpy array [2, H, W]
        """
        try:
            # Get bounds of coordinate system
            min_lat = np.min(self.target_data['latitude'])
            max_lat = np.max(self.target_data['latitude'])
            min_lon = np.min(self.target_data['longitude'])
            max_lon = np.max(self.target_data['longitude'])
            
            # Map to terrain indices - corrected mapping
            terrain_height, terrain_width = self.terrain_data.shape
            x_idx = int((lon - min_lon) / (max_lon - min_lon) * (terrain_width - 1))
            y_idx = int((max_lat - lat) / (max_lat - min_lat) * (terrain_height - 1))
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, terrain_width - 1))
            y_idx = max(0, min(y_idx, terrain_height - 1))
            
            # Extract terrain patch centered on position
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
            return np.stack([terrain_patch, elevation_patch], axis=0)
            
        except Exception as e:
            print(f"Error extracting terrain patch: {e}")
            # Create a dummy terrain patch if extraction fails
            return np.zeros((2, self.terrain_window_size, self.terrain_window_size))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Prepare input - ensure it has the right shape
        input_data = sequence['input']
        input_seq = torch.tensor(input_data, dtype=torch.float32)
        
        # Prepare target
        target_seq = torch.tensor(sequence['target'], dtype=torch.float32)
        
        # Prepare terrain if available
        terrain = None
        if sequence['terrain'] is not None:
            terrain = torch.tensor(sequence['terrain'], dtype=torch.float32)
        
        # Extract metadata
        metadata = sequence['metadata']
        
        # Return essential elements
        return {
            'input': input_seq,
            'target': target_seq,
            'terrain': terrain,
            'target_id': metadata['target_id'],
            'start_idx': metadata['start_idx']
        }

class TargetMovementPredictor:
    """Main class for predicting target movements."""
    def __init__(self, 
                config=None,
                terrain_data_path=None, 
                elevation_data_path=None):
        
        # Default configuration
        default_config = {
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
        
        # Update with provided config
        self.config = default_config
        if config is not None:
            self.config.update(config)
        
        # Load terrain and elevation data if provided
        self.terrain_data = None
        self.elevation_data = None
        
        if terrain_data_path is not None and os.path.exists(terrain_data_path):
            self.terrain_data = np.load(terrain_data_path)
            print(f"Loaded terrain data with shape: {self.terrain_data.shape}")
        
        if elevation_data_path is not None and os.path.exists(elevation_data_path):
            self.elevation_data = np.load(elevation_data_path)
            print(f"Loaded elevation data with shape: {self.elevation_data.shape}")
        
        # Initialize model and scalers
        self.model = None
        self.input_scaler = None
        self.output_scaler = None
        
        # Loss function
        self.mse_loss = nn.MSELoss()
    
    def setup_feature_scaling(self, train_dataset):
        """Set up feature scaling based on training data."""
        # Initialize scalers
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        # Collect all input and output features from a subset of training data
        all_inputs = []
        all_outputs = []
        
        max_samples = min(len(train_dataset), 1000)
        indices = np.random.choice(len(train_dataset), max_samples, replace=False)
        
        for idx in indices:
            sample = train_dataset[idx]
            inputs = sample['input'].numpy()
            outputs = sample['target'].numpy()
            
            # Reshape for stacking
            if len(inputs.shape) == 2:  # [seq, features]
                inputs_flat = inputs
            else:  # Single feature vector
                inputs_flat = inputs.reshape(1, -1)
                
            if len(outputs.shape) == 2:  # [seq, features]
                outputs_flat = outputs
            else:  # Single feature vector
                outputs_flat = outputs.reshape(1, -1)
            
            all_inputs.append(inputs_flat)
            all_outputs.append(outputs_flat)
        
        # Reshape and fit scalers
        all_inputs = np.vstack([x.reshape(-1, x.shape[-1]) for x in all_inputs])
        all_outputs = np.vstack([x.reshape(-1, x.shape[-1]) for x in all_outputs])
        
        self.input_scaler.fit(all_inputs)
        self.output_scaler.fit(all_outputs)
        
        print(f"Feature scaling set up based on {all_inputs.shape[0]} input samples")
        
    def build_model(self, input_dim):
        """Build the model."""
        self.model = TargetTrajectoryModel(
            input_dim=input_dim,
            terrain_dim=self.config['terrain_feature_dim'],
            hidden_dim=self.config['hidden_dim'],
            n_layers=self.config['n_layers'],
            n_heads=self.config['n_heads'],
            dropout=self.config['dropout']
        )
        
        # Move model to device
        self.model.to(self.config['device'])
    
    def prepare_data(self, target_data, blue_force_data=None, train_ratio=0.8):
        """Prepare data for training and validation."""
        # Ensure datetime is properly parsed
        if 'datetime' in target_data.columns and target_data['datetime'].dtype == object:
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

        # Determine input dimension from sample
        sample = dataset[0]
        input_tensor = sample['input']
        input_dim = input_tensor.shape[1]  # Feature dimension
        
        # Build model
        self.build_model(input_dim)

        return train_loader, val_loader
    
    def normalize_features(self, inputs, outputs=None):
        """Normalize features using fitted scalers."""
        # Handle input normalization
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.cpu().numpy()
            input_shape = inputs.shape
            
            # Reshape for scaling
            inputs_flat = inputs_np.reshape(-1, inputs_np.shape[-1])
            inputs_scaled = self.input_scaler.transform(inputs_flat)
            
            # Reshape back
            inputs_scaled = inputs_scaled.reshape(input_shape)
            inputs_scaled = torch.tensor(inputs_scaled, dtype=inputs.dtype, device=inputs.device)
        else:
            inputs_scaled = self.input_scaler.transform(inputs)
        
        # Handle output normalization if provided
        outputs_scaled = None
        if outputs is not None:
            if isinstance(outputs, torch.Tensor):
                outputs_np = outputs.cpu().numpy()
                output_shape = outputs.shape
                
                # Reshape for scaling
                outputs_flat = outputs_np.reshape(-1, outputs_np.shape[-1])
                outputs_scaled = self.output_scaler.transform(outputs_flat)
                
                # Reshape back
                outputs_scaled = outputs_scaled.reshape(output_shape)
                outputs_scaled = torch.tensor(outputs_scaled, dtype=outputs.dtype, device=outputs.device)
            else:
                outputs_scaled = self.output_scaler.transform(outputs)
        
        return inputs_scaled, outputs_scaled
    
    def denormalize_features(self, outputs):
        """Denormalize features using fitted scalers."""
        if isinstance(outputs, torch.Tensor):
            outputs_np = outputs.cpu().numpy()
            output_shape = outputs.shape
            
            # Reshape for inverse scaling
            outputs_flat = outputs_np.reshape(-1, outputs_np.shape[-1])
            outputs_denorm = self.output_scaler.inverse_transform(outputs_flat)
            
            # Reshape back
            outputs_denorm = outputs_denorm.reshape(output_shape)
            outputs_denorm = torch.tensor(outputs_denorm, dtype=outputs.dtype, device=outputs.device)
        else:
            outputs_denorm = self.output_scaler.inverse_transform(outputs)
        
        return outputs_denorm
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """Train the model."""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
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
            self.model.train()
            train_loss = 0.0
            train_nll_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch in train_pbar:
                # Get batch data
                inputs = batch['input'].to(self.config['device'])
                targets = batch['target'].to(self.config['device'])
                terrain = batch['terrain'].to(self.config['device']) if batch['terrain'] is not None else None
                
                # Normalize
                inputs_norm, targets_norm = self.normalize_features(inputs, targets)
                
                # Check for missing terrain data and create dummy if needed
                if terrain is None:
                    # Create dummy terrain data
                    terrain = torch.zeros((inputs.shape[0], 2, 32, 32), device=self.config['device'])
                
                # Forward pass
                optimizer.zero_grad()

                print(f"inputs_norm shape: {inputs_norm.shape}")
                print(f"terrain shape: {terrain.shape}")
                pred_mean, pred_logvar = self.model(inputs_norm, terrain)
                
                # Extract predictions for the forecast horizon
                pred_mean = pred_mean[:, -self.config['prediction_horizon']:, :]
                pred_logvar = pred_logvar[:, -self.config['prediction_horizon']:, :]
                
                # Calculate losses
                mse_loss = self.mse_loss(pred_mean, targets_norm)
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
            self.model.eval()
            val_loss = 0.0
            val_nll_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for batch in val_pbar:
                    # Get batch data
                    inputs = batch['input'].to(self.config['device'])
                    targets = batch['target'].to(self.config['device'])
                    terrain = batch['terrain'].to(self.config['device']) if batch['terrain'] is not None else None
                    
                    # Normalize
                    inputs_norm, targets_norm = self.normalize_features(inputs, targets)
                    
                    # Check for missing terrain data and create dummy if needed
                    if terrain is None:
                        # Create dummy terrain data
                        terrain = torch.zeros((inputs.shape[0], 2, 32, 32), device=self.config['device'])
                    
                    # Forward pass
                    pred_mean, pred_logvar = self.model(inputs_norm, terrain)
                    
                    # Extract predictions for the forecast horizon
                    pred_mean = pred_mean[:, -self.config['prediction_horizon']:, :]
                    pred_logvar = pred_logvar[:, -self.config['prediction_horizon']:, :]
                    
                    # Calculate losses
                    mse_loss = self.mse_loss(pred_mean, targets_norm)
                    nll_loss = self.gaussian_nll_loss(pred_mean, pred_logvar, targets_norm)
                    
                    # Total loss
                    loss = mse_loss + 0.1 * nll_loss
                    
                    # Update metrics
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
        """Calculate negative log likelihood loss for Gaussian distribution."""
        # Calculate precision (inverse variance)
        precision = torch.exp(-logvar)
        
        # Calculate squared error
        squared_error = (target - mean) ** 2
        
        # Calculate NLL loss
        nll_loss = 0.5 * (logvar + squared_error * precision)
        
        return torch.mean(nll_loss)
    
    def extract_terrain_patch(self, lon, lat, terrain_window_size=32):
        """Extract terrain patch for a given location."""
        if self.terrain_data is None or self.elevation_data is None:
            # Create dummy terrain patch
            return np.zeros((2, terrain_window_size, terrain_window_size))
        
        try:
            # Get data bounds
            min_lat = np.min(np.where(~np.isnan(self.terrain_data), self.terrain_data, 0))
            max_lat = np.max(np.where(~np.isnan(self.terrain_data), self.terrain_data, 0))
            data_height, data_width = self.terrain_data.shape
            
            # Get coordinate bounds
            target_data = pd.DataFrame({'latitude': [lat], 'longitude': [lon]})
            min_lat_coord = np.min(target_data['latitude'])
            max_lat_coord = np.max(target_data['latitude'])
            min_lon_coord = np.min(target_data['longitude'])
            max_lon_coord = np.max(target_data['longitude'])
            
            # Map to terrain indices - corrected mapping
            x_idx = int((lon - min_lon_coord) / (max_lon_coord - min_lon_coord) * (data_width - 1))
            y_idx = int((max_lat_coord - lat) / (max_lat_coord - min_lat_coord) * (data_height - 1))
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, data_width - 1))
            y_idx = max(0, min(y_idx, data_height - 1))
            
            # Extract terrain patch centered at position
            half_window = terrain_window_size // 2
            
            # Ensure within bounds
            x_min = max(0, x_idx - half_window)
            x_max = min(data_width, x_idx + half_window)
            y_min = max(0, y_idx - half_window)
            y_max = min(data_height, y_idx + half_window)
            
            # Extract patches
            terrain_patch = self.terrain_data[y_min:y_max, x_min:x_max]
            elevation_patch = self.elevation_data[y_min:y_max, x_min:x_max]
            
            # Pad if necessary
            if terrain_patch.shape[0] < terrain_window_size or terrain_patch.shape[1] < terrain_window_size:
                padded_terrain = np.zeros((terrain_window_size, terrain_window_size))
                padded_elevation = np.zeros((terrain_window_size, terrain_window_size))
                
                # Copy available data
                h, w = terrain_patch.shape
                padded_terrain[:h, :w] = terrain_patch
                padded_elevation[:h, :w] = elevation_patch
                
                terrain_patch = padded_terrain
                elevation_patch = padded_elevation
            
            # Stack terrain and elevation
            return np.stack([terrain_patch, elevation_patch], axis=0)
        
        except Exception as e:
            print(f"Error extracting terrain patch: {e}")
            # Return dummy patch on error
            return np.zeros((2, terrain_window_size, terrain_window_size))

    def predict_trajectory(self, input_sequence, time_delta=60, steps=60):
        """
        Predict target trajectory for the next 60 minutes using auto-regressive approach.
        
        Args:
            input_sequence: Initial position sequence [seq_len, features]
            time_delta: Time step in seconds (default: 60s = 1 minute)
            steps: Number of steps to predict (default: 60 steps = 60 minutes)
        
        Returns:
            Dictionary with predicted positions and confidence intervals
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Convert to tensor and add batch dimension
        if isinstance(input_sequence, np.ndarray):
            input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        
        if len(input_sequence.shape) == 2:
            input_sequence = input_sequence.unsqueeze(0)
        
        input_sequence = input_sequence.to(device)
        
        # Initialize with input sequence
        current_sequence = input_sequence.clone()
        
        # For storing results
        all_predictions = []
        all_lower_ci = []
        all_upper_ci = []
        time_points = []
        
        # Start time (from the last observation)
        current_time = pd.Timestamp.now()  # Placeholder, will be replaced if available
        
        # Try to extract time from input features if available
        # Assuming time features are in a specific format at indices 13-22
        # This is heuristic and depends on your feature format
        
        # For storing predicted times
        starting_time = None
        
        with torch.no_grad():
            for step in range(steps):
                # Extract terrain at current position
                current_pos = current_sequence[0, -1, :2].cpu().numpy()  # Last position

                # Dynamically get terrain data for the current position
                terrain_patch = self.extract_terrain_patch(current_pos[0], current_pos[1])
                terrain_patch = torch.tensor(terrain_patch, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Normalize input
                current_norm, _ = self.normalize_features(current_sequence)
                
                # Predict next position
                pred_mean, pred_logvar = self.model(current_norm, terrain_patch)
                
                # Take the last prediction
                next_pos_mean = pred_mean[:, -1:, :]
                next_pos_logvar = pred_logvar[:, -1:, :]
                
                # Calculate confidence interval
                pred_std = torch.exp(0.5 * next_pos_logvar)
                lower_ci = next_pos_mean - 1.96 * pred_std
                upper_ci = next_pos_mean + 1.96 * pred_std
                
                # Denormalize predictions
                next_pos_mean_denorm = self.denormalize_features(next_pos_mean)
                lower_ci_denorm = self.denormalize_features(lower_ci)
                upper_ci_denorm = self.denormalize_features(upper_ci)
                
                # Store predictions
                all_predictions.append(next_pos_mean_denorm.cpu().numpy())
                all_lower_ci.append(lower_ci_denorm.cpu().numpy())
                all_upper_ci.append(upper_ci_denorm.cpu().numpy())
                
                # Calculate time point
                if starting_time is None:
                    starting_time = pd.Timestamp.now()
                current_time = starting_time + timedelta(seconds=time_delta * (step + 1))
                time_points.append(current_time)
                
                # Update sequence for next step (auto-regressive)
                # Remove oldest position and add new prediction
                # Keep all features intact except position
                new_features = current_sequence[:, -1:, :].clone()
                new_features[:, :, :2] = next_pos_mean_denorm  # Update position only
                
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],  # Remove first element
                    new_features  # Add new position with other features
                ], dim=1)
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=1)
        lower_bounds = np.concatenate(all_lower_ci, axis=1)
        upper_bounds = np.concatenate(all_upper_ci, axis=1)
        
        # Remove batch dimension if only one sample
        if predictions.shape[0] == 1:
            predictions = predictions.squeeze(0)
            lower_bounds = lower_bounds.squeeze(0)
            upper_bounds = upper_bounds.squeeze(0)
        
        return {
            'mean': predictions,
            'lower_ci': lower_bounds,
            'upper_ci': upper_bounds,
            'time_points': time_points
        }

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
        self.model.eval()
        
        # Add batch dimension if necessary
        if len(input_sequence.shape) == 2:
            input_sequence = input_sequence.unsqueeze(0)
        
        if terrain_patch is not None and len(terrain_patch.shape) == 3:
            terrain_patch = terrain_patch.unsqueeze(0)
        
        # Move to device
        input_sequence = input_sequence.to(self.config['device'])
        
        # Handle missing terrain data
        if terrain_patch is None:
            # Get current position
            current_pos = input_sequence[0, -1, :2].cpu().numpy()
            
            # Extract terrain patch
            terrain_data = self.extract_terrain_patch(current_pos[0], current_pos[1])
            terrain_patch = torch.tensor(terrain_data, dtype=torch.float32).unsqueeze(0)
        
        terrain_patch = terrain_patch.to(self.config['device'])
        
        # Normalize input
        input_norm, _ = self.normalize_features(input_sequence)
        
        # Predict with the model
        with torch.no_grad():
            pred_mean, pred_logvar = self.model(input_norm, terrain_patch)
            
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
        """Save the model."""
        save_dict = {
            'model_state': self.model.state_dict(),
            'config': self.config,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler
        }
        
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load the model."""
        if not os.path.exists(filename):
            print(f"Model file {filename} not found")
            return False
        
        try:
            # Load checkpoint
            checkpoint = torch.load(filename, map_location=self.config['device'])
            
            # Update config
            self.config.update(checkpoint['config'])
            
            # Load scalers
            self.input_scaler = checkpoint['input_scaler']
            self.output_scaler = checkpoint['output_scaler']
            
            # Initialize model if not already initialized
            if self.model is None:
                # Determine input dimension
                input_dim = self.input_scaler.mean_.shape[0]
                self.build_model(input_dim)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state'])
            
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_out_of_view(self, target_data, target_id, last_seen_time, prediction_duration,
                           terrain_data=None, blue_force_data=None):
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
            
            # Extract temporal features from timestamp
            dt = row['datetime']
            
            # Cyclical encoding of time features
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
        
        # Convert to tensor
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        
        # Get terrain at last position
        latest_pos = input_sequence[-1, :2].numpy()
        terrain_patch = self.extract_terrain_patch(latest_pos[0], latest_pos[1])
        terrain_patch = torch.tensor(terrain_patch, dtype=torch.float32)
        
        # Make prediction for requested duration
        if isinstance(prediction_duration, (int, float)):
            # Duration in seconds
            time_delta = prediction_duration / self.config['prediction_horizon']
        else:
            # Assume it's already a timedelta
            time_delta = prediction_duration.total_seconds() / self.config['prediction_horizon']
        
        # Make predictions
        predictions = self.predict(input_sequence, terrain_patch)
        
        # Calculate time points for predictions
        step_delta = timedelta(seconds=time_delta)
        time_points = [last_seen_time + step_delta * (i + 1) for i in range(predictions['mean'].shape[0])]
        
        # Add time information to predictions
        predictions['time_points'] = time_points
        
        return predictions

    def visualize_predictions(self, input_sequence, true_future=None, terrain_patch=None, 
                            title="Target Movement Prediction", save_path=None):
        """Visualize predictions with confidence intervals."""
        # Make predictions
        predictions = self.predict(input_sequence, terrain_patch)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot input sequence
        input_np = input_sequence.cpu().numpy() if isinstance(input_sequence, torch.Tensor) else input_sequence
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
            ellipse = Ellipse(
                (mean[i, 0], mean[i, 1]),
                width=upper[i, 0] - lower[i, 0],
                height=upper[i, 1] - lower[i, 1],
                color='red', alpha=0.2
            )
            ax.add_patch(ellipse)
        
        # Plot true future if provided
        if true_future is not None:
            true_np = true_future.cpu().numpy() if isinstance(true_future, torch.Tensor) else true_future
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
    
    def visualize_all_targets_predictions(self, target_data, timestamp, prediction_duration=300,
                                         blue_force_data=None, output_path=None):
        """Visualize predictions for all targets on the terrain map."""
        # Ensure timestamp is datetime object
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Get coordinate bounds
        lon_min, lon_max = target_data['longitude'].min(), target_data['longitude'].max()
        lat_min, lat_max = target_data['latitude'].min(), target_data['latitude'].max()
        
        # Add padding
        lon_padding = (lon_max - lon_min) * 0.05
        lat_padding = (lat_max - lat_min) * 0.05
        lon_min -= lon_padding
        lon_max += lon_padding
        lat_min -= lat_padding
        lat_max += lat_padding
        
        # Plot terrain if available
        if self.terrain_data is not None:
            # Define colors for each land use category
            land_use_colors = [
                '#FFFFFF',  # 0: No data
                '#1A5BAB',  # 1: Broadleaf Forest
                '#358221',  # 2: Deciduous Forest
                '#2E8B57',  # 3: Evergreen Forest
                '#52A72D',  # 4: Mixed Forest
                '#76B349',  # 5: Tree Open
                '#D2B48C',  # 7: Shrub
                '#9ACD32',  # 8: Herbaceous
                '#F5DEB3',  # 10: Sparse vegetation
                '#FFD700',  # 11: Cropland
                '#F4A460',  # 12: Agricultural
                '#2F4F4F',  # 14: Wetland
                '#A0522D',  # 16: Bare area
                '#808080',  # 18: Urban
                '#0000FF',  # 20: Water
            ]
            
            terrain_cmap = ListedColormap(land_use_colors)
            
            # Plot terrain
            ax.imshow(self.terrain_data, cmap=terrain_cmap, alpha=0.5,
                     extent=[lon_min, lon_max, lat_max, lat_min],
                     aspect='auto', origin='upper', zorder=0)
        
        # Plot blue force positions
        if blue_force_data is not None:
            ax.scatter(blue_force_data['longitude'], blue_force_data['latitude'],
                     c='blue', s=100, marker='^', label='Blue Forces', zorder=10, edgecolor='black')
        
        # Target color map by class
        target_colors = {
            'tank': 'darkred',
            'armoured personnel carrier': 'orangered',
            'light vehicle': 'coral'
        }
        
        # Track legend entries
        added_target_types = set()
        legend_handles = []
        
        if blue_force_data is not None:
            legend_handles.append(plt.Line2D([], [], marker='^', color='blue',
                                          markersize=10, label='Blue Forces'))
        
        # Process unique targets
        unique_targets = target_data['target_id'].unique()
        successful_predictions = 0
        
        for target_id in tqdm(unique_targets, desc="Generating predictions"):
            # Filter data for this target
            target_df = target_data[target_data['target_id'] == target_id]
            
            # Skip if no data before timestamp
            if target_df[target_df['datetime'] <= timestamp].empty:
                continue
            
            # Get target class
            target_class = target_df['target_class'].iloc[0] if 'target_class' in target_df.columns else 'Unknown'
            color = target_colors.get(target_class, 'red')
            
            # Make prediction
            prediction = self.predict_out_of_view(
                target_data, 
                target_id, 
                timestamp,
                prediction_duration
            )
            
            if prediction is None:
                continue
            
            successful_predictions += 1
            
            # Get history for this target
            history = target_df[target_df['datetime'] <= timestamp]
            
            # Plot history trajectory
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
            
            # Add target ID label
            ax.annotate(f"ID: {target_id}", 
                      (mean_traj[-1, 0], mean_traj[-1, 1]), 
                      textcoords="offset points", 
                      xytext=(5, 5), 
                      fontsize=8,
                      color=color)
            
            # Add confidence ellipse for final prediction
            ellipse = Ellipse(
                (mean_traj[-1, 0], mean_traj[-1, 1]),
                width=upper_ci[-1, 0] - lower_ci[-1, 0],
                height=upper_ci[-1, 1] - lower_ci[-1, 1],
                color=color, alpha=0.2, zorder=2
            )
            ax.add_patch(ellipse)
            
            # Add to legend if not already added for this target type
            if target_class not in added_target_types:
                legend_handles.append(plt.Line2D([], [], color=color, linestyle='--',
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
            f"Targets with predictions: {successful_predictions} of {len(unique_targets)}\n"
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

def load_and_process_data(target_csv="data/red_sightings.csv", 
                         blue_csv="data/blue_locations.csv",
                         terrain_path="adapted_data/terrain_map.npy",
                         elevation_path="adapted_data/elevation_map.npy"):
    """Load and process data files."""
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

def run_training_pipeline(target_csv="data/red_sightings.csv",
                         blue_csv="data/blue_locations.csv",
                         terrain_path="adapted_data/terrain_map.npy",
                         elevation_path="adapted_data/elevation_map.npy",
                         output_dir="models/target_prediction",
                         config=None):
    """Run the full training pipeline for target movement prediction."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_and_process_data(
        target_csv=target_csv,
        blue_csv=blue_csv,
        terrain_path=terrain_path,
        elevation_path=elevation_path
    )
    
    # Initialize predictor
    predictor = TargetMovementPredictor(
        config=config,
        terrain_data_path=terrain_path,
        elevation_data_path=elevation_path
    )
    
    # Prepare data
    train_loader, val_loader = predictor.prepare_data(data['target_data'], data['blue_force_data'])
    
    # Train model
    history = predictor.train(train_loader, val_loader)
    
    # Save model
    predictor.save_model(os.path.join(output_dir, "target_predictor_model.pt"))
    
    print(f"\nTraining pipeline completed. Model saved to {output_dir}")
    
    return predictor

def predict_test_set(predictor, test_data_path, output_path, prediction_duration=3600):
    """Run inference on a test set and save predictions to a file."""
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Target Movement Prediction')
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'predict', 'visualize', 'test'],
                      help='Mode to run (train, predict, visualize, test)')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing data files')
    parser.add_argument('--model_path', type=str, default='models/target_predictor_model.pt',
                      help='Path to save/load model')
    parser.add_argument('--target_id', type=str, default=None,
                      help='Target ID for prediction (optional)')
    parser.add_argument('--prediction_duration', type=int, default=3600,
                      help='Duration to predict in seconds (default: 60 minutes)')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_and_process_data(
        target_csv=os.path.join(args.data_dir, "red_sightings.csv"),
        blue_csv=os.path.join(args.data_dir, "blue_locations.csv"),
        terrain_path="adapted_data/terrain_map.npy",
        elevation_path="adapted_data/elevation_map.npy"
    )
    
    # Execute the selected mode
    if args.mode == 'train':
        # Train mode
        run_training_pipeline(
            target_csv=os.path.join(args.data_dir, "red_sightings.csv"),
            blue_csv=os.path.join(args.data_dir, "blue_locations.csv"),
            terrain_path="adapted_data/terrain_map.npy",
            elevation_path="adapted_data/elevation_map.npy",
            output_dir=args.output_dir
        )
        
    elif args.mode == 'predict':
        # Prediction mode
        predictor = TargetMovementPredictor(
            terrain_data_path="adapted_data/terrain_map.npy",
            elevation_data_path="adapted_data/elevation_map.npy"
        )
        
        # Load model
        if not predictor.load_model(args.model_path):
            print(f"Error: Could not load model from {args.model_path}")
            exit(1)
        
        # Select a timestamp
        timestamps = sorted(data['target_data']['datetime'].unique())
        mid_idx = len(timestamps) // 2
        timestamp = timestamps[mid_idx]
        
        # Make predictions for all targets
        predictor.visualize_all_targets_predictions(
            data['target_data'],
            timestamp,
            prediction_duration=args.prediction_duration,
            blue_force_data=data['blue_force_data'],
            output_path=os.path.join(args.output_dir, "all_targets_prediction.png")
        )
        
    elif args.mode == 'visualize':
        # Visualization mode (terrain/elevation/positions)
        # Create colormap for terrain visualization
        land_use_colors = [
            '#FFFFFF',  # 0: No data
            '#1A5BAB',  # 1: Broadleaf Forest
            '#358221',  # 2: Deciduous Forest
            '#2E8B57',  # 3: Evergreen Forest
            '#52A72D',  # 4: Mixed Forest
            '#76B349',  # 5: Tree Open
            '#D2B48C',  # 7: Shrub
            '#9ACD32',  # 8: Herbaceous
            '#F5DEB3',  # 10: Sparse vegetation
            '#FFD700',  # 11: Cropland
            '#F4A460',  # 12: Agricultural
            '#2F4F4F',  # 14: Wetland
            '#A0522D',  # 16: Bare area
            '#808080',  # 18: Urban
            '#0000FF',  # 20: Water
        ]
        
        terrain_cmap = ListedColormap(land_use_colors)
        
        # Visualize terrain
        if data['terrain_data'] is not None:
            plt.figure(figsize=(12, 10))
            plt.imshow(data['terrain_data'], cmap=terrain_cmap, origin='upper')
            plt.colorbar(label='Terrain Type')
            plt.title('Terrain Map')
            plt.savefig(os.path.join(args.output_dir, "terrain_map.png"), dpi=300)
        
        # Visualize elevation
        if data['elevation_data'] is not None:
            plt.figure(figsize=(12, 10))
            plt.imshow(data['elevation_data'], cmap='terrain', origin='upper')
            plt.colorbar(label='Elevation (m)')
            plt.title('Elevation Map')
            plt.savefig(os.path.join(args.output_dir, "elevation_map.png"), dpi=300)
        
        # Display target and blue force positions
        plt.figure(figsize=(12, 10))
        
        # Get coordinate bounds
        lon_min, lon_max = data['target_data']['longitude'].min(), data['target_data']['longitude'].max()
        lat_min, lat_max = data['target_data']['latitude'].min(), data['target_data']['latitude'].max()
        
        # Plot targets
        plt.scatter(data['target_data']['longitude'], data['target_data']['latitude'],
                  c='red', s=50, marker='o', label='Red Forces')
        
        # Plot blue forces
        plt.scatter(data['blue_force_data']['longitude'], data['blue_force_data']['latitude'],
                  c='blue', s=80, marker='^', label='Blue Forces')
        
        plt.title('Target and Blue Force Positions')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "positions_map.png"), dpi=300)
        
        print(f"Visualizations saved to {args.output_dir}")
        
    elif args.mode == 'test':
        # Test mode - run on test set
        predictor = TargetMovementPredictor(
            terrain_data_path="adapted_data/terrain_map.npy",
            elevation_data_path="adapted_data/elevation_map.npy"
        )
        
        # Load model
        if not predictor.load_model(args.model_path):
            print(f"Error: Could not load model from {args.model_path}")
            exit(1)
        
        # Generate predictions
        predict_test_set(
            predictor,
            os.path.join(args.data_dir, "test_data.csv"),
            os.path.join(args.output_dir, "test_predictions.csv"),
            prediction_duration=args.prediction_duration
        )
