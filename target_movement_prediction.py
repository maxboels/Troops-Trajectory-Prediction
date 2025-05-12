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
# import torch.serialization
# torch.serialization.add_safe_globals([StandardScaler, np.core.multiarray.scalar])

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
    """Target trajectory prediction model with terrain awareness."""
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
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim)  # Match hidden_dim for compatibility
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
        # Debug input shapes
        batch_size, seq_len, feature_dim = pos_seq.shape
        
        # Encode position sequence
        pos_embed = self.pos_encoder(pos_seq)
        pos_embed = self.pos_enc(pos_embed)  # [B, T, hidden_dim]
        
        # Encode terrain
        terrain_embed = self.terrain_encoder(terrain_patch)  # [B, hidden_dim]
        # Expand terrain embedding to match sequence length for cross-attention
        terrain_embed = terrain_embed.unsqueeze(1).expand(-1, 1, -1)  # [B, 1, hidden_dim]
        
        # Create causal mask
        causal_mask = self._gen_causal_mask(seq_len, pos_seq.device)
        
        # Transformer decoding
        decoded = self.decoder(
            tgt=pos_embed,  # [B, T, hidden_dim]
            memory=terrain_embed,  # [B, 1, hidden_dim]
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
            'num_epochs': 100,
            'sequence_length': 10,
            'prediction_horizon': 5,
            'terrain_feature_dim': 32,
            'use_terrain': True,
            'max_confidence_threshold': 0.5,  # Maximum confidence interval size (in normalized space)
            'device': 'cpu'  # Default to CPU, will be updated in run_training_pipeline
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
        
        # Move model to device with error handling
        try:
            self.model.to(self.config['device'])
            print(f"Model successfully moved to {self.config['device']}")
        except Exception as e:
            print(f"Error moving model to {self.config['device']}: {str(e)}")
            print("Falling back to CPU")
            self.config['device'] = 'cpu'
            self.model.to('cpu')
    
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
            for batch_idx, batch in enumerate(train_pbar):
                # Get batch data
                inputs = batch['input'].to(self.config['device'])
                targets = batch['target'].to(self.config['device'])
                terrain = None
                if 'terrain' in batch and batch['terrain'] is not None:
                    terrain = batch['terrain'].to(self.config['device'])
                
                # Normalize
                inputs_norm, targets_norm = self.normalize_features(inputs, targets)
                
                # Print shapes for debugging (first batch only)
                if batch_idx == 0 and epoch == 0:
                    print(f"inputs_norm shape: {inputs_norm.shape}")
                    if terrain is not None:
                        print(f"terrain shape: {terrain.shape}")
                
                # Check for missing terrain data and create dummy if needed
                if terrain is None:
                    # Create dummy terrain data
                    terrain = torch.zeros((inputs.shape[0], 2, 32, 32), device=self.config['device'])
                
                # Forward pass
                optimizer.zero_grad()
                
                try:
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
                    
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    print(f"Input shape: {inputs_norm.shape}, Terrain shape: {terrain.shape}")
                    # Skip this batch and continue with next
                    continue
            
            # Calculate average training loss
            train_loss /= max(1, len(train_loader))
            train_nll_loss /= max(1, len(train_loader))
            
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
                    terrain = None
                    if 'terrain' in batch and batch['terrain'] is not None:
                        terrain = batch['terrain'].to(self.config['device'])
                    
                    # Normalize
                    inputs_norm, targets_norm = self.normalize_features(inputs, targets)
                    
                    # Check for missing terrain data and create dummy if needed
                    if terrain is None:
                        # Create dummy terrain data
                        terrain = torch.zeros((inputs.shape[0], 2, 32, 32), device=self.config['device'])
                    
                    try:
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
                    
                    except RuntimeError as e:
                        print(f"Error in validation batch: {e}")
                        continue
            
            # Calculate average validation loss
            val_loss /= max(1, len(val_loader))
            val_nll_loss /= max(1, len(val_loader))
            
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
            # Get terrain dimensions
            data_height, data_width = self.terrain_data.shape
            
            # Define coordinate bounds
            # Use all data points or a predefined range
            min_lat = 44.5  # Approximate for Nova Scotia area 
            max_lat = 46.0
            min_lon = -65.0
            max_lon = -62.0
            
            # Convert lat/lon to terrain indices
            # Note: Y-axis is inverted (max_lat at the top)
            x_idx = int((lon - min_lon) / (max_lon - min_lon) * (data_width - 1))
            y_idx = int((max_lat - lat) / (max_lat - min_lat) * (data_height - 1))
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, data_width - 1))
            y_idx = max(0, min(y_idx, data_height - 1))
            
            # Extract terrain patch centered at position
            half_window = terrain_window_size // 2
            
            # Ensure patch boundaries are within dataset bounds
            x_min = max(0, x_idx - half_window)
            x_max = min(data_width, x_idx + half_window + (terrain_window_size % 2))
            y_min = max(0, y_idx - half_window)
            y_max = min(data_height, y_idx + half_window + (terrain_window_size % 2))
            
            # Extract patches
            terrain_patch = self.terrain_data[y_min:y_max, x_min:x_max].copy()
            elevation_patch = self.elevation_data[y_min:y_max, x_min:x_max].copy()
            
            # Ensure exact size with padding if needed
            if terrain_patch.shape[0] != terrain_window_size or terrain_patch.shape[1] != terrain_window_size:
                padded_terrain = np.zeros((terrain_window_size, terrain_window_size))
                padded_elevation = np.zeros((terrain_window_size, terrain_window_size))
                
                # Copy available data
                h, w = terrain_patch.shape
                h_to_copy = min(h, terrain_window_size)
                w_to_copy = min(w, terrain_window_size)
                
                padded_terrain[:h_to_copy, :w_to_copy] = terrain_patch[:h_to_copy, :w_to_copy]
                padded_elevation[:h_to_copy, :w_to_copy] = elevation_patch[:h_to_copy, :w_to_copy]
                
                terrain_patch = padded_terrain
                elevation_patch = padded_elevation
            
            # Stack terrain and elevation
            result = np.stack([terrain_patch, elevation_patch], axis=0)
            
            # Normalize for better neural network processing
            result = result.astype(np.float32)
            # Ensure no NaN or inf values
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
            return result
        
        except Exception as e:
            print(f"Error extracting terrain patch for location ({lon}, {lat}): {e}")
            # Return dummy patch on error
            return np.zeros((2, terrain_window_size, terrain_window_size), dtype=np.float32)

    def predict_trajectory(self, input_sequence, time_delta=60, steps=60, max_confidence_threshold=None):
        """
        Predict target trajectory for the next 60 minutes using auto-regressive approach.
        With early stopping based on confidence interval size.
        
        Args:
            input_sequence: Initial position sequence [seq_len, features]
            time_delta: Time step in seconds (default: 60s = 1 minute)
            steps: Number of steps to predict (default: 60 steps = 60 minutes)
            max_confidence_threshold: Max allowable confidence interval size (if None, use config value)
            
        Returns:
            Dictionary with predicted positions, confidence intervals, timestamps and speeds
        """
        if max_confidence_threshold is None:
            max_confidence_threshold = self.config['max_confidence_threshold']
            
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
        speeds = []
        
        # Start time (from the last observation)
        current_time = pd.Timestamp.now()  # Placeholder, will be replaced if available
        
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
                
                # Check if confidence interval exceeds threshold (early stopping)
                max_std = torch.max(pred_std).item()
                if max_std > max_confidence_threshold:
                    print(f"Early stopping at step {step+1}: confidence interval {max_std:.4f} exceeds threshold {max_confidence_threshold:.4f}")
                    break
                
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
                
                # Calculate speed if we have at least 2 predictions
                if len(all_predictions) > 1:
                    prev_pos = all_predictions[-2][0, 0]  # [batch, seq_pos, coord]
                    curr_pos = all_predictions[-1][0, 0]
                    
                    # Calculate distance in meters (approximate for small distances)
                    # Using haversine formula would be more accurate for large distances
                    lat1, lon1 = prev_pos[1], prev_pos[0]
                    lat2, lon2 = curr_pos[1], curr_pos[0]
                    
                    # Approximate conversion to meters (rough estimate)
                    km_per_degree_lat = 111.32  # km per degree of latitude
                    km_per_degree_lon = 111.32 * np.cos(np.radians((lat1 + lat2) / 2))
                    
                    dist_lat_km = (lat2 - lat1) * km_per_degree_lat
                    dist_lon_km = (lon2 - lon1) * km_per_degree_lon
                    
                    distance_km = np.sqrt(dist_lat_km**2 + dist_lon_km**2)
                    distance_m = distance_km * 1000
                    
                    # Calculate speed in m/s
                    speed = distance_m / time_delta
                    speeds.append(speed)
                else:
                    speeds.append(0.0)  # First point has no speed
                
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
        predictions = np.concatenate(all_predictions, axis=1) if all_predictions else np.array([])
        lower_bounds = np.concatenate(all_lower_ci, axis=1) if all_lower_ci else np.array([])
        upper_bounds = np.concatenate(all_upper_ci, axis=1) if all_upper_ci else np.array([])
        
        # Remove batch dimension if only one sample
        if len(predictions) > 0 and predictions.shape[0] == 1:
            predictions = predictions.squeeze(0)
            lower_bounds = lower_bounds.squeeze(0)
            upper_bounds = upper_bounds.squeeze(0)
        
        return {
            'mean': predictions,
            'lower_ci': lower_bounds,
            'upper_ci': upper_bounds,
            'time_points': time_points,
            'speeds': speeds
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
        """Save the model weights and configuration separately."""
        # Save model weights
        weights_path = filename
        torch.save(self.model.state_dict(), weights_path)
        
        # Save configuration and scalers
        config_path = filename.replace('.pt', '_config.pkl')
        config_dict = {
            'config': self.config,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler
        }
        
        with open(config_path, 'wb') as f:
            import pickle
            pickle.dump(config_dict, f)
        
        print(f"Model weights saved to {weights_path}")
        print(f"Model configuration saved to {config_path}")
    
    def load_model(self, filename):
        """Load the model weights and configuration."""
        if not os.path.exists(filename):
            print(f"Model file {filename} not found")
            return False
        
        try:
            # First load configuration
            config_path = filename.replace('.pt', '_config.pkl')
            if not os.path.exists(config_path):
                print(f"Config file {config_path} not found")
                return False
            
            # Load config and scalers
            with open(config_path, 'rb') as f:
                import pickle
                config_dict = pickle.load(f)
            
            # Update attributes
            self.config.update(config_dict['config'])
            self.input_scaler = config_dict['input_scaler']
            self.output_scaler = config_dict['output_scaler']
            
            # Initialize model if not already initialized
            if self.model is None:
                # Determine input dimension
                input_dim = self.input_scaler.mean_.shape[0]
                self.build_model(input_dim)
            
            # Load model weights
            self.model.load_state_dict(torch.load(filename, map_location=self.config['device']))
            
            print(f"Model loaded from {filename} and {config_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_out_of_view(self, target_data, target_id, last_seen_time, prediction_duration,
                           terrain_data=None, blue_force_data=None, max_confidence_threshold=None):
        """
        Predict the target's movement after it goes out of view.
        
        Args:
            target_data: DataFrame with target observations
            target_id: ID of the target to predict
            last_seen_time: Time when target was last seen
            prediction_duration: Duration to predict in seconds
            terrain_data: Terrain data (optional)
            blue_force_data: Blue force locations (optional)
            max_confidence_threshold: Max confidence interval size for early stopping
            
        Returns:
            Dictionary with predictions, confidence intervals, speeds and timestamps
        """
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
        
        # Determine time delta based on prediction duration
        if isinstance(prediction_duration, (int, float)):
            # Time step size in seconds
            time_delta = 60  # Default 1 minute per step
            steps = prediction_duration // time_delta
        else:
            # Assume it's already a timedelta
            time_delta = 60  # 1 minute steps
            steps = prediction_duration.total_seconds() // time_delta
        
        # Ensure at least 1 step
        steps = max(1, int(steps))
        
        # Make predictions
        predictions = self.predict_trajectory(
            input_sequence, 
            time_delta=time_delta, 
            steps=steps,
            max_confidence_threshold=max_confidence_threshold
        )
        
        # Ensure timestamps start from last_seen_time
        time_points = []
        start_time = last_seen_time
        for i in range(len(predictions['mean'])):
            time_points.append(start_time + timedelta(seconds=time_delta * (i + 1)))
        
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
            self.terrain_data = np.flipud(self.terrain_data)
            ax.imshow(self.terrain_data, cmap=terrain_cmap, alpha=0.5,
                extent=[lon_min, lon_max, lat_min, lat_max],
                aspect='auto', origin='lower', zorder=0)
            
            # Add elevation overlay if available
            if self.elevation_data is not None:
                # Normalize elevation
                from matplotlib.colors import Normalize
                elev_min = np.min(self.elevation_data)
                elev_max = np.max(self.elevation_data)
                elev_norm = Normalize(vmin=elev_min, vmax=elev_max)
                
                # Plot with low alpha
                self.elevation_data = np.flipud(self.elevation_data)
                ax.imshow(self.elevation_data, cmap='terrain', norm=elev_norm, alpha=0.3,
                    extent=[lon_min, lon_max, lat_min, lat_max],
                    aspect='auto', origin='lower', zorder=1)
        
        # Plot blue force positions
        if blue_force_data is not None:
            ax.scatter(blue_force_data['longitude'], blue_force_data['latitude'],
                     c='blue', s=100, marker='^', label='Blue Forces', zorder=10, edgecolor='black')
            
            # Add labels for blue forces if 'name' column exists
            if 'name' in blue_force_data.columns:
                for _, row in blue_force_data.iterrows():
                    ax.annotate(row['name'],
                              (row['longitude'], row['latitude']),
                              xytext=(5, 5),
                              textcoords="offset points",
                              fontsize=9,
                              color='blue',
                              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
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
            
            if prediction is None or len(prediction['mean']) == 0:
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
            speeds = prediction.get('speeds', [])
            
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
            
            # Add speed label to final prediction if available
            if speeds and len(speeds) > 0:
                avg_speed = sum(speeds) / len(speeds)
                ax.annotate(f"Avg: {avg_speed:.1f} m/s", 
                          (mean_traj[-1, 0], mean_traj[-1, 1]), 
                          textcoords="offset points", 
                          xytext=(5, 20), 
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
    
    def visualize_prediction_with_terrain(self, target_data, target_id, timestamp, prediction_duration=300,
                            terrain_data=None, blue_force_data=None, output_path=None):
        """
        Create a detailed visualization of a single target's prediction with terrain map.
        Modified to reduce text clutter and only show start/end information.
        """
        # Ensure timestamp is datetime
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Make prediction
        prediction = self.predict_out_of_view(
            target_data, 
            target_id, 
            timestamp,
            prediction_duration
        )
        
        if prediction is None:
            print(f"Could not generate prediction for target {target_id}")
            return None
        
        # Get data for this target
        target_df = target_data[target_data['target_id'] == target_id].copy()
        if 'datetime' in target_df.columns and target_df['datetime'].dtype == object:
            target_df['datetime'] = pd.to_datetime(target_df['datetime'])
        
        # Get the target class
        target_class = target_df['target_class'].iloc[0] if 'target_class' in target_df.columns else 'Unknown'
        
        # Choose color based on target class
        target_colors = {
            'tank': 'darkred',
            'armoured personnel carrier': 'orangered',
            'light vehicle': 'coral'
        }
        color = target_colors.get(target_class, 'red')
        
        # Filter history and future data
        history = target_df[target_df['datetime'] <= timestamp]
        
        # Get end time for prediction
        end_time = timestamp + timedelta(seconds=prediction_duration)
        future = target_df[(target_df['datetime'] > timestamp) & (target_df['datetime'] <= end_time)]
        has_future = len(future) > 0
        
        # Get prediction data
        mean_traj = prediction['mean']
        lower_ci = prediction['lower_ci']
        upper_ci = prediction['upper_ci']
        time_points = prediction['time_points']
        speeds = prediction.get('speeds', [])
        
        # Create figure with terrain map
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Calculate bounds with padding
        x_points = np.concatenate([history['longitude'].values, mean_traj[:, 0]])
        y_points = np.concatenate([history['latitude'].values, mean_traj[:, 1]])
        
        if has_future:
            x_points = np.concatenate([x_points, future['longitude'].values])
            y_points = np.concatenate([y_points, future['latitude'].values])
        
        if blue_force_data is not None:
            x_points = np.concatenate([x_points, blue_force_data['longitude'].values])
            y_points = np.concatenate([y_points, blue_force_data['latitude'].values])
        
        lon_min, lon_max = np.min(x_points), np.max(x_points)
        lat_min, lat_max = np.min(y_points), np.max(y_points)
        
        # Add padding for better context
        lon_padding = (lon_max - lon_min) * 0.15
        lat_padding = (lat_max - lat_min) * 0.15
        lon_min -= lon_padding
        lon_max += lon_padding
        lat_min -= lat_padding
        lat_max += lat_padding
        
        # Use the provided terrain data or fall back to the instance's terrain data
        terrain = terrain_data if terrain_data is not None else self.terrain_data
        elevation = self.elevation_data
        
        if terrain is not None:
            # Create colormap for terrain
            land_use_colors = [
                '#FFFFFF',  # 0: No data
                '#1A5BAB',  # 1: Broadleaf Forest
                '#358221',  # 2: Deciduous Forest
                '#2E8B57',  # 3: Evergreen Forest
                '#52A72D',  # 4: Mixed Forest
                '#76B349',  # 5: Tree Open
                '#90EE90',  # 6: Tree Open
                '#D2B48C',  # 7: Shrub
                '#9ACD32',  # 8: Herbaceous
                '#ADFF2F',  # 9: Herbaceous with Sparse Tree/Shrub
                '#F5DEB3',  # 10: Sparse vegetation
                '#FFD700',  # 11: Cropland
                '#F4A460',  # 12: Agricultural
                '#DAA520',  # 13: Cropland / Other Vegetation
                '#2F4F4F',  # 14: Mangrove/Wetland
                '#00FFFF',  # 15: Wetland
                '#A0522D',  # 16: Bare area
                '#DEB887',  # 17: Bare area
                '#808080',  # 18: Urban
                '#FFFFFF',  # 19: Snow/Ice
                '#0000FF',  # 20: Water
            ]
            
            terrain_cmap = ListedColormap(land_use_colors)
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(0, 20)
            
            # Plot terrain - with vertical flip for correct orientation
            terrain = np.flipud(terrain)  # Flip vertically (South ↔ North)
            im = ax.imshow(terrain, cmap=terrain_cmap, norm=norm, alpha=0.7,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        aspect='auto', origin='lower', zorder=0)
            
            # Add elevation as a subtle overlay if available
            if elevation is not None:
                elev_min = np.min(elevation)
                elev_max = np.max(elevation)
                elev_norm = mcolors.Normalize(vmin=elev_min, vmax=elev_max)
                
                # Plot with low alpha and vertical flip
                elevation = np.flipud(elevation)  # Flip vertically (South ↔ North)
                ax.imshow(elevation, cmap='terrain', norm=elev_norm, alpha=0.3,
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        aspect='auto', origin='lower', zorder=1)
            
            # Add terrain legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#0000FF', label='Water'),
                Patch(facecolor='#808080', label='Urban'),
                Patch(facecolor='#358221', label='Forest'),
                Patch(facecolor='#9ACD32', label='Herbaceous'),
                Patch(facecolor='#FFD700', label='Cropland'),
                Patch(facecolor='#00FFFF', label='Wetland')
            ]
            
            terrain_legend = ax.legend(handles=legend_elements, loc='upper left', 
                                    title="Terrain Types", fontsize=10)
            ax.add_artist(terrain_legend)
        else:
            ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # Plot blue forces if provided
        if blue_force_data is not None:
            ax.scatter(blue_force_data['longitude'], blue_force_data['latitude'],
                    c='blue', s=120, marker='^', label='Blue Forces', zorder=10, edgecolor='black')
        
        # Plot target history trajectory
        ax.plot(history['longitude'], history['latitude'], '-', color=color, 
                alpha=0.8, linewidth=2.5, zorder=3, label='Past Trajectory')
        
        # Mark last known position with label
        last_pos_x = history['longitude'].iloc[-1]
        last_pos_y = history['latitude'].iloc[-1]
        ax.scatter(last_pos_x, last_pos_y, 
                c=color, s=100, marker='o', zorder=5, edgecolor='black')
        
        # Add starting time label
        ax.annotate(timestamp.strftime('%H:%M:%S'), 
                    (last_pos_x, last_pos_y), 
                    textcoords="offset points", 
                    xytext=(5, 10), 
                    ha='left',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Plot predicted trajectory with confidence intervals
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], '--', color=color, 
                linewidth=3, alpha=0.9, zorder=4, label='Predicted Trajectory')
        
        # Add final confidence ellipse only
        final_ellipse = Ellipse(
            (mean_traj[-1, 0], mean_traj[-1, 1]),
            width=upper_ci[-1, 0] - lower_ci[-1, 0],
            height=upper_ci[-1, 1] - lower_ci[-1, 1],
            color=color, alpha=0.25, zorder=2
        )
        ax.add_patch(final_ellipse)
        
        # Add final prediction point with special emphasis and label
        final_x = mean_traj[-1, 0]
        final_y = mean_traj[-1, 1]
        ax.scatter(final_x, final_y, 
                c=color, s=120, marker='x', zorder=6, 
                linewidth=2.5, edgecolor='black')
        
        # Add end time and speed label
        if len(time_points) > 0:
            end_time_text = time_points[-1].strftime('%H:%M:%S')
            speed_text = f"{speeds[-1]:.1f} m/s" if speeds and len(speeds) > 0 else ""
            
            ax.annotate(f"{end_time_text}\n{speed_text}", 
                        (final_x, final_y), 
                        textcoords="offset points", 
                        xytext=(5, 10), 
                        ha='left',
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Plot future trajectory if available
        if has_future:
            ax.plot(future['longitude'], future['latitude'], 'g-', 
                    linewidth=2.5, zorder=3, label='Actual Future Trajectory')
            ax.scatter(future['longitude'].iloc[-1], future['latitude'].iloc[-1],
                    c='green', s=120, marker='*', zorder=6, 
                    edgecolor='black')
        
        # Set title and labels
        ax.set_title(f"{target_class.title()} '{target_id}' Movement Prediction\n"
                    f"Starting at {timestamp} for {prediction_duration} seconds",
                    fontsize=14)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Set axis limits
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        
        # Add legend for trajectories
        trajectory_handles = []
        if blue_force_data is not None:
            trajectory_handles.append(plt.Line2D([], [], color='blue', marker='^', 
                                                markersize=10, linestyle='none',
                                                label='Blue Forces', markeredgecolor='black'))
        
        trajectory_handles.extend([
            plt.Line2D([], [], color=color, linestyle='-', linewidth=2.5, label='Past Trajectory'),
            plt.Line2D([], [], color=color, linestyle='--', linewidth=2.5, label='Predicted Trajectory'),
            plt.Line2D([], [], color=color, marker='o', markersize=8, linestyle='none', 
                    label='Last Known Position', markeredgecolor='black'),
            plt.Line2D([], [], color=color, marker='x', markersize=8, linestyle='none', 
                    label='Predicted Final Position', markeredgewidth=2, markeredgecolor='black')
        ])
        
        if has_future:
            trajectory_handles.extend([
                plt.Line2D([], [], color='green', linestyle='-', linewidth=2.5, label='Actual Future'),
                plt.Line2D([], [], color='green', marker='*', markersize=10, linestyle='none', 
                        label='Actual Final Position', markeredgecolor='black')
            ])
        
        trajectory_legend = ax.legend(handles=trajectory_handles, loc='upper right', 
                                    title=f"{target_class.title()} Trajectory", fontsize=10)
        
        # Add info box
        info_text = (
            f"Target: {target_id} ({target_class})\n"
            f"Last seen: {timestamp}\n"
            f"Prediction duration: {prediction_duration} seconds\n"
            f"95% confidence interval shown"
        )
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'), fontsize=10)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Detailed prediction visualization saved to {output_path}")
        
        return fig

def visualize_all_blue_red_movements(target_data, blue_force_data, terrain_data, elevation_data, output_path=None):
    """
    Create a detailed visualization of all forces and their movements on a terrain map.
    
    Args:
        target_data: DataFrame with target observations (red forces)
        blue_force_data: DataFrame with blue force locations
        terrain_data: Terrain map as numpy array
        elevation_data: Elevation map as numpy array
        output_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Get all unique timestamps
    timestamps = sorted(pd.unique(target_data['datetime']))
    
    # Get coordinate bounds with padding
    lon_min, lon_max = target_data['longitude'].min(), target_data['longitude'].max()
    lat_min, lat_max = target_data['latitude'].min(), target_data['latitude'].max()
    
    # Add padding
    lon_padding = (lon_max - lon_min) * 0.05
    lat_padding = (lat_max - lat_min) * 0.05
    lon_min -= lon_padding
    lon_max += lon_padding
    lat_min -= lat_padding
    lat_max += lat_padding
    
    # Define colors for each land use category
    land_use_colors = [
        '#FFFFFF',  # 0: No data
        '#1A5BAB',  # 1: Broadleaf Forest - dark blue-green
        '#358221',  # 2: Deciduous Forest - green
        '#2E8B57',  # 3: Evergreen Forest - sea green
        '#52A72D',  # 4: Needleleaf Forest - light green
        '#76B349',  # 5: Mixed Forest - medium green
        '#90EE90',  # 6: Tree Open - light green
        '#D2B48C',  # 7: Shrub - tan
        '#9ACD32',  # 8: Herbaceous - yellow-green
        '#ADFF2F',  # 9: Herbaceous with Sparse Tree/Shrub - green-yellow
        '#F5DEB3',  # 10: Sparse vegetation - wheat
        '#FFD700',  # 11: Cropland - gold
        '#F4A460',  # 12: Agricultural - sandy brown
        '#DAA520',  # 13: Cropland / Other Vegetation - goldenrod
        '#2F4F4F',  # 14: Mangrove/Wetland - dark slate gray
        '#00FFFF',  # 15: Wetland - cyan
        '#A0522D',  # 16: Bare area - consolidated - sienna
        '#DEB887',  # 17: Bare area - unconsolidated - burlywood
        '#808080',  # 18: Urban - dark gray
        '#FFFFFF',  # 19: Snow/Ice - white
        '#0000FF',  # 20: Water - blue
    ]
    
    terrain_cmap = ListedColormap(land_use_colors)
    
    # Add normalization for the correct range (0-20)
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(0, 20)
    
    # Plot terrain
    if terrain_data is not None:
        # NOTE: no need to flip elevation data here
        # terrain_data = np.flipud(terrain_data)
        im = ax.imshow(terrain_data, cmap=terrain_cmap, norm=norm, alpha=0.7,
                     extent=[lon_min, lon_max, lat_min, lat_max],
                     aspect='auto', zorder=0)
        
        # Add elevation as a subtle overlay if available
        if elevation_data is not None:
            # Normalize elevation data
            elev_min = np.min(elevation_data)
            elev_max = np.max(elevation_data)
            elev_norm = mcolors.Normalize(vmin=elev_min, vmax=elev_max)
            
            # Plot with low alpha
            # NOTE: no need to flip elevation data here
            # elevation_data = np.flipud(elevation_data)
            ax.imshow(elevation_data, cmap='terrain', norm=elev_norm, alpha=0.3,
                     extent=[lon_min, lon_max, lat_min, lat_max],
                     aspect='auto', zorder=1)
        
        # Add terrain legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#0000FF', label='Water'),
            Patch(facecolor='#808080', label='Urban'),
            Patch(facecolor='#358221', label='Forest'),
            Patch(facecolor='#9ACD32', label='Herbaceous'),
            Patch(facecolor='#FFD700', label='Cropland'),
            Patch(facecolor='#00FFFF', label='Wetland')
        ]
        
        # Create a separate legend for terrain
        terrain_legend = ax.legend(handles=legend_elements, loc='upper left', 
                                 title="Terrain Types", fontsize=10)
        ax.add_artist(terrain_legend)
    
    # Plot blue force positions
    if blue_force_data is not None:
        ax.scatter(blue_force_data['longitude'], blue_force_data['latitude'],
                 c='blue', s=120, marker='^', zorder=10, edgecolor='black')
        
        # Add labels for blue forces if 'name' column exists
        if 'name' in blue_force_data.columns:
            for _, row in blue_force_data.iterrows():
                ax.annotate(row['name'],
                          (row['longitude'], row['latitude']),
                          xytext=(5, 5),
                          textcoords="offset points",
                          fontsize=9,
                          color='blue',
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Target color map by class
    target_colors = {
        'tank': 'darkred',
        'armoured personnel carrier': 'orangered',
        'light vehicle': 'coral'
    }
    
    # Create separate collections for each target class
    target_collections = {}
    
    # Get unique target IDs
    target_ids = target_data['target_id'].unique()
    
    # Plot trajectories for each target
    for target_id in target_ids:
        # Get data for this target
        target_df = target_data[target_data['target_id'] == target_id].sort_values('datetime')
        
        # Get target class
        target_class = target_df['target_class'].iloc[0] if 'target_class' in target_df.columns else 'Unknown'
        color = target_colors.get(target_class, 'red')
        
        # Plot trajectory
        line, = ax.plot(target_df['longitude'], target_df['latitude'], '-', 
                      color=color, alpha=0.8, linewidth=2, zorder=3)
        
        # Add to collections
        if target_class not in target_collections:
            target_collections[target_class] = []
        target_collections[target_class].append(line)
        
        # Mark last position
        ax.scatter(target_df['longitude'].iloc[-1], target_df['latitude'].iloc[-1],
                 c=color, s=80, marker='o', zorder=5, edgecolor='black')
        
        # Add target ID label
        ax.annotate(target_id,
                   (target_df['longitude'].iloc[-1], target_df['latitude'].iloc[-1]),
                   xytext=(5, 5),
                   textcoords="offset points",
                   fontsize=9,
                   color='black',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round'))
    
    # Create legend handles
    legend_handles = []
    
    # Add Blue Forces to legend
    if blue_force_data is not None:
        legend_handles.append(plt.Line2D([], [], color='blue', marker='^', 
                                       markersize=10, linestyle='none',
                                       label='Blue Forces', markeredgecolor='black'))
    
    # Add target classes to legend
    for target_class, color in target_colors.items():
        if target_class in target_collections:
            legend_handles.append(plt.Line2D([], [], color=color, marker='o',
                                           markersize=8, linestyle='-',
                                           label=f'{target_class.title()}', 
                                           markeredgecolor='black'))
    
    # Add legend
    if legend_handles:
        forces_legend = ax.legend(handles=legend_handles, loc='upper right', 
                                title="Forces", fontsize=10)
    
    # Set title and labels
    last_time = target_data['datetime'].max()
    first_time = target_data['datetime'].min()
    ax.set_title(f'Nova Scotia Battlefield - Force Movements\n'
               f'Time Period: {first_time} to {last_time}',
               fontsize=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Set axis limits
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    
    # Add info box
    info_text = (
        f"Time Period: {first_time} to {last_time}\n"
        f"Red Forces: {len(target_ids)}\n"
        f"Blue Forces: {len(blue_force_data) if blue_force_data is not None else 0}\n"
        f"Nova Scotia Battlefield"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'), fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Battlefield visualization saved to {output_path}")
    
    return fig

def load_and_process_data(target_csv="data/red_sightings.csv", 
                         blue_csv="data/blue_locations.csv",
                         terrain_path="adapted_data/terrain_map.npy",
                         elevation_path="adapted_data/elevation_map.npy"):
    """Load and process data files."""
    # Load target data
    target_df = pd.read_csv(target_csv)
    print(f"Loaded target data with {len(target_df)} entries")
    
    # Convert datetime column
    if 'datetime' in target_df.columns:
        target_df['datetime'] = pd.to_datetime(target_df['datetime'])
    
    # Load blue force data
    blue_df = pd.read_csv(blue_csv)
    print(f"Loaded blue force data with {len(blue_df)} entries")
    
    # Load terrain and elevation if available
    terrain_data = None
    elevation_data = None
    
    if os.path.exists(terrain_path):
        terrain_data = np.load(terrain_path)
        print(f"Loaded terrain data with shape {terrain_data.shape}")
    else:
        print(f"Terrain data not found at {terrain_path}")
    
    if os.path.exists(elevation_path):
        elevation_data = np.load(elevation_path)
        print(f"Loaded elevation data with shape {elevation_data.shape}")
    else:
        print(f"Elevation data not found at {elevation_path}")
    
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
    
    # Configure device with fallback to CPU if CUDA fails
    if config is None:
        config = {}
    
    # Try CUDA first, fall back to CPU if issues arise
    try:
        if torch.cuda.is_available():
            config['device'] = 'cuda'
            # Test CUDA availability with a small tensor operation
            test_tensor = torch.zeros(1).cuda()
            del test_tensor  # Clean up the test tensor
            print("Using CUDA for training")
        else:
            config['device'] = 'cpu'
            print("CUDA not available, using CPU for training")
    except Exception as e:
        config['device'] = 'cpu'
        print(f"Error setting up CUDA: {str(e)}")
        print("Falling back to CPU for training")
    
    # Initialize predictor with safe device setting
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

def predict_test_set(predictor, test_data_path, output_path, prediction_duration=3600, max_confidence_threshold=None):
    """
    Run inference on a test set and save predictions to a file.
    
    Args:
        predictor: TargetMovementPredictor instance
        test_data_path: Path to test data CSV
        output_path: Path to save prediction results
        prediction_duration: How far to predict into the future (seconds)
        max_confidence_threshold: Maximum confidence interval size for early stopping
        
    Returns:
        DataFrame with prediction results
    """
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
            prediction_duration,
            max_confidence_threshold=max_confidence_threshold
        )
        
        if prediction is None or len(prediction['mean']) == 0:
            print(f"Could not generate prediction for target {target_id}")
            continue
        
        # Get target class
        target_class = target_data['target_class'].iloc[0] if 'target_class' in target_data.columns else 'Unknown'
        
        # Extract predicted coordinates and time points
        for i, (time_point, mean_pos, lower_ci, upper_ci) in enumerate(zip(
            prediction['time_points'], 
            prediction['mean'], 
            prediction['lower_ci'], 
            prediction['upper_ci']
        )):
            # Get speed if available
            speed = prediction.get('speeds', [0.0])[i] if i < len(prediction.get('speeds', [])) else 0.0
            
            # Calculate confidence interval sizes
            lon_ci_size = upper_ci[0] - lower_ci[0]
            lat_ci_size = upper_ci[1] - lower_ci[1]
            
            results.append({
                'target_id': target_id,
                'target_class': target_class,
                'prediction_step': i + 1,
                'timestamp': time_point,
                'predicted_longitude': mean_pos[0],
                'predicted_latitude': mean_pos[1],
                'longitude_lower_ci': lower_ci[0],
                'longitude_upper_ci': upper_ci[0],
                'latitude_lower_ci': lower_ci[1],
                'latitude_upper_ci': upper_ci[1],
                'longitude_ci_size': lon_ci_size,
                'latitude_ci_size': lat_ci_size,
                'speed_m_s': speed,
                'time_since_last_seen_s': (time_point - last_seen_time).total_seconds(),
                'last_seen_time': last_seen_time
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    results_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print(f"Total predictions: {len(results_df)} for {results_df['target_id'].nunique()} unique targets")
    
    return results_df

def visualize_all_forces(target_csv="data/red_sightings.csv", 
                        blue_csv="data/blue_locations.csv",
                        terrain_path="adapted_data/terrain_map.npy", 
                        elevation_path="adapted_data/elevation_map.npy",
                        output_path="visualizations/all_forces_movement.png"):
    """Create a visualization of all forces and their movements."""
    # Load data
    data = load_and_process_data(
        target_csv=target_csv,
        blue_csv=blue_csv,
        terrain_path=terrain_path,
        elevation_path=elevation_path
    )
    
    # Visualize all forces
    return visualize_all_blue_red_movements(
        data['target_data'],
        data['blue_force_data'],
        data['terrain_data'],
        data['elevation_data'],
        output_path=output_path
    )

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
    
    # Set paths
    target_csv = os.path.join(args.data_dir, "red_sightings.csv")
    blue_csv = os.path.join(args.data_dir, "blue_locations.csv")
    terrain_path = "adapted_data/terrain_map.npy"
    elevation_path = "adapted_data/elevation_map.npy"
    
    
    # Execute the selected mode
    if args.mode == 'train':
        # Train mode
        print(f"Starting training pipeline with data from {args.data_dir}")
        run_training_pipeline(
            target_csv=target_csv,
            blue_csv=blue_csv,
            terrain_path=terrain_path,
            elevation_path=elevation_path,
            output_dir=args.output_dir
        )
        
    elif args.mode == 'predict':
        # Prediction mode
        print(f"Loading model from {args.model_path} for prediction")
        predictor = TargetMovementPredictor(
            terrain_data_path=terrain_path,
            elevation_data_path=elevation_path
        )
        
        # Load model
        if not predictor.load_model(args.model_path):
            print(f"Error: Could not load model from {args.model_path}")
            exit(1)
        
        # Load data
        data = load_and_process_data(
            target_csv=target_csv,
            blue_csv=blue_csv,
            terrain_path=terrain_path,
            elevation_path=elevation_path
        )
        
        # Select a timestamp
        timestamps = sorted(data['target_data']['datetime'].unique())
        mid_idx = len(timestamps) // 2
        timestamp = timestamps[mid_idx]
        print(f"Selected timestamp for prediction: {timestamp}")
        
        # Make predictions for all targets
        output_path = os.path.join(args.output_dir, "all_targets_prediction.png")
        predictor.visualize_all_targets_predictions(
            data['target_data'],
            timestamp,
            prediction_duration=args.prediction_duration,
            blue_force_data=data['blue_force_data'],
            output_path=output_path
        )
        
        # If specific target ID provided, create detailed visualization for it
        if args.target_id:
            print(f"Creating detailed visualization for target {args.target_id}")
            target_output_path = os.path.join(args.output_dir, f"target_{args.target_id}_prediction.png")
            predictor.visualize_prediction_with_terrain(
                data['target_data'],
                args.target_id,
                timestamp,
                prediction_duration=args.prediction_duration,
                blue_force_data=data['blue_force_data'],
                output_path=target_output_path
            )
        
    elif args.mode == 'visualize':
        # Visualization mode (terrain/elevation/positions)
        print(f"Creating visualizations in {args.output_dir}")
        
        # Load data
        data = load_and_process_data(
            target_csv=target_csv,
            blue_csv=blue_csv,
            terrain_path=terrain_path,
            elevation_path=elevation_path
        )
        
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
        terrain_data = data['terrain_data']
        if data['terrain_data'] is not None:
            plt.figure(figsize=(12, 10))
            terrain_data = np.flipud(terrain_data)        # Flip vertically (South ↔ North)
            plt.imshow(terrain_data, cmap=terrain_cmap, origin='upper')
            plt.colorbar(label='Terrain Type')
            plt.title('Terrain Map')
            plt.savefig(os.path.join(args.output_dir, "terrain_map.png"), dpi=300)
            plt.close()
        
        # Visualize elevation
        if data['elevation_data'] is not None:
            plt.figure(figsize=(12, 10))
            #
            elevation_data = data['elevation_data']
            elevation_data = np.flipud(elevation_data)        # Flip vertically (South ↔ North)
            plt.imshow(elevation_data, cmap='terrain', origin='upper')
            plt.colorbar(label='Elevation (m)')
            plt.title('Elevation Map')
            plt.savefig(os.path.join(args.output_dir, "elevation_map.png"), dpi=300)
            plt.close()
        
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
        plt.close()
        
        # Create full visualization with movements
        visualize_all_forces(
            target_csv=target_csv,
            blue_csv=blue_csv,
            terrain_path=terrain_path,
            elevation_path=elevation_path,
            output_path=os.path.join(args.output_dir, "all_forces_movement.png")
        )
        
        print(f"Visualizations saved to {args.output_dir}")
        
    elif args.mode == 'test':
        # Test mode - run on test set
        print(f"Loading model from {args.model_path} for testing")
        predictor = TargetMovementPredictor(
            terrain_data_path=terrain_path,
            elevation_data_path=elevation_path
        )
        
        # Load model
        if not predictor.load_model(args.model_path):
            print(f"Error: Could not load model from {args.model_path}")
            exit(1)
        
        # Generate predictions
        output_path = os.path.join(args.output_dir, "test_predictions.csv")
        predictions_df = predict_test_set(
            predictor,
            target_csv,
            output_path,
            prediction_duration=args.prediction_duration
        )
        
        # Create a visualization of a few test predictions
        print("Creating sample visualizations...")
        
        # Load data
        data = load_and_process_data(
            target_csv=target_csv,
            blue_csv=blue_csv,
            terrain_path=terrain_path,
            elevation_path=elevation_path
        )
        
        # Select a few targets to visualize (3 at most)
        target_ids = predictions_df['target_id'].unique()[:3] if len(predictions_df) > 0 else []
        
        for target_id in target_ids:
            # Get timestamp
            timestamp = predictions_df[predictions_df['target_id'] == target_id]['last_seen_time'].iloc[0]
            
            # Create visualization
            target_output_path = os.path.join(args.output_dir, f"test_target_{target_id}_prediction.png")
            predictor.visualize_prediction_with_terrain(
                data['target_data'],
                target_id,
                timestamp,
                prediction_duration=args.prediction_duration,
                blue_force_data=data['blue_force_data'],
                output_path=target_output_path
            )
        
        print(f"Testing completed. Results saved to {args.output_dir}")