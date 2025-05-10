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
from datetime import timedelta
import time

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

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class JammerTransformerDecoder(nn.Module):
    """
    Transformer decoder that fuses positional history via causal self-attention
    and terrain/vision embeddings via cross-attention.
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
        self.out_mean   = nn.Linear(hidden_dim, output_dim)
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
        V = K                            # share for cross-attn
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
        mean   = self.out_mean(H_dec)
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

class JammerPositionDataset(Dataset):
    """
    Dataset for training the jammer position predictor.
    """
    def __init__(self, 
                jammer_data,
                terrain_data=None, 
                elevation_data=None, 
                sequence_length=10, 
                prediction_horizon=5, 
                stride=1,
                terrain_window_size=32):
        """
        Initialize the dataset.
        
        Args:
            jammer_data: DataFrame with jammer observations
            terrain_data: Terrain map as numpy array
            elevation_data: Elevation map as numpy array
            sequence_length: Number of timesteps to use as input
            prediction_horizon: Number of future timesteps to predict
            stride: Stride between sequences
            terrain_window_size: Size of terrain patch to extract
        """
        self.jammer_data = jammer_data
        self.terrain_data = terrain_data
        self.elevation_data = elevation_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.terrain_window_size = terrain_window_size
        
        # Process data
        self.process_data()
        
    def process_data(self):
        # Group by jammer ID
        jammer_groups = self.jammer_data.groupby('id')
        
        self.sequences = []
        
        for jammer_id, group in jammer_groups:
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Skip if not enough data points
            if len(group) < self.sequence_length + self.prediction_horizon:
                continue
            
            # Extract features
            x_coords = group['x_coord'].values
            y_coords = group['y_coord'].values
            
            if 'power' in group.columns:
                power = group['power'].values
            else:
                power = np.ones_like(x_coords) * 50  # Default power
                
            if 'range' in group.columns:
                jammer_range = group['range'].values
            else:
                jammer_range = np.ones_like(x_coords) * 1000  # Default range
                
            if 'direction' in group.columns:
                direction = group['direction'].values
            else:
                direction = np.zeros_like(x_coords)  # Default direction
            
            # Add jammer type as one-hot encoding if available
            jammer_types = None
            if 'jammer_type' in group.columns:
                # Get unique jammer types
                all_types = self.jammer_data['jammer_type'].unique()
                
                # Create one-hot encoding
                jammer_types = np.zeros((len(group), len(all_types)))
                for i, jtype in enumerate(all_types):
                    jammer_types[:, i] = (group['jammer_type'] == jtype).astype(int)
            
            # Create sequences
            for i in range(0, len(group) - self.sequence_length - self.prediction_horizon + 1, self.stride):
                # Input sequence
                input_sequence = []
                
                for j in range(i, i + self.sequence_length):
                    features = [x_coords[j], y_coords[j], power[j], jammer_range[j], direction[j]]
                    
                    # Add jammer type if available
                    if jammer_types is not None:
                        features.extend(jammer_types[j])
                    
                    input_sequence.append(features)
                
                # Target sequence
                target_sequence = []
                
                for j in range(i + self.sequence_length, i + self.sequence_length + self.prediction_horizon):
                    target_sequence.append([x_coords[j], y_coords[j]])
                
                # Extract terrain patch if available
                terrain_patch = None
                if self.terrain_data is not None and self.elevation_data is not None:
                    # Get the latest position
                    latest_x = int(x_coords[i + self.sequence_length - 1])
                    latest_y = int(y_coords[i + self.sequence_length - 1])
                    
                    # Extract terrain patch centered on latest position
                    half_window = self.terrain_window_size // 2
                    
                    # Ensure within bounds
                    x_min = max(0, latest_x - half_window)
                    x_max = min(self.terrain_data.shape[0], latest_x + half_window)
                    y_min = max(0, latest_y - half_window)
                    y_max = min(self.terrain_data.shape[1], latest_y + half_window)
                    
                    # Extract patches
                    terrain_patch = self.terrain_data[x_min:x_max, y_min:y_max]
                    elevation_patch = self.elevation_data[x_min:x_max, y_min:y_max]
                    
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
                
                # Add to sequences
                self.sequences.append({
                    'input': np.array(input_sequence, dtype=np.float32),
                    'target': np.array(target_sequence, dtype=np.float32),
                    'terrain': terrain_patch,
                    'jammer_id': jammer_id,
                    'start_idx': i
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Prepare input
        input_seq = torch.tensor(sequence['input'], dtype=torch.float32)
        
        # Prepare target
        target_seq = torch.tensor(sequence['target'], dtype=torch.float32)
        
        # Prepare terrain if available
        terrain = None
        if sequence['terrain'] is not None:
            terrain = torch.tensor(sequence['terrain'], dtype=torch.float32)
        
        return {
            'input': input_seq,
            'target': target_seq,
            'terrain': terrain,
            'jammer_id': sequence['jammer_id'],
            'start_idx': sequence['start_idx']
        }

class JammerPositionPredictor:
    """
    Main class for predicting jammer positions.
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
        
    def build_models(self, input_dim, output_dim=2):
        """
        Build the models.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features (default: 2 for x, y coordinates)
        """
        # Transformer model for sequence prediction
        self.transformer_model = JammerTransformerDecoder(
            input_dim=input_dim,
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
    
    def prepare_data(self, jammer_data, train_ratio=0.8):
        """
        Prepare data for training and validation.

        Args:
            jammer_data: DataFrame with jammer observations
            train_ratio: Ratio of data to use for training

        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        # Create dataset
        dataset = JammerPositionDataset(
            jammer_data=jammer_data,
            terrain_data=self.terrain_data,
            elevation_data=self.elevation_data,
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

        # Determine input dimension including terrain features if in use
        sample = train_dataset[0]
        input_dim = sample['input'].shape[1]

        if (self.config.get('use_terrain', False)
            and self.terrain_data is not None
            and self.elevation_data is not None):
            input_dim += self.config['terrain_feature_dim']

        # Build models with the correct input dimension
        self.build_models(input_dim)

        return train_loader, val_loader

    
    def setup_feature_scaling(self, dataset):
        """
        Set up feature scaling for normalization.
        
        Args:
            dataset: Dataset to use for scaling
        """
        # Collect all input and output data
        all_inputs = []
        all_outputs = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            inputs = sample['input'].numpy()
            outputs = sample['target'].numpy()
            
            # Flatten the sequences
            all_inputs.append(inputs.reshape(-1, inputs.shape[-1]))
            all_outputs.append(outputs.reshape(-1, outputs.shape[-1]))
        
        all_inputs = np.vstack(all_inputs)
        all_outputs = np.vstack(all_outputs)
        
        # Create and fit scalers
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        self.input_scaler.fit(all_inputs)
        self.output_scaler.fit(all_outputs)
    
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
            patience=5,
            # verbose=True
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
                
                # Forward pass
                optimizer.zero_grad()
                
                # If terrain features available, concatenate with the last input
                if terrain_features is not None:
                    # Extract last input from sequence
                    last_inputs = inputs_norm[:, -1, :]
                    
                    # Concatenate with terrain features
                    enhanced_inputs = torch.cat([last_inputs, terrain_features], dim=1)
                    
                    # Expand to match sequence length
                    expanded_features = enhanced_inputs.unsqueeze(1).expand(-1, inputs_norm.size(1), -1)
                    
                    # Replace inputs with enhanced inputs
                    inputs_norm = expanded_features
                
                # Forward pass through transformer
                pred_mean, pred_logvar = self.transformer_model(inputs_norm)
                
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
                    
                    # If terrain features available, concatenate with the last input
                    if terrain_features is not None:
                        # Extract last input from sequence
                        last_inputs = inputs_norm[:, -1, :]
                        
                        # Concatenate with terrain features
                        enhanced_inputs = torch.cat([last_inputs, terrain_features], dim=1)
                        
                        # Expand to match sequence length
                        expanded_features = enhanced_inputs.unsqueeze(1).expand(-1, inputs_norm.size(1), -1)
                        
                        # Replace inputs with enhanced inputs
                        inputs_norm = expanded_features
                    
                    # Forward pass through transformer
                    pred_mean, pred_logvar = self.transformer_model(inputs_norm)
                    
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
                self.save_model('best_model.pt')
        
        # Calculate total training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model
        self.load_model('best_model.pt')
        
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
        
        # Combine with terrain features if available
        if terrain_features is not None:
            # Extract last input from sequence
            last_inputs = input_norm[:, -1, :]
            
            # Concatenate with terrain features
            enhanced_inputs = torch.cat([last_inputs, terrain_features], dim=1)
            
            # Expand to match sequence length
            expanded_features = enhanced_inputs.unsqueeze(1).expand(-1, input_norm.size(1), -1)
            
            # Replace inputs with enhanced inputs
            input_norm = expanded_features
        
        # Predict with the model
        with torch.no_grad():
            pred_mean, pred_logvar = self.transformer_model(input_norm)
            
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
        
        checkpoint = torch.load(filename, map_location=self.config['device'])
        
        # Update config
        self.config.update(checkpoint['config'])
        
        # Load scalers
        self.input_scaler = checkpoint['input_scaler']
        self.output_scaler = checkpoint['output_scaler']
        
        # Initialize models if not already initialized
        if self.transformer_model is None:
            # Determine input dimension
            # This is a placeholder - in practice, you need to know the input dimension
            input_dim = next(iter(checkpoint['transformer_state'].values())).shape[1]
            
            self.build_models(input_dim)
        
        # Load model weights
        self.transformer_model.load_state_dict(checkpoint['transformer_state'])
        
        if 'terrain_state' in checkpoint and self.terrain_model is not None:
            self.terrain_model.load_state_dict(checkpoint['terrain_state'])
        
        print(f"Model loaded from {filename}")
        return True
    
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
                
                # Collect results
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(predictions['mean'])
                all_lower_ci.append(predictions['lower_ci'])
                all_upper_ci.append(predictions['upper_ci'])
        
        # Concatenate results
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
                             title="Jammer Position Predictions", save_path=None):
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
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
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

def load_and_process_jammer_data(jammer_csv="synthetic_data/jammer_observations.csv"):
    """
    Load and process jammer data from CSV.
    
    Args:
        jammer_csv: Path to jammer observations CSV
        
    Returns:
        DataFrame with processed jammer data
    """
    if not os.path.exists(jammer_csv):
        print(f"Jammer CSV not found: {jammer_csv}")
        return None
    
    # Load data
    jammer_df = pd.read_csv(jammer_csv)
    
    # Convert timestamp to datetime
    jammer_df['timestamp'] = pd.to_datetime(jammer_df['timestamp'])
    
    # Sort by ID and timestamp
    jammer_df = jammer_df.sort_values(['id', 'timestamp'])
    
    print(f"Loaded {len(jammer_df)} jammer observations for {jammer_df['id'].nunique()} jammers")
    
    return jammer_df

def run_training_pipeline(
    jammer_csv="synthetic_data/jammer_observations.csv",
    terrain_path="simulation_data/terrain_map.npy",
    elevation_path="simulation_data/elevation_map.npy",
    output_dir="models",
    config=None
):
    """
    Run the full training pipeline.
    
    Args:
        jammer_csv: Path to jammer observations CSV
        terrain_path: Path to terrain map file
        elevation_path: Path to elevation map file
        output_dir: Directory to save models and plots
        config: Model configuration dictionary (optional)
        
    Returns:
        Trained predictor object
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load jammer data
    jammer_df = load_and_process_jammer_data(jammer_csv)
    if jammer_df is None:
        return None
    
    # Initialize predictor
    predictor = JammerPositionPredictor(
        config=config,
        terrain_data_path=terrain_path,
        elevation_data_path=elevation_path
    )
    
    # Prepare data
    train_loader, val_loader = predictor.prepare_data(jammer_df)
    
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
            title=f"Jammer {batch['jammer_id'][0]} Predictions",
            save_path=os.path.join(output_dir, f"prediction_sample_{i}.png")
        )
    
    # Save model
    predictor.save_model(os.path.join(output_dir, "jammer_predictor_model.pt"))
    
    print(f"\nTraining pipeline completed. Model and visualizations saved to {output_dir}")
    
    return predictor

if __name__ == "__main__":
    # Example usage
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
    
    predictor = run_training_pipeline(config=config)
