import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from tqdm import tqdm
import time
import math
import random

class RFSignalEncoder(nn.Module):
    """
    Neural network to encode RF signals into a latent representation.
    Takes signal characteristics as input and produces an embedding.
    """
    def __init__(self, input_dim, hidden_dims=[64, 128], latent_dim=32, dropout=0.1):
        super(RFSignalEncoder, self).__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)

class TerrainAwareNavigation(nn.Module):
    """
    Neural network for terrain-aware navigation based on RF signals.
    Takes RF signal embeddings and terrain features as input
    and produces navigation commands.
    """
    def __init__(self, rf_dim, terrain_dim, hidden_dims=[128, 64], output_dim=2, dropout=0.1):
        super(TerrainAwareNavigation, self).__init__()
        
        # Input dimensions
        self.input_dim = rf_dim + terrain_dim
        
        # Build network
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer for navigation commands
        # Typically direction (theta) and speed
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, rf_embedding, terrain_features):
        # Concatenate inputs
        x = torch.cat([rf_embedding, terrain_features], dim=1)
        
        # Pass through network
        x = self.network(x)
        
        # Output layer
        output = self.output_layer(x)
        
        return output

class NavigationController(nn.Module):
    """
    Combined model for RF signal-based navigation.
    Integrates RF signal encoding and navigation control.
    """
    def __init__(self, 
                rf_input_dim, 
                terrain_dim, 
                rf_hidden_dims=[64, 128], 
                rf_latent_dim=32,
                nav_hidden_dims=[128, 64], 
                output_dim=2, 
                dropout=0.1):
        super(NavigationController, self).__init__()
        
        # RF signal encoder
        self.rf_encoder = RFSignalEncoder(
            input_dim=rf_input_dim,
            hidden_dims=rf_hidden_dims,
            latent_dim=rf_latent_dim,
            dropout=dropout
        )
        
        # Navigation controller
        self.navigator = TerrainAwareNavigation(
            rf_dim=rf_latent_dim,
            terrain_dim=terrain_dim,
            hidden_dims=nav_hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
        
    def forward(self, rf_signal, terrain_features):
        # Encode RF signal
        rf_embedding = self.rf_encoder(rf_signal)
        
        # Generate navigation commands
        nav_commands = self.navigator(rf_embedding, terrain_features)
        
        return nav_commands, rf_embedding

class TerrainFeatureExtractor(nn.Module):
    """
    CNN to extract features from terrain data.
    Takes terrain patches as input and produces a feature vector.
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

class RFNavigationDataset(Dataset):
    """
    Dataset for training the RF signal navigation model.
    """
    def __init__(self, 
                rf_data,
                terrain_data=None, 
                elevation_data=None,
                window_size=32,
                target_type='avoid'):
        """
        Initialize the dataset.
        
        Args:
            rf_data: DataFrame with RF signal data
            terrain_data: Terrain map as numpy array
            elevation_data: Elevation map as numpy array
            window_size: Size of terrain patch to extract
            target_type: 'avoid' to move away from jammers, 'approach' to move toward them
        """
        self.rf_data = rf_data
        self.terrain_data = terrain_data
        self.elevation_data = elevation_data
        self.window_size = window_size
        self.target_type = target_type
        
        # Process data
        self.process_data()
        
    def process_data(self):
        """Process the data to create samples."""
        # Get unique timestamps
        timestamps = self.rf_data['timestamp'].unique()
        
        self.samples = []
        
        for timestamp in timestamps:
            # Get RF data for this timestamp
            rf_data_at_time = self.rf_data[self.rf_data['timestamp'] == timestamp]
            
            # Skip if no data
            if len(rf_data_at_time) == 0:
                continue
                
            # Process each observation at this timestamp
            for _, row in rf_data_at_time.iterrows():
                # Extract RF features
                rf_features = self.extract_rf_features(row)
                
                # Extract position
                x_pos = row['x_coord']
                y_pos = row['y_coord']
                
                # Generate navigation target based on jammer position and type
                jammer_x = row['jammer_x_coord']
                jammer_y = row['jammer_y_coord']
                
                # Calculate direction and distance to jammer
                dx = jammer_x - x_pos
                dy = jammer_y - y_pos
                distance = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)  # In radians
                
                # Set target based on target_type
                if self.target_type == 'avoid':
                    # Target is to move away from jammer
                    target_angle = (angle + np.pi) % (2 * np.pi)  # Opposite direction
                    target_speed = min(1.0, 0.5 + 0.5 * (1.0 - min(1.0, distance / 1000.0)))  # Faster when closer
                else:
                    # Target is to move toward jammer
                    target_angle = angle
                    target_speed = min(1.0, 0.5 + 0.5 * min(1.0, distance / 1000.0))  # Faster when farther
                
                # Extract terrain patch if available
                terrain_patch = None
                if self.terrain_data is not None and self.elevation_data is not None:
                    terrain_patch = self.extract_terrain_patch(x_pos, y_pos)
                
                # Add to samples
                self.samples.append({
                    'rf_features': rf_features,
                    'terrain_patch': terrain_patch,
                    'target_angle': target_angle,
                    'target_speed': target_speed,
                    'jammer_distance': distance,
                    'jammer_angle': angle,
                    'x_pos': x_pos,
                    'y_pos': y_pos,
                    'jammer_x': jammer_x,
                    'jammer_y': jammer_y,
                    'timestamp': timestamp
                })
    
    def extract_rf_features(self, row):
        """Extract RF signal features from a data row."""
        features = []
        
        # Basic features: signal strength, frequency metrics
        if 'signal_strength' in row:
            features.append(row['signal_strength'])
        else:
            # Calculate based on jammer effect and distance
            jammer_effect = row.get('jamming_effect', 0.0)
            distance = row.get('distance', 100.0)  # Default if not available
            signal_strength = jammer_effect * (1000.0 / max(10.0, distance))
            features.append(signal_strength)
        
        # Frequency data if available
        if 'freq_low' in row and 'freq_high' in row:
            freq_low = row['freq_low']
            freq_high = row['freq_high']
            freq_mid = (freq_low + freq_high) / 2
            freq_range = freq_high - freq_low
            
            features.extend([freq_low, freq_high, freq_mid, freq_range])
        else:
            # Default values
            features.extend([100.0, 1000.0, 550.0, 900.0])
        
        # Add angle metrics if available
        if 'angle' in row and 'direction' in row:
            jammer_angle = row['angle']
            jammer_direction = row['direction']
            
            # Calculate angular difference (useful for directional jammers)
            angle_diff = abs(jammer_direction - jammer_angle)
            angle_diff = min(angle_diff, 360 - angle_diff)
            
            features.append(jammer_angle / 360.0)  # Normalize to [0, 1]
            features.append(jammer_direction / 360.0)  # Normalize to [0, 1]
            features.append(angle_diff / 180.0)  # Normalize to [0, 1]
        else:
            # Default values
            features.extend([0.5, 0.5, 0.5])
        
        # Add jammer power if available
        if 'power' in row:
            features.append(row['power'] / 100.0)  # Normalize assuming max power is 100
        else:
            features.append(0.5)  # Default
        
        return np.array(features, dtype=np.float32)
    
    def extract_terrain_patch(self, x_pos, y_pos):
        """Extract a terrain patch centered at the given position."""
        # Convert to integer indices
        x_idx = int(x_pos)
        y_idx = int(y_pos)
        
        # Ensure within bounds
        half_window = self.window_size // 2
        
        # Calculate bounds with clamping
        x_min = max(0, x_idx - half_window)
        x_max = min(self.terrain_data.shape[0], x_idx + half_window)
        y_min = max(0, y_idx - half_window)
        y_max = min(self.terrain_data.shape[1], y_idx + half_window)
        
        # Extract patches
        terrain_patch = self.terrain_data[x_min:x_max, y_min:y_max]
        elevation_patch = self.elevation_data[x_min:x_max, y_min:y_max]
        
        # Pad if necessary
        if terrain_patch.shape[0] < self.window_size or terrain_patch.shape[1] < self.window_size:
            padded_terrain = np.zeros((self.window_size, self.window_size))
            padded_elevation = np.zeros((self.window_size, self.window_size))
            
            # Copy available data
            h, w = terrain_patch.shape
            padded_terrain[:h, :w] = terrain_patch
            padded_elevation[:h, :w] = elevation_patch
            
            terrain_patch = padded_terrain
            elevation_patch = padded_elevation
        
        # Stack terrain and elevation
        return np.stack([terrain_patch, elevation_patch], axis=0).astype(np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Prepare inputs
        rf_features = torch.tensor(sample['rf_features'], dtype=torch.float32)
        
        # Prepare terrain if available
        terrain_patch = None
        if sample['terrain_patch'] is not None:
            terrain_patch = torch.tensor(sample['terrain_patch'], dtype=torch.float32)
        
        # Prepare targets
        targets = torch.tensor(
            [sample['target_angle'], sample['target_speed']], 
            dtype=torch.float32
        )
        
        return {
            'rf_features': rf_features,
            'terrain_patch': terrain_patch,
            'targets': targets,
            'jammer_distance': sample['jammer_distance'],
            'jammer_angle': sample['jammer_angle'],
            'x_pos': sample['x_pos'],
            'y_pos': sample['y_pos'],
            'jammer_x': sample['jammer_x'],
            'jammer_y': sample['jammer_y']
        }

class RFNavigationController:
    """
    Main class for RF signal-based navigation.
    """
    def __init__(self, 
                config=None,
                terrain_data_path=None, 
                elevation_data_path=None):
        """
        Initialize the navigation controller.
        
        Args:
            config: Configuration dictionary
            terrain_data_path: Path to terrain map file
            elevation_data_path: Path to elevation map file
        """
        # Default configuration
        default_config = {
            'rf_hidden_dims': [64, 128],
            'rf_latent_dim': 32,
            'nav_hidden_dims': [128, 64],
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_epochs': 50,
            'terrain_feature_dim': 32,
            'use_terrain': True,
            'nav_mode': 'avoid',  # 'avoid' or 'approach'
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
        self.model = None
        self.terrain_model = None
        self.feature_scaler = None
        
    def build_models(self, rf_input_dim):
        """
        Build the models.
        
        Args:
            rf_input_dim: Input dimension for RF features
        """
        # Terrain feature extractor
        if self.config['use_terrain'] and self.terrain_data is not None and self.elevation_data is not None:
            self.terrain_model = TerrainFeatureExtractor(
                input_channels=2,  # Terrain + elevation
                output_dim=self.config['terrain_feature_dim']
            )
            terrain_dim = self.config['terrain_feature_dim']
        else:
            terrain_dim = 0
        
        # Navigation controller
        self.model = NavigationController(
            rf_input_dim=rf_input_dim,
            terrain_dim=terrain_dim,
            rf_hidden_dims=self.config['rf_hidden_dims'],
            rf_latent_dim=self.config['rf_latent_dim'],
            nav_hidden_dims=self.config['nav_hidden_dims'],
            output_dim=2,  # Direction and speed
            dropout=self.config['dropout']
        )
        
        # Move models to device
        self.model.to(self.config['device'])
        if self.terrain_model is not None:
            self.terrain_model.to(self.config['device'])
    
    def prepare_data(self, rf_data, train_ratio=0.8):
        """
        Prepare data for training and validation.
        
        Args:
            rf_data: DataFrame with RF signal data
            train_ratio: Ratio of data to use for training
            
        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        # Create dataset
        dataset = RFNavigationDataset(
            rf_data=rf_data,
            terrain_data=self.terrain_data,
            elevation_data=self.elevation_data,
            target_type=self.config['nav_mode']
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
        
        # Build models
        sample = train_dataset[0]
        rf_input_dim = sample['rf_features'].shape[0]
        
        self.build_models(rf_input_dim)
        
        return train_loader, val_loader
    
    def setup_feature_scaling(self, dataset):
        """
        Set up feature scaling for normalization.
        
        Args:
            dataset: Dataset to use for scaling
        """
        # Collect all RF feature data
        all_features = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            features = sample['rf_features'].numpy()
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        
        # Create and fit scaler
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(all_features)
    
    def normalize_features(self, features):
        """
        Normalize features using the fitted scaler.
        
        Args:
            features: RF features
            
        Returns:
            Normalized features
        """
        # Convert to numpy if tensor
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
            features_shape = features.shape
            
            # Check if we need reshaping
            if len(features_shape) > 2:
                # Reshape for scaling
                features_flat = features_np.reshape(-1, features_np.shape[-1])
                
                # Scale
                features_scaled = self.feature_scaler.transform(features_flat)
                
                # Reshape back
                features_scaled = features_scaled.reshape(features_shape)
            else:
                # Simple scaling for 2D tensor
                features_scaled = self.feature_scaler.transform(features_np)
            
            # Convert back to tensor
            features_scaled = torch.tensor(features_scaled, dtype=features.dtype, device=features.device)
        else:
            features_scaled = self.feature_scaler.transform(features)
        
        return features_scaled
    
    def circular_loss(self, pred_angle, true_angle):
        """
        Calculate loss for circular angle predictions.
        
        Args:
            pred_angle: Predicted angles in radians
            true_angle: True angles in radians
            
        Returns:
            Loss value
        """
        # Convert angles to unit vectors
        pred_x = torch.cos(pred_angle)
        pred_y = torch.sin(pred_angle)
        
        true_x = torch.cos(true_angle)
        true_y = torch.sin(true_angle)
        
        # Calculate squared error in vector space
        loss = ((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2) / 2
        
        return loss.mean()
    
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
            list(self.model.parameters()) + 
            (list(self.terrain_model.parameters()) if self.terrain_model is not None else []),
            lr=self.config['learning_rate']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # MSE loss for speed
        mse_loss = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_angle_loss': [],
            'train_speed_loss': [],
            'val_angle_loss': [],
            'val_speed_loss': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
        # Training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            if self.terrain_model is not None:
                self.terrain_model.train()
            
            train_loss = 0.0
            train_angle_loss = 0.0
            train_speed_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch in train_pbar:
                # Get batch
                rf_features = batch['rf_features'].to(self.config['device'])
                targets = batch['targets'].to(self.config['device'])
                
                # Normalize RF features
                rf_features = self.normalize_features(rf_features)
                
                # Extract target components
                target_angle = targets[:, 0]  # Angle in radians
                target_speed = targets[:, 1]  # Speed in [0, 1]
                
                # Process terrain if available
                terrain_features = None
                if self.terrain_model is not None and 'terrain_patch' in batch and batch['terrain_patch'] is not None:
                    terrain_patch = batch['terrain_patch'].to(self.config['device'])
                    terrain_features = self.terrain_model(terrain_patch)
                else:
                    # Create dummy terrain features of zeros
                    terrain_features = torch.zeros((rf_features.size(0), self.config['terrain_feature_dim']), 
                                                device=self.config['device'])
                
                # Forward pass
                optimizer.zero_grad()
                pred, _ = self.model(rf_features, terrain_features)
                
                # Extract predictions
                pred_angle = pred[:, 0]  # Angle in radians
                pred_speed = pred[:, 1]  # Speed in [0, 1]
                
                # Calculate losses
                angle_loss = self.circular_loss(pred_angle, target_angle)
                speed_loss = mse_loss(pred_speed, target_speed)
                
                # Total loss (angle is more important)
                loss = angle_loss + 0.5 * speed_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                train_loss += loss.item()
                train_angle_loss += angle_loss.item()
                train_speed_loss += speed_loss.item()
                train_pbar.set_postfix({
                    'loss': f"{train_loss/(train_pbar.n+1):.4f}",
                    'angle': f"{train_angle_loss/(train_pbar.n+1):.4f}",
                    'speed': f"{train_speed_loss/(train_pbar.n+1):.4f}"
                })
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            train_angle_loss /= len(train_loader)
            train_speed_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            if self.terrain_model is not None:
                self.terrain_model.eval()
            
            val_loss = 0.0
            val_angle_loss = 0.0
            val_speed_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for batch in val_pbar:
                    # Get batch
                    rf_features = batch['rf_features'].to(self.config['device'])
                    targets = batch['targets'].to(self.config['device'])
                    
                    # Normalize RF features
                    rf_features = self.normalize_features(rf_features)
                    
                    # Extract target components
                    target_angle = targets[:, 0]  # Angle in radians
                    target_speed = targets[:, 1]  # Speed in [0, 1]
                    
                    # Process terrain if available
                    terrain_features = None
                    if self.terrain_model is not None and 'terrain_patch' in batch and batch['terrain_patch'] is not None:
                        terrain_patch = batch['terrain_patch'].to(self.config['device'])
                        terrain_features = self.terrain_model(terrain_patch)
                    else:
                        # Create dummy terrain features of zeros
                        terrain_features = torch.zeros((rf_features.size(0), self.config['terrain_feature_dim']), 
                                                    device=self.config['device'])
                    
                    # Forward pass
                    pred, _ = self.model(rf_features, terrain_features)
                    
                    # Extract predictions
                    pred_angle = pred[:, 0]  # Angle in radians
                    pred_speed = pred[:, 1]  # Speed in [0, 1]
                    
                    # Calculate losses
                    angle_loss = self.circular_loss(pred_angle, target_angle)
                    speed_loss = mse_loss(pred_speed, target_speed)
                    
                    # Total loss
                    loss = angle_loss + 0.5 * speed_loss
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_angle_loss += angle_loss.item()
                    val_speed_loss += speed_loss.item()
                    val_pbar.set_postfix({
                        'loss': f"{val_loss/(val_pbar.n+1):.4f}",
                        'angle': f"{val_angle_loss/(val_pbar.n+1):.4f}",
                        'speed': f"{val_speed_loss/(val_pbar.n+1):.4f}"
                    })
            
            # Calculate average validation loss
            val_loss /= len(val_loader)
            val_angle_loss /= len(val_loader)
            val_speed_loss /= len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_angle_loss'].append(train_angle_loss)
            history['train_speed_loss'].append(train_speed_loss)
            history['val_angle_loss'].append(val_angle_loss)
            history['val_speed_loss'].append(val_speed_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
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
    
    def predict_navigation(self, rf_features, terrain_patch=None):
        """
        Predict navigation commands from RF features.
        
        Args:
            rf_features: RF signal features
            terrain_patch: Terrain patch (optional)
            
        Returns:
            Dictionary with navigation commands
        """
        # Set model to evaluation mode
        self.model.eval()
        if self.terrain_model is not None:
            self.terrain_model.eval()
        
        # Add batch dimension if necessary
        if len(rf_features.shape) == 1:
            rf_features = rf_features.unsqueeze(0)
        
        if terrain_patch is not None and len(terrain_patch.shape) == 3:
            terrain_patch = terrain_patch.unsqueeze(0)
        
        # Move to device
        rf_features = rf_features.to(self.config['device'])
        if terrain_patch is not None:
            terrain_patch = terrain_patch.to(self.config['device'])
        
        # Normalize features
        rf_features = self.normalize_features(rf_features)
        
        # Process terrain if available
        terrain_features = None
        if self.terrain_model is not None and terrain_patch is not None:
            terrain_features = self.terrain_model(terrain_patch)
        else:
            # Create dummy terrain features of zeros
            terrain_features = torch.zeros((rf_features.size(0), self.config['terrain_feature_dim']), 
                                          device=self.config['device'])
        
        # Make prediction
        with torch.no_grad():
            navigation, rf_embedding = self.model(rf_features, terrain_features)
            
            # Extract components
            angle = navigation[:, 0].cpu().numpy()
            speed = navigation[:, 1].cpu().numpy()
            
            # Convert angles to degrees for easier interpretation
            angle_degrees = np.degrees(angle) % 360
            
            # Remove batch dimension if only one sample
            if angle.shape[0] == 1:
                angle = angle[0]
                speed = speed[0]
                angle_degrees = angle_degrees[0]
                rf_embedding = rf_embedding[0]
            
        return {
            'angle_rad': angle,
            'angle_deg': angle_degrees,
            'speed': speed,
            'rf_embedding': rf_embedding.cpu().numpy()
        }
    
    def save_model(self, filename):
        """
        Save the model.
        
        Args:
            filename: Filename to save the model to
        """
        save_dict = {
            'model_state': self.model.state_dict(),
            'config': self.config,
            'feature_scaler': self.feature_scaler
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
        
        # Load feature scaler
        self.feature_scaler = checkpoint['feature_scaler']
        
        # Initialize models if not already initialized
        if self.model is None:
            # Determine RF input dimension from the first layer of the encoder
            first_layer = next(iter(checkpoint['model_state'].items()))
            if 'rf_encoder.encoder.0.weight' in first_layer:
                rf_input_dim = first_layer[1].shape[1]
            else:
                # Default to 9 if can't determine
                rf_input_dim = 9
            
            self.build_models(rf_input_dim)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state'])
        
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
        self.model.eval()
        if self.terrain_model is not None:
            self.terrain_model.eval()
        
        # MSE loss for speed
        mse_loss = nn.MSELoss()
        
        angle_errors = []
        speed_errors = []
        distance_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Get batch
                rf_features = batch['rf_features'].to(self.config['device'])
                targets = batch['targets'].to(self.config['device'])
                jammer_distances = batch['jammer_distance'].numpy()
                
                # Normalize RF features
                rf_features = self.normalize_features(rf_features)
                
                # Extract target components
                target_angle = targets[:, 0]  # Angle in radians
                target_speed = targets[:, 1]  # Speed in [0, 1]
                
                # Process terrain if available
                terrain_features = None
                if self.terrain_model is not None and 'terrain_patch' in batch and batch['terrain_patch'] is not None:
                    terrain_patch = batch['terrain_patch'].to(self.config['device'])
                    terrain_features = self.terrain_model(terrain_patch)
                else:
                    # Create dummy terrain features of zeros
                    terrain_features = torch.zeros((rf_features.size(0), self.config['terrain_feature_dim']), 
                                                  device=self.config['device'])
                
                # Forward pass
                pred, _ = self.model(rf_features, terrain_features)
                
                # Extract predictions
                pred_angle = pred[:, 0]  # Angle in radians
                pred_speed = pred[:, 1]  # Speed in [0, 1]
                
                # Calculate angle error (circular)
                angle_diff = torch.abs(torch.atan2(
                    torch.sin(pred_angle - target_angle),
                    torch.cos(pred_angle - target_angle)
                ))
                
                # Convert to degrees
                angle_diff_deg = angle_diff.cpu().numpy() * 180 / np.pi
                
                # Calculate speed error
                speed_error = torch.abs(pred_speed - target_speed).cpu().numpy()
                
                # Collect metrics
                angle_errors.extend(angle_diff_deg)
                speed_errors.extend(speed_error)
                
                # Group by distance
                for i in range(len(jammer_distances)):
                    distance_metrics.append({
                        'distance': jammer_distances[i],
                        'angle_error': angle_diff_deg[i],
                        'speed_error': speed_error[i]
                    })
        
        # Calculate overall metrics
        mean_angle_error = np.mean(angle_errors)
        mean_speed_error = np.mean(speed_errors)
        
        # Group metrics by distance range
        distance_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, np.inf)]
        distance_metrics_by_range = {}
        
        for min_dist, max_dist in distance_ranges:
            range_metrics = [m for m in distance_metrics if min_dist <= m['distance'] < max_dist]
            
            if range_metrics:
                range_angle_error = np.mean([m['angle_error'] for m in range_metrics])
                range_speed_error = np.mean([m['speed_error'] for m in range_metrics])
                
                distance_metrics_by_range[f"{min_dist}-{max_dist}"] = {
                    'count': len(range_metrics),
                    'angle_error': range_angle_error,
                    'speed_error': range_speed_error
                }
        
        return {
            'mean_angle_error': mean_angle_error,
            'mean_speed_error': mean_speed_error,
            'distance_metrics': distance_metrics_by_range
        }
    
    def visualize_navigation(self, rf_features, terrain_patch=None, position=None, jammer_position=None, 
                           title="Navigation Commands", save_path=None):
        """
        Visualize navigation commands.
        
        Args:
            rf_features: RF signal features
            terrain_patch: Terrain patch (optional)
            position: Current position (x, y) (optional)
            jammer_position: Jammer position (x, y) (optional)
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Get navigation commands
        nav_commands = self.predict_navigation(
            torch.tensor(rf_features, dtype=torch.float32),
            torch.tensor(terrain_patch, dtype=torch.float32) if terrain_patch is not None else None
        )
        
        # Extract commands
        angle_deg = nav_commands['angle_deg']
        speed = nav_commands['speed']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot a circle representing the navigation area
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
        ax.add_patch(circle)
        
        # Plot navigation vector
        angle_rad = np.radians(angle_deg)
        dx = np.cos(angle_rad) * speed
        dy = np.sin(angle_rad) * speed
        
        ax.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue', 
                linewidth=2, zorder=10)
        
        # Plot additional markers for original position and jammer if provided
        if position is not None and jammer_position is not None:
            # Calculate relative positions (normalized)
            max_dist = 2000  # Max distance to show
            
            dx_jammer = (jammer_position[0] - position[0]) / max_dist
            dy_jammer = (jammer_position[1] - position[1]) / max_dist
            
            jammer_dist = np.sqrt(dx_jammer**2 + dy_jammer**2)
            
            # Clamp to visible area
            if jammer_dist > 1:
                dx_jammer /= jammer_dist
                dy_jammer /= jammer_dist
            
            # Plot jammer position
            ax.scatter(dx_jammer, dy_jammer, c='red', s=100, marker='x', zorder=5)
            
            # Add a line from center to jammer
            ax.plot([0, dx_jammer], [0, dy_jammer], 'r--', alpha=0.5)
            
            # Add jammer info
            ax.annotate(f"Jammer\n({jammer_dist*max_dist:.0f}m)", 
                       (dx_jammer, dy_jammer), 
                       xytext=(dx_jammer*1.1, dy_jammer*1.1),
                       bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Set aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # Add coordinate axes
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add markers for directions
        ax.text(1.1, 0, "E", fontsize=12)
        ax.text(-1.1, 0, "W", fontsize=12)
        ax.text(0, 1.1, "N", fontsize=12)
        ax.text(0, -1.1, "S", fontsize=12)
        
        # Add current position marker
        ax.scatter(0, 0, c='green', s=100, marker='o', zorder=10)
        
        # Add navigation info
        ax.set_title(title)
        ax.text(0.05, 0.95, f"Direction: {angle_deg:.1f}°\nSpeed: {speed:.2f}", 
               transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', fc='white', alpha=0.7), 
               verticalalignment='top')
        
        # Mode info
        mode_label = "AVOID" if self.config['nav_mode'] == 'avoid' else "APPROACH"
        ax.text(0.95, 0.05, f"Mode: {mode_label}", 
               transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', fc='white', alpha=0.7), 
               horizontalalignment='right', verticalalignment='bottom')
        
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
        
        # Plot component losses
        ax2.plot(epochs, history['train_angle_loss'], 'b-', label='Train Angle Loss')
        ax2.plot(epochs, history['val_angle_loss'], 'r-', label='Val Angle Loss')
        ax2.plot(epochs, history['train_speed_loss'], 'b--', label='Train Speed Loss')
        ax2.plot(epochs, history['val_speed_loss'], 'r--', label='Val Speed Loss')
        ax2.axvline(x=history['best_epoch'] + 1, color='g', linestyle='--', label='Best Model')
        ax2.set_title('Component Losses')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def generate_rf_signal_data(jammer_csv="synthetic_data/jammer_observations.csv", 
                           effects_csv="synthetic_data/jamming_effects.csv",
                           output_csv="synthetic_data/rf_signal_data.csv"):
    """
    Generate synthetic RF signal data from jammer and effects data.
    
    Args:
        jammer_csv: Path to jammer observations CSV
        effects_csv: Path to jamming effects CSV
        output_csv: Path to save the RF signal data
        
    Returns:
        DataFrame with RF signal data
    """
    # Check if files exist
    if not os.path.exists(jammer_csv):
        print(f"Jammer CSV not found: {jammer_csv}")
        return None
    
    if not os.path.exists(effects_csv):
        print(f"Effects CSV not found: {effects_csv}")
        return None
    
    # Load data
    jammers_df = pd.read_csv(jammer_csv)
    effects_df = pd.read_csv(effects_csv)
    
    # Convert timestamps to datetime
    jammers_df['timestamp'] = pd.to_datetime(jammers_df['timestamp'])
    effects_df['timestamp'] = pd.to_datetime(effects_df['timestamp'])
    
    # Merge jammer data with effects
    rf_data = pd.merge(
        effects_df,
        jammers_df[['id', 'timestamp', 'x_coord', 'y_coord', 'power', 'range', 'angle', 'direction', 'freq_low', 'freq_high']],
        left_on=['jammer_id', 'timestamp'],
        right_on=['id', 'timestamp'],
        how='left',
        suffixes=('', '_jammer')
    )
    
    # Rename columns for clarity
    rf_data = rf_data.rename(columns={
        'x_coord': 'jammer_x_coord',
        'y_coord': 'jammer_y_coord',
        'id_jammer': 'jammer_id'
    })
    
    # In case of merge issues, get original jammer coordinates
    if 'jammer_x_coord' not in rf_data.columns or 'jammer_y_coord' not in rf_data.columns:
        # Try to extract coordinates from original jammer data
        jammer_coords = {}
        for _, row in jammers_df.iterrows():
            jammer_id = row['id']
            timestamp = row['timestamp']
            x = row['x_coord']
            y = row['y_coord']
            jammer_coords[(jammer_id, timestamp)] = (x, y)
        
        # Add to dataframe
        rf_data['jammer_x_coord'] = rf_data.apply(
            lambda row: jammer_coords.get((row['jammer_id'], row['timestamp']), (0, 0))[0], 
            axis=1
        )
        rf_data['jammer_y_coord'] = rf_data.apply(
            lambda row: jammer_coords.get((row['jammer_id'], row['timestamp']), (0, 0))[1],
            axis=1
        )
    
    # Calculate signal strength based on jamming effect and distance
    rf_data['signal_strength'] = rf_data['jamming_effect'] * (1000.0 / rf_data['distance'].clip(10.0))
    
    # Calculate angle from entity to jammer (for RF direction finding)
    rf_data['signal_angle'] = np.arctan2(
        rf_data['jammer_y_coord'] - rf_data['y_coord'],
        rf_data['jammer_x_coord'] - rf_data['x_coord']
    )
    
    # Convert angle to degrees
    rf_data['signal_angle_deg'] = np.degrees(rf_data['signal_angle']) % 360
    
    # Save to CSV
    rf_data.to_csv(output_csv, index=False)
    print(f"Generated {len(rf_data)} RF signal observations, saved to {output_csv}")
    
    return rf_data

def run_navigation_training_pipeline(
    rf_data_csv="synthetic_data/rf_signal_data.csv",
    terrain_path="simulation_data/terrain_map.npy",
    elevation_path="simulation_data/elevation_map.npy",
    output_dir="models",
    config=None
):
    """
    Run the full training pipeline for the RF navigation model.
    
    Args:
        rf_data_csv: Path to RF signal data CSV
        terrain_path: Path to terrain map file
        elevation_path: Path to elevation map file
        output_dir: Directory to save models and plots
        config: Model configuration dictionary (optional)
        
    Returns:
        Trained navigation controller object
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if RF data file exists, generate if not
    if not os.path.exists(rf_data_csv):
        print(f"RF data file not found, generating from jammer and effects data...")
        rf_data = generate_rf_signal_data(output_csv=rf_data_csv)
        
        if rf_data is None:
            print("Failed to generate RF data")
            return None
    else:
        # Load RF data
        rf_data = pd.read_csv(rf_data_csv)
        rf_data['timestamp'] = pd.to_datetime(rf_data['timestamp'])
    
    # Initialize navigation controller
    if config is None:
        # Default config
        config = {
            'rf_hidden_dims': [64, 128],
            'rf_latent_dim': 32,
            'nav_hidden_dims': [128, 64],
            'dropout': 0.1,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_epochs': 50,
            'terrain_feature_dim': 32,
            'use_terrain': True,
            'nav_mode': 'avoid',  # 'avoid' or 'approach'
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    controller = RFNavigationController(
        config=config,
        terrain_data_path=terrain_path,
        elevation_data_path=elevation_path
    )
    
    # Prepare data
    train_loader, val_loader = controller.prepare_data(rf_data)
    
    # Train model
    history = controller.train(train_loader, val_loader)
    
    # Save training history plot
    controller.visualize_training_history(
        history,
        save_path=os.path.join(output_dir, "rf_nav_training_history.png")
    )
    
    # Evaluate on validation set
    metrics = controller.evaluate(val_loader)
    print("\nValidation Metrics:")
    for key, value in metrics.items():
        if key != 'distance_metrics':
            print(f"  {key}: {value}")
        else:
            print("  Distance-based metrics:")
            for dist_range, dist_metrics in value.items():
                print(f"    {dist_range}m: angle error = {dist_metrics['angle_error']:.2f}°, "
                      f"speed error = {dist_metrics['speed_error']:.3f}, "
                      f"samples = {dist_metrics['count']}")
    
    # Visualize sample navigation commands
    for i, batch in enumerate(val_loader):
        if i >= 3:  # Show 3 examples
            break
            
        rf_features = batch['rf_features'][0].numpy()
        terrain_patch = batch['terrain_patch'][0].numpy() if batch['terrain_patch'] is not None else None
        position = (batch['x_pos'][0].item(), batch['y_pos'][0].item())
        jammer_position = (batch['jammer_x'][0].item(), batch['jammer_y'][0].item())
        
        controller.visualize_navigation(
            rf_features, terrain_patch, position, jammer_position,
            title=f"Navigation Command Example {i+1}",
            save_path=os.path.join(output_dir, f"rf_nav_example_{i}.png")
        )
    
    # Save model
    controller.save_model(os.path.join(output_dir, "rf_navigation_model.pt"))
    
    print(f"\nTraining pipeline completed. Model and visualizations saved to {output_dir}")
    
    return controller

def run_dual_model_simulation(
    jammer_predictor_model="models/jammer_predictor_model.pt",
    rf_navigation_model="models/rf_navigation_model.pt",
    terrain_path="simulation_data/terrain_map.npy",
    elevation_path="simulation_data/elevation_map.npy",
    output_dir="simulations"
):
    """
    Run a simulation using both the jammer predictor and RF navigation models.
    
    Args:
        jammer_predictor_model: Path to trained jammer predictor model
        rf_navigation_model: Path to trained RF navigation model
        terrain_path: Path to terrain map file
        elevation_path: Path to elevation map file
        output_dir: Directory to save simulation results
        
    Returns:
        Dictionary with simulation results
    """
    from matplotlib.animation import FuncAnimation
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load jammer predictor model
    from jammer_position_predictor import JammerPositionPredictor
    jammer_predictor = JammerPositionPredictor(
        terrain_data_path=terrain_path,
        elevation_data_path=elevation_path
    )
    jammer_predictor.load_model(jammer_predictor_model)
    
    # Load RF navigation model
    rf_navigator = RFNavigationController(
        terrain_data_path=terrain_path,
        elevation_data_path=elevation_path
    )
    rf_navigator.load_model(rf_navigation_model)
    
    # Load terrain data
    terrain_data = np.load(terrain_path)
    elevation_data = np.load(elevation_path)
    
    # Create a simulation environment
    print("Setting up simulation environment...")
    
    # Define simulation parameters
    sim_steps = 50  # Number of simulation steps
    jammer_start_pos = (500, 500)  # Starting position of the jammer
    drone_start_pos = (200, 200)  # Starting position of the drone
    
    # Load jammer trajectory history (last 10 observations)
    jammer_history = []
    
    # Create synthetic jammer history with some randomness
    for i in range(10):
        jammer_history.append([
            jammer_start_pos[0] - i * 10 + random.uniform(-5, 5),
            jammer_start_pos[1] - i * 5 + random.uniform(-5, 5),
            50.0,  # Power
            1500.0,  # Range
            45.0,   # Direction
        ])
    
    # Reverse to have oldest observations first
    jammer_history.reverse()
    
    # Current positions
    jammer_pos = jammer_start_pos
    drone_pos = drone_start_pos
    
    # History of positions
    jammer_positions = [jammer_pos]
    drone_positions = [drone_pos]
    
    # Run simulation
    print("Running simulation...")
    for step in tqdm(range(sim_steps)):
        # 1. Predict future jammer positions
        jammer_input = torch.tensor(jammer_history, dtype=torch.float32)
        
        # Extract terrain patch around current jammer position
        half_window = 16
        x_min = max(0, int(jammer_pos[0]) - half_window)
        x_max = min(terrain_data.shape[0], int(jammer_pos[0]) + half_window)
        y_min = max(0, int(jammer_pos[1]) - half_window)
        y_max = min(terrain_data.shape[1], int(jammer_pos[1]) + half_window)
        
        # Extract patches
        t_patch = terrain_data[x_min:x_max, y_min:y_max]
        e_patch = elevation_data[x_min:x_max, y_min:y_max]
        
        # Pad if necessary
        if t_patch.shape[0] < 32 or t_patch.shape[1] < 32:
            padded_t = np.zeros((32, 32))
            padded_e = np.zeros((32, 32))
            
            h, w = t_patch.shape
            padded_t[:h, :w] = t_patch
            padded_e[:h, :w] = e_patch
            
            t_patch = padded_t
            e_patch = padded_e
        
        # Stack terrain and elevation for terrain feature extraction
        terrain_patch = np.stack([t_patch, e_patch], axis=0)
        terrain_patch = torch.tensor(terrain_patch, dtype=torch.float32)
        
        # Get jammer prediction
        jammer_pred = jammer_predictor.predict(jammer_input, terrain_patch)
        
        # Extract predicted position (first step)
        predicted_jammer_pos = jammer_pred['mean'][0]
        
        # 2. Generate RF signal observation from current jammer and drone positions
        # Calculate distance
        dx = jammer_pos[0] - drone_pos[0]
        dy = jammer_pos[1] - drone_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Generate signal strength based on distance
        signal_strength = min(1.0, 1000.0 / max(10.0, distance))
        
        # Calculate angle from drone to jammer
        signal_angle = math.atan2(dy, dx)
        
        # Create RF signal features
        rf_features = np.array([
            signal_strength,  # Signal strength
            100.0, 1000.0, 550.0, 900.0,  # Frequency data
            signal_angle / (2 * math.pi),  # Normalized angle
            0.5, 0.5,  # Direction placeholders
            0.5  # Power placeholder
        ], dtype=np.float32)
        
        # 3. Get navigation command from RF navigation model
        # Extract terrain patch around drone
        x_min = max(0, int(drone_pos[0]) - half_window)
        x_max = min(terrain_data.shape[0], int(drone_pos[0]) + half_window)
        y_min = max(0, int(drone_pos[1]) - half_window)
        y_max = min(terrain_data.shape[1], int(drone_pos[1]) + half_window)
        
        # Extract patches
        t_patch = terrain_data[x_min:x_max, y_min:y_max]
        e_patch = elevation_data[x_min:x_max, y_min:y_max]
        
        # Pad if necessary
        if t_patch.shape[0] < 32 or t_patch.shape[1] < 32:
            padded_t = np.zeros((32, 32))
            padded_e = np.zeros((32, 32))
            
            h, w = t_patch.shape
            padded_t[:h, :w] = t_patch
            padded_e[:h, :w] = e_patch
            
            t_patch = padded_t
            e_patch = padded_e
        
        # Stack terrain and elevation for terrain feature extraction
        drone_terrain_patch = np.stack([t_patch, e_patch], axis=0)
        drone_terrain_patch = torch.tensor(drone_terrain_patch, dtype=torch.float32)
        
        # Get navigation command
        nav_command = rf_navigator.predict_navigation(
            torch.tensor(rf_features, dtype=torch.float32),
            drone_terrain_patch
        )
        
        # 4. Move drone based on navigation command
        angle_rad = nav_command['angle_rad']
        speed = nav_command['speed']
        
        # Calculate movement vector
        move_distance = speed * 20.0  # Scale speed to distance
        move_x = move_distance * math.cos(angle_rad)
        move_y = move_distance * math.sin(angle_rad)
        
        # Update drone position
        new_drone_x = drone_pos[0] + move_x
        new_drone_y = drone_pos[1] + move_y
        
        # Ensure drone stays within terrain bounds
        new_drone_x = max(0, min(terrain_data.shape[0] - 1, new_drone_x))
        new_drone_y = max(0, min(terrain_data.shape[1] - 1, new_drone_y))
        
        drone_pos = (new_drone_x, new_drone_y)
        
        # 5. Move jammer based on pattern and prediction error
        # Jammer could follow a predefined pattern with some noise
        # Here, we simulate a jammer moving along a path with noise
        
        # Add some randomness to jammer movement
        jammer_move_angle = (step * 5) % 360  # Circular pattern
        jammer_move_dist = 5.0  # Fixed distance per step
        
        # Convert angle to radians
        jammer_move_angle_rad = jammer_move_angle * math.pi / 180.0
        
        # Calculate movement vector
        jammer_move_x = jammer_move_dist * math.cos(jammer_move_angle_rad)
        jammer_move_y = jammer_move_dist * math.sin(jammer_move_angle_rad)
        
        # Update jammer position
        new_jammer_x = jammer_pos[0] + jammer_move_x
        new_jammer_y = jammer_pos[1] + jammer_move_y
        
        # Ensure jammer stays within terrain bounds
        new_jammer_x = max(0, min(terrain_data.shape[0] - 1, new_jammer_x))
        new_jammer_y = max(0, min(terrain_data.shape[1] - 1, new_jammer_y))
        
        jammer_pos = (new_jammer_x, new_jammer_y)
        
        # 6. Update histories
        jammer_positions.append(jammer_pos)
        drone_positions.append(drone_pos)
        
        # Update jammer history for next prediction
        jammer_history.append([
            jammer_pos[0],
            jammer_pos[1],
            50.0,  # Power
            1500.0,  # Range
            45.0,   # Direction
        ])
        jammer_history = jammer_history[1:]  # Remove oldest
    
    # Create visualization of the simulation
    print("Creating simulation visualization...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot terrain
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
    terrain_cmap = mcolors.ListedColormap(terrain_colors)
    ax.imshow(terrain_data.T, origin='lower', cmap=terrain_cmap, 
             vmin=0, vmax=len(terrain_colors)-1, alpha=0.7)
    
    # Plot jammer and drone trajectories
    jammer_x, jammer_y = zip(*jammer_positions)
    drone_x, drone_y = zip(*drone_positions)
    
    ax.plot(jammer_x, jammer_y, 'r-', linewidth=2, label='Jammer Path')
    ax.plot(drone_x, drone_y, 'b-', linewidth=2, label='Drone Path')
    
    # Mark start and end positions
    ax.scatter(jammer_x[0], jammer_y[0], c='red', s=100, marker='o', label='Jammer Start')
    ax.scatter(jammer_x[-1], jammer_y[-1], c='red', s=100, marker='x', label='Jammer End')
    
    ax.scatter(drone_x[0], drone_y[0], c='blue', s=100, marker='o', label='Drone Start')
    ax.scatter(drone_x[-1], drone_y[-1], c='blue', s=100, marker='x', label='Drone End')
    
    # Add title and legend
    ax.set_title('Jammer Prediction and RF Navigation Simulation')
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    static_path = os.path.join(output_dir, "simulation_path.png")
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    
    # Create an animation of the simulation
    def animate(i):
        ax.clear()
        
        # Plot terrain
        ax.imshow(terrain_data.T, origin='lower', cmap=terrain_cmap, 
                 vmin=0, vmax=len(terrain_colors)-1, alpha=0.7)
        
        # Plot paths up to current frame
        if i > 0:
            ax.plot(jammer_x[:i], jammer_y[:i], 'r-', linewidth=2, alpha=0.7)
            ax.plot(drone_x[:i], drone_y[:i], 'b-', linewidth=2, alpha=0.7)
        
        # Current positions
        ax.scatter(jammer_x[i], jammer_y[i], c='red', s=100, marker='*')
        ax.scatter(drone_x[i], drone_y[i], c='blue', s=100, marker='*')
        
        # Draw a line showing current distance
        ax.plot([jammer_x[i], drone_x[i]], [jammer_y[i], drone_y[i]], 'k--', alpha=0.3)
        
        # Calculate current distance
        current_dist = math.sqrt((jammer_x[i] - drone_x[i])**2 + (jammer_y[i] - drone_y[i])**2)
        
        # Add status text
        ax.text(0.02, 0.98, 
               f"Step: {i}\nDistance: {current_dist:.1f}m\n"
               f"Jammer: ({jammer_x[i]:.1f}, {jammer_y[i]:.1f})\n"
               f"Drone: ({drone_x[i]:.1f}, {drone_y[i]:.1f})",
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add title
        ax.set_title('Jammer Prediction and RF Navigation Simulation')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Keep same axis limits for consistency
        ax.set_xlim(0, terrain_data.shape[0])
        ax.set_ylim(0, terrain_data.shape[1])
    
    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(jammer_positions), interval=200, blit=False)
    
    # Save animation
    animation_path = os.path.join(output_dir, "simulation_animation.mp4")
    ani.save(animation_path, dpi=200, fps=10)
    
    plt.close()
    
    print(f"Simulation complete. Results saved to {output_dir}")
    
    return {
        'static_path': static_path,
        'animation_path': animation_path,
        'jammer_positions': jammer_positions,
        'drone_positions': drone_positions
    }

if __name__ == "__main__":
    # Example usage
    
    # Generate RF signal data
    rf_data = generate_rf_signal_data()
    
    # Train RF navigation model
    config = {
        'rf_hidden_dims': [64, 128],
        'rf_latent_dim': 32,
        'nav_hidden_dims': [128, 64],
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'num_epochs': 50,
        'terrain_feature_dim': 32,
        'use_terrain': True,
        'nav_mode': 'avoid',  # 'avoid' or 'approach'
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    controller = run_navigation_training_pipeline(config=config)
    
    # Optional: Run simulation with both models
    # run_dual_model_simulation()
