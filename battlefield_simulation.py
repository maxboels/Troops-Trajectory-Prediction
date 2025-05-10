import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LightSource
import random
import math
import os
import datetime
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class BattlefieldSimulation:
    """
    Battlefield simulation environment for target tracking prediction.
    
    This simulation integrates terrain data (elevation and land use) with
    target and blue force movement data to create a realistic battlefield
    environment for training predictive models.
    """
    
    # Terrain types and their characteristics
    TERRAIN_TYPES = {
        0: {'name': 'Water', 'color': 'blue', 'passable': False, 'speed_factor': 0.0},
        1: {'name': 'Urban', 'color': 'gray', 'passable': True, 'speed_factor': 0.8},
        2: {'name': 'Agricultural', 'color': 'yellow', 'passable': True, 'speed_factor': 0.9},
        3: {'name': 'Forest', 'color': 'darkgreen', 'passable': True, 'speed_factor': 0.7},
        4: {'name': 'Grassland', 'color': 'lightgreen', 'passable': True, 'speed_factor': 1.0},
        5: {'name': 'Barren', 'color': 'brown', 'passable': True, 'speed_factor': 0.9},
        6: {'name': 'Wetland', 'color': 'cyan', 'passable': True, 'speed_factor': 0.6},
        7: {'name': 'Snow/Ice', 'color': 'white', 'passable': True, 'speed_factor': 0.5}
    }
    
    # Target classes and their characteristics
    TARGET_CLASSES = {
        'infantry': {'speed': 5, 'terrain_adaptability': 0.9},
        'light_vehicle': {'speed': 15, 'terrain_adaptability': 0.7},
        'heavy_vehicle': {'speed': 10, 'terrain_adaptability': 0.5},
        'uav': {'speed': 40, 'terrain_adaptability': 0.1},  # Less affected by terrain
        'civilian': {'speed': 3, 'terrain_adaptability': 0.8}
    }
    
    # Blue force classes and their characteristics
    BLUE_FORCE_CLASSES = {
        'infantry_squad': {'speed': 4, 'detection_range': 2},
        'mechanized_patrol': {'speed': 20, 'detection_range': 3},
        'recon_team': {'speed': 8, 'detection_range': 5},
        'command_post': {'speed': 1, 'detection_range': 4},
        'uav_surveillance': {'speed': 30, 'detection_range': 8}
    }
    
    def __init__(self, size=(100, 100), seed=None):
        """
        Initialize the battlefield simulation.
        
        Args:
            size: Tuple (width, height) for the terrain grid size
            seed: Random seed for reproducibility
        """
        self.size = size
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize terrain and elevation maps
        self.terrain_map = np.ones(size, dtype=int) * 4  # Default to grassland
        self.elevation_map = np.zeros(size, dtype=float)
        
        # Storage for entities and observations
        self.targets = {}
        self.blue_forces = {}
        self.target_observations = []
        self.blue_force_observations = []
        
        # Simulation time
        self.current_time = datetime.datetime.now()
        self.simulation_steps = 0
        
        # Scaling factors for mapping between coordinates and grid indices
        self.x_scale = 1.0
        self.y_scale = 1.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        
        # Optional: Cached terrain features for faster simulation
        self._terrain_speeds = None
        self._elevation_gradients = None
    
    def load_terrain_data(self, terrain_data_path, elevation_data_path=None):
        """
        Load terrain and elevation data from files.
        
        Args:
            terrain_data_path: Path to terrain type map (NumPy .npy file)
            elevation_data_path: Path to elevation map (NumPy .npy file)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load terrain data
            if terrain_data_path.endswith('.npy'):
                terrain_map = np.load(terrain_data_path)
                self.terrain_map = terrain_map
                print(f"Loaded terrain map with shape {terrain_map.shape}")
            else:
                print(f"Unsupported terrain data format: {terrain_data_path}")
                return False
            
            # Load elevation data if provided
            if elevation_data_path:
                if elevation_data_path.endswith('.npy'):
                    elevation_map = np.load(elevation_data_path)
                    self.elevation_map = elevation_map
                    print(f"Loaded elevation map with shape {elevation_map.shape}")
                else:
                    print(f"Unsupported elevation data format: {elevation_data_path}")
                    return False
            
            # Set size based on loaded data
            self.size = self.terrain_map.shape
            
            # Precompute terrain speeds for faster simulation
            self._precompute_terrain_features()
            
            return True
            
        except Exception as e:
            print(f"Error loading terrain data: {e}")
            return False
    
    def _precompute_terrain_features(self):
        """Precompute terrain features for faster simulation"""
        # Compute speed factors for all terrain types
        self._terrain_speeds = np.zeros_like(self.terrain_map, dtype=float)
        for i in range(self.terrain_map.shape[0]):
            for j in range(self.terrain_map.shape[1]):
                terrain_type = self.terrain_map[i, j]
                if terrain_type in self.TERRAIN_TYPES:
                    self._terrain_speeds[i, j] = self.TERRAIN_TYPES[terrain_type]['speed_factor']
                else:
                    self._terrain_speeds[i, j] = 1.0  # Default speed factor
        
        # Compute elevation gradients (slope)
        if self.elevation_map is not None and self.elevation_map.size > 0:
            dx, dy = np.gradient(self.elevation_map)
            self._elevation_gradients = np.sqrt(dx**2 + dy**2)
            
            # Adjust speed factors based on slope
            # Steeper slopes reduce speed
            max_slope_factor = 0.8  # Maximum speed reduction from slope
            normalized_gradients = self._elevation_gradients / np.percentile(self._elevation_gradients, 95)
            normalized_gradients = np.clip(normalized_gradients, 0, 1)
            slope_factors = 1.0 - (normalized_gradients * max_slope_factor)
            
            # Multiply terrain speed by slope factor
            self._terrain_speeds *= slope_factors
    
    def load_observation_data(self, target_csv, blue_force_csv=None):
        """
        Load target and blue force observation data from CSV files.
        
        Args:
            target_csv: Path to target observations CSV
            blue_force_csv: Path to blue force observations CSV (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load target observations
            target_df = pd.read_csv(target_csv)
            
            # Convert timestamps to datetime objects
            if 'timestamp' in target_df.columns:
                target_df['timestamp'] = pd.to_datetime(target_df['timestamp'])
            
            # Store as observations
            self.target_observations = target_df.to_dict('records')
            print(f"Loaded {len(self.target_observations)} target observations")
            
            # Extract unique targets
            target_ids = target_df['id'].unique()
            for target_id in target_ids:
                target_rows = target_df[target_df['id'] == target_id]
                target_class = target_rows['target_class'].iloc[0] if 'target_class' in target_df.columns else 'unknown'
                
                # Create target entity
                self.targets[target_id] = {
                    'id': target_id,
                    'class': target_class,
                    'observations': target_rows.to_dict('records'),
                    'position': (
                        target_rows['x_coord'].iloc[-1], 
                        target_rows['y_coord'].iloc[-1]
                    ) if not target_rows.empty else (0, 0),
                    'characteristics': self.TARGET_CLASSES.get(
                        target_class, self.TARGET_CLASSES.get('infantry', {})
                    )
                }
            
            # Load blue force observations if provided
            if blue_force_csv:
                blue_force_df = pd.read_csv(blue_force_csv)
                
                # Convert timestamps to datetime objects
                if 'timestamp' in blue_force_df.columns:
                    blue_force_df['timestamp'] = pd.to_datetime(blue_force_df['timestamp'])
                
                # Store as observations
                self.blue_force_observations = blue_force_df.to_dict('records')
                print(f"Loaded {len(self.blue_force_observations)} blue force observations")
                
                # Extract unique blue forces
                force_ids = blue_force_df['id'].unique()
                for force_id in force_ids:
                    force_rows = blue_force_df[blue_force_df['id'] == force_id]
                    force_class = force_rows['force_class'].iloc[0] if 'force_class' in blue_force_df.columns else 'unknown'
                    
                    # Create blue force entity
                    self.blue_forces[force_id] = {
                        'id': force_id,
                        'class': force_class,
                        'observations': force_rows.to_dict('records'),
                        'position': (
                            force_rows['x_coord'].iloc[-1],
                            force_rows['y_coord'].iloc[-1]
                        ) if not force_rows.empty else (0, 0),
                        'characteristics': self.BLUE_FORCE_CLASSES.get(
                            force_class, self.BLUE_FORCE_CLASSES.get('infantry_squad', {})
                        )
                    }
            
            # Set up coordinate scaling for mapping between data and grid
            self._setup_coordinate_scaling()
            
            return True
            
        except Exception as e:
            print(f"Error loading observation data: {e}")
            return False
    
    def _setup_coordinate_scaling(self):
        """Set up scaling factors for mapping between data coordinates and grid indices"""
        # Get the range of coordinates
        x_coords = [obs['x_coord'] for obs in self.target_observations + self.blue_force_observations]
        y_coords = [obs['y_coord'] for obs in self.target_observations + self.blue_force_observations]
        
        if not x_coords or not y_coords:
            print("Warning: No observation data available for coordinate scaling")
            return
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add a small buffer
        buffer = 0.05  # 5% buffer
        x_range = (x_max - x_min) * (1 + buffer)
        y_range = (y_max - y_min) * (1 + buffer)
        
        x_min -= x_range * buffer / 2
        y_min -= y_range * buffer / 2
        
        # Calculate scaling factors
        grid_width, grid_height = self.size
        self.x_scale = (grid_width - 1) / x_range if x_range > 0 else 1.0
        self.y_scale = (grid_height - 1) / y_range if y_range > 0 else 1.0
        
        # Calculate offsets
        self.x_offset = x_min
        self.y_offset = y_min
        
        print(f"Coordinate scaling: x_scale={self.x_scale:.4f}, y_scale={self.y_scale:.4f}")
        print(f"Coordinate offsets: x_offset={self.x_offset:.1f}, y_offset={self.y_offset:.1f}")
    
    def data_to_grid(self, x, y):
        """
        Convert data coordinates to grid indices.
        
        Args:
            x: X coordinate in data space
            y: Y coordinate in data space
            
        Returns:
            Tuple of (grid_x, grid_y) indices
        """
        grid_x = int((x - self.x_offset) * self.x_scale)
        grid_y = int((y - self.y_offset) * self.y_scale)
        
        # Clamp to grid bounds
        grid_x = max(0, min(self.size[0] - 1, grid_x))
        grid_y = max(0, min(self.size[1] - 1, grid_y))
        
        return grid_x, grid_y
    
    def grid_to_data(self, grid_x, grid_y):
        """
        Convert grid indices to data coordinates.
        
        Args:
            grid_x: X index in grid space
            grid_y: Y index in grid space
            
        Returns:
            Tuple of (x, y) coordinates in data space
        """
        x = (grid_x / self.x_scale) + self.x_offset
        y = (grid_y / self.y_scale) + self.y_offset
        
        return x, y
    
    def build_trajectory_datasets(self, test_ratio=0.2, window_size=5, prediction_horizons=[1, 3, 5], 
                                 include_terrain=True, include_blue_forces=True):
        """
        Build training and testing datasets for trajectory prediction.
        
        Args:
            test_ratio: Portion of targets to use for testing
            window_size: Number of past observations to use as input
            prediction_horizons: List of time steps ahead to predict
            include_terrain: Whether to include terrain features
            include_blue_forces: Whether to include blue force positions
            
        Returns:
            Dictionary of trajectory datasets
        """
        print("Building trajectory datasets...")
        
        # Group observations by target
        target_trajectories = {}
        
        for target_id, target in self.targets.items():
            # Sort observations by timestamp
            if 'observations' in target:
                observations = sorted(target['observations'], key=lambda x: x['timestamp'])
                target_trajectories[target_id] = observations
        
        # Split targets into training and testing sets
        target_ids = list(target_trajectories.keys())
        if not target_ids:
            print("No trajectory data available")
            return None
            
        train_ids, test_ids = train_test_split(target_ids, test_size=test_ratio, random_state=42)
        
        print(f"Split {len(target_ids)} targets into {len(train_ids)} training and {len(test_ids)} testing")
        
        # Prepare blue force data if included
        blue_force_data = None
        if include_blue_forces and self.blue_force_observations:
            # Map blue force observations by timestamp
            blue_force_data = {}
            for obs in self.blue_force_observations:
                timestamp = obs['timestamp']
                if timestamp not in blue_force_data:
                    blue_force_data[timestamp] = []
                blue_force_data[timestamp].append(obs)
        
        # Build datasets for each prediction horizon
        datasets = {}
        
        for horizon in prediction_horizons:
            print(f"Building datasets for {horizon}-step prediction...")
            
            # Create training dataset
            train_inputs = []
            train_outputs = []
            
            for target_id in train_ids:
                trajectory = target_trajectories[target_id]
                target_class = self.targets[target_id]['class']
                
                # For each valid start position
                for i in range(len(trajectory) - window_size - horizon + 1):
                    # Extract input sequence
                    input_seq = trajectory[i:i+window_size]
                    
                    # Extract target (future position)
                    target_obs = trajectory[i+window_size+horizon-1]
                    
                    # Create input features
                    input_features = self._create_features(
                        input_seq, target_id, target_class, 
                        include_terrain=include_terrain, 
                        include_blue_forces=include_blue_forces, 
                        blue_force_data=blue_force_data
                    )
                    
                    # Create output features (future position)
                    output_features = [target_obs['x_coord'], target_obs['y_coord']]
                    
                    train_inputs.append(input_features)
                    train_outputs.append(output_features)
            
            # Create testing dataset
            test_inputs = []
            test_outputs = []
            
            for target_id in test_ids:
                trajectory = target_trajectories[target_id]
                target_class = self.targets[target_id]['class']
                
                # For each valid start position
                for i in range(len(trajectory) - window_size - horizon + 1):
                    # Extract input sequence
                    input_seq = trajectory[i:i+window_size]
                    
                    # Extract target (future position)
                    target_obs = trajectory[i+window_size+horizon-1]
                    
                    # Create input features
                    input_features = self._create_features(
                        input_seq, target_id, target_class, 
                        include_terrain=include_terrain, 
                        include_blue_forces=include_blue_forces, 
                        blue_force_data=blue_force_data
                    )
                    
                    # Create output features (future position)
                    output_features = [target_obs['x_coord'], target_obs['y_coord']]
                    
                    test_inputs.append(input_features)
                    test_outputs.append(output_features)
            
            # Store datasets
            datasets[f'horizon_{horizon}'] = {
                'X_train': train_inputs,
                'y_train': train_outputs,
                'X_test': test_inputs,
                'y_test': test_outputs
            }
            
            print(f"  - Training samples: {len(train_inputs)}")
            print(f"  - Testing samples: {len(test_inputs)}")
        
        return datasets
    
    def _create_features(self, observations, target_id, target_class, include_terrain=True, 
                        include_blue_forces=True, blue_force_data=None):
        """
        Create feature vector for a sequence of observations.
        
        Args:
            observations: List of observation dictionaries
            target_id: ID of the target
            target_class: Class of the target
            include_terrain: Whether to include terrain features
            include_blue_forces: Whether to include blue force positions
            blue_force_data: Dictionary of blue force observations by timestamp
            
        Returns:
            Dictionary of features
        """
        # Extract basic sequence features
        seq_features = []
        for obs in observations:
            # Extract position and convert to grid coordinates
            x_coord = obs['x_coord']
            y_coord = obs['y_coord']
            grid_x, grid_y = self.data_to_grid(x_coord, y_coord)
            
            # Get terrain features if available and requested
            terrain_features = []
            if include_terrain and self.terrain_map is not None:
                # Get terrain type and elevation
                terrain_type = self.terrain_map[grid_x, grid_y]
                elevation = self.elevation_map[grid_x, grid_y] if self.elevation_map is not None else 0.0
                
                # Convert terrain type to one-hot encoding
                terrain_one_hot = [0] * len(self.TERRAIN_TYPES)
                if 0 <= terrain_type < len(self.TERRAIN_TYPES):
                    terrain_one_hot[terrain_type] = 1
                
                # Terrain speed factor
                speed_factor = self.TERRAIN_TYPES.get(terrain_type, {}).get('speed_factor', 1.0)
                
                # Slope if available
                slope = 0.0
                if self._elevation_gradients is not None:
                    slope = self._elevation_gradients[grid_x, grid_y]
                
                terrain_features = [elevation / 1000.0, speed_factor, slope] + terrain_one_hot
            
            # Extract timestamp features
            timestamp = obs['timestamp']
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            hour_of_day = timestamp.hour / 24.0
            day_of_week = timestamp.weekday() / 7.0
            
            # Combine all observation features
            obs_features = [x_coord, y_coord, hour_of_day, day_of_week] + terrain_features
            seq_features.append(obs_features)
        
        # Extract blue force features if available and requested
        blue_force_features = []
        if include_blue_forces and blue_force_data is not None:
            # Get the timestamp of the last observation in the sequence
            last_timestamp = observations[-1]['timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = pd.to_datetime(last_timestamp)
            
            # Find the closest blue force observations
            closest_timestamp = None
            min_time_diff = float('inf')
            
            for bf_timestamp in blue_force_data.keys():
                if isinstance(bf_timestamp, str):
                    bf_timestamp = pd.to_datetime(bf_timestamp)
                
                time_diff = abs((bf_timestamp - last_timestamp).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_timestamp = bf_timestamp
            
            # If we found close blue force observations, extract features
            if closest_timestamp is not None and min_time_diff < 3600:  # Within 1 hour
                bf_observations = blue_force_data[closest_timestamp]
                
                # Sort by distance to target
                target_x = observations[-1]['x_coord']
                target_y = observations[-1]['y_coord']
                
                for bf_obs in bf_observations:
                    bf_x = bf_obs['x_coord']
                    bf_y = bf_obs['y_coord']
                    
                    # Calculate distance
                    distance = math.sqrt((bf_x - target_x)**2 + (bf_y - target_y)**2)
                    bf_obs['distance'] = distance
                
                # Sort by distance and take the closest few
                bf_observations.sort(key=lambda x: x['distance'])
                max_blue_forces = 5
                closest_bfs = bf_observations[:max_blue_forces]
                
                # Extract features for each close blue force
                for bf_obs in closest_bfs:
                    bf_x = bf_obs['x_coord']
                    bf_y = bf_obs['y_coord']
                    bf_class = bf_obs.get('force_class', 'unknown')
                    
                    # Distance and direction to blue force
                    distance = bf_obs['distance']
                    direction = math.atan2(bf_y - target_y, bf_x - target_x)
                    
                    # One-hot encoding of blue force class
                    bf_class_one_hot = [0] * len(self.BLUE_FORCE_CLASSES)
                    if bf_class in self.BLUE_FORCE_CLASSES:
                        bf_class_index = list(self.BLUE_FORCE_CLASSES.keys()).index(bf_class)
                        bf_class_one_hot[bf_class_index] = 1
                    
                    # Combine blue force features
                    bf_features = [bf_x, bf_y, distance, direction] + bf_class_one_hot
                    blue_force_features.extend(bf_features)
                
                # Pad if we have fewer than max_blue_forces
                padding_needed = max_blue_forces - len(closest_bfs)
                if padding_needed > 0:
                    # Padding with zeros
                    padding_size = (4 + len(self.BLUE_FORCE_CLASSES)) * padding_needed
                    blue_force_features.extend([0.0] * padding_size)
        
        # Convert target class to one-hot encoding
        target_class_one_hot = [0] * len(self.TARGET_CLASSES)
        if target_class in self.TARGET_CLASSES:
            target_class_index = list(self.TARGET_CLASSES.keys()).index(target_class)
            target_class_one_hot[target_class_index] = 1
        
        # Combine all features
        features = {
            'target_sequence': seq_features,
            'blue_forces': blue_force_features,
            'target_type': target_class_one_hot,
            'target_id': target_id
        }
        
        return features
    
    def visualize_terrain(self, show_elevation=True, figsize=(12, 10)):
        """
        Visualize the terrain map.
        
        Args:
            show_elevation: Whether to include elevation contours
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        # Create custom colormap for terrain
        terrain_colors = [self.TERRAIN_TYPES[i]['color'] for i in range(len(self.TERRAIN_TYPES))]
        terrain_cmap = mcolors.ListedColormap(terrain_colors)
        
        # Plot terrain
        plt.imshow(self.terrain_map.T, origin='lower', cmap=terrain_cmap, 
                  vmin=0, vmax=len(self.TERRAIN_TYPES)-1)
        
        # Add elevation contours if requested
        if show_elevation and self.elevation_map is not None:
            # Contour levels
            levels = np.linspace(self.elevation_map.min(), self.elevation_map.max(), 10)
            
            # Plot contours
            contour = plt.contour(self.elevation_map.T, levels=levels, colors='black', alpha=0.5, linewidths=0.5)
            plt.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')
            
        # Create legend for terrain types
        legend_patches = []
        for i, info in self.TERRAIN_TYPES.items():
            patch = plt.Rectangle((0, 0), 1, 1, facecolor=info['color'])
            legend_patches.append(patch)
            
        plt.legend(legend_patches, [info['name'] for info in self.TERRAIN_TYPES.values()], 
                  loc='lower right', title='Terrain Types')
        
        plt.title('Battlefield Terrain Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_entities(self, timestamp=None, show_terrain=True, figsize=(12, 10)):
        """
        Visualize targets and blue forces on the terrain.
        
        Args:
            timestamp: Specific timestamp to visualize (None for latest)
            show_terrain: Whether to include terrain in the background
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        # Plot terrain if requested
        if show_terrain:
            # Create custom colormap for terrain
            terrain_colors = [self.TERRAIN_TYPES[i]['color'] for i in range(len(self.TERRAIN_TYPES))]
            terrain_cmap = mcolors.ListedColormap(terrain_colors)
            
            # Plot terrain
            plt.imshow(self.terrain_map.T, origin='lower', cmap=terrain_cmap, 
                      vmin=0, vmax=len(self.TERRAIN_TYPES)-1, alpha=0.7)
        
        # Find the relevant observations for the given timestamp
        if timestamp is not None:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            # Filter target observations by timestamp
            target_obs = [obs for obs in self.target_observations if obs['timestamp'] == timestamp]
            blue_force_obs = [obs for obs in self.blue_force_observations if obs['timestamp'] == timestamp]
        else:
            # Use all observations
            target_obs = self.target_observations
            blue_force_obs = self.blue_force_observations
        
        # Plot targets
        target_colors = {
            'infantry': 'red',
            'light_vehicle': 'orange',
            'heavy_vehicle': 'darkred',
            'uav': 'purple',
            'civilian': 'pink'
        }
        
        target_markers = {
            'infantry': 's',  # Square
            'light_vehicle': '^',  # Triangle
            'heavy_vehicle': 'D',  # Diamond
            'uav': '*',  # Star
            'civilian': 'o'  # Circle
        }
        
        if target_obs:
            for obs in target_obs:
                target_id = obs['id']
                x, y = obs['x_coord'], obs['y_coord']
                grid_x, grid_y = self.data_to_grid(x, y)
                
                # Get target class
                target_class = 'unknown'
                if target_id in self.targets:
                    target_class = self.targets[target_id]['class']
                
                # Set marker and color based on class
                marker = target_markers.get(target_class, 'o')
                color = target_colors.get(target_class, 'red')
                
                # Plot the target
                plt.scatter(grid_x, grid_y, marker=marker, color=color, s=80, edgecolors='black', zorder=10)
        
        # Plot blue forces
        blue_force_markers = {
            'infantry_squad': 's',  # Square
            'mechanized_patrol': '^',  # Triangle
            'recon_team': 'o',  # Circle
            'command_post': 'p',  # Pentagon
            'uav_surveillance': '*'  # Star
        }
        
        if blue_force_obs:
            for obs in blue_force_obs:
                force_id = obs['id']
                x, y = obs['x_coord'], obs['y_coord']
                grid_x, grid_y = self.data_to_grid(x, y)
                
                # Get blue force class
                force_class = 'unknown'
                if force_id in self.blue_forces:
                    force_class = self.blue_forces[force_id]['class']
                
                # Set marker based on class
                marker = blue_force_markers.get(force_class, 'o')
                
                # Plot the blue force
                plt.scatter(grid_x, grid_y, marker=marker, color='blue', s=80, edgecolors='black', zorder=10)
                
                # Draw detection range circle
                detection_range = self.BLUE_FORCE_CLASSES.get(force_class, {}).get('detection_range', 3)
                detection_circle = plt.Circle((grid_x, grid_y), detection_range, color='blue', fill=False, 
                                            linestyle='--', alpha=0.5, zorder=5)
                plt.gca().add_patch(detection_circle)
        
        # Create legends
        target_legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Infantry'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10, label='Light Vehicle'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='darkred', markersize=10, label='Heavy Vehicle'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='purple', markersize=10, label='UAV'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label='Civilian')
        ]
        
        blue_legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Blue Infantry'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Blue Mechanized'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Blue Recon'),
            plt.Line2D([0], [0], marker='p', color='w', markerfacecolor='blue', markersize=10, label='Blue Command'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=10, label='Blue UAV')
        ]
        
        # Add legends
        plt.legend(handles=target_legend_elements, loc='upper right', title='Targets')
        plt.gca().add_artist(plt.legend(handles=blue_legend_elements, loc='upper left', title='Blue Forces'))
        
        # Set title with timestamp if provided
        if timestamp is not None:
            plt.title(f'Battlefield Entities at {timestamp}')
        else:
            plt.title('Battlefield Entities')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_trajectories(self, target_ids=None, show_terrain=True, figsize=(12, 10)):
        """
        Visualize trajectories of targets.
        
        Args:
            target_ids: List of target IDs to visualize (None for all)
            show_terrain: Whether to include terrain in the background
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        # Plot terrain if requested
        if show_terrain:
            # Create custom colormap for terrain
            terrain_colors = [self.TERRAIN_TYPES[i]['color'] for i in range(len(self.TERRAIN_TYPES))]
            terrain_cmap = mcolors.ListedColormap(terrain_colors)
            
            # Plot terrain
            plt.imshow(self.terrain_map.T, origin='lower', cmap=terrain_cmap, 
                      vmin=0, vmax=len(self.TERRAIN_TYPES)-1, alpha=0.7)
        
        # If no target IDs provided, use all targets
        if target_ids is None:
            target_ids = list(self.targets.keys())
        
        # Target colors by class
        target_colors = {
            'infantry': 'red',
            'light_vehicle': 'orange',
            'heavy_vehicle': 'darkred',
            'uav': 'purple',
            'civilian': 'pink'
        }
        
        # Plot each target's trajectory
        for target_id in target_ids:
            if target_id not in self.targets:
                continue
                
            target = self.targets[target_id]
            target_class = target['class']
            color = target_colors.get(target_class, 'red')
            
            # Get observations for this target
            observations = []
            for obs in self.target_observations:
                if obs['id'] == target_id:
                    observations.append(obs)
            
            # Sort by timestamp
            observations.sort(key=lambda x: x['timestamp'])
            
            if not observations:
                continue
                
            # Convert coordinates to grid indices
            grid_points = []
            for obs in observations:
                x, y = obs['x_coord'], obs['y_coord']
                grid_x, grid_y = self.data_to_grid(x, y)
                grid_points.append((grid_x, grid_y))
            
            # Extract x and y coordinates
            grid_xs, grid_ys = zip(*grid_points)
            
            # Plot trajectory
            plt.plot(grid_xs, grid_ys, '-', color=color, linewidth=2, alpha=0.7)
            
            # Plot start and end points
            plt.scatter(grid_xs[0], grid_ys[0], marker='o', color=color, s=100, edgecolors='black', zorder=10)
            plt.scatter(grid_xs[-1], grid_ys[-1], marker='x', color=color, s=100, edgecolors='black', zorder=10)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=2, label='Infantry'),
            plt.Line2D([0], [0], color='orange', lw=2, label='Light Vehicle'),
            plt.Line2D([0], [0], color='darkred', lw=2, label='Heavy Vehicle'),
            plt.Line2D([0], [0], color='purple', lw=2, label='UAV'),
            plt.Line2D([0], [0], color='pink', lw=2, label='Civilian'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Start'),
            plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=8, label='End')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title('Target Trajectories')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_prediction(self, model, target_id, timestamp, prediction_horizon=5, 
                            show_terrain=True, figsize=(12, 10)):
        """
        Visualize a trajectory prediction for a specific target.
        
        Args:
            model: Trained prediction model
            target_id: ID of the target to predict
            timestamp: Timestamp to predict from
            prediction_horizon: Number of time steps to predict ahead
            show_terrain: Whether to include terrain in the background
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure
        """
        # Find the target
        if target_id not in self.targets:
            print(f"Target {target_id} not found")
            return None
        
        # Get observations for this target
        observations = []
        for obs in self.target_observations:
            if obs['id'] == target_id:
                observations.append(obs)
        
        # Sort by timestamp
        observations.sort(key=lambda x: x['timestamp'])
        
        if not observations:
            print(f"No observations for target {target_id}")
            return None
        
        # Find the observation at the given timestamp
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Find the closest observation to the given timestamp
        closest_obs = None
        min_time_diff = float('inf')
        
        for i, obs in enumerate(observations):
            obs_timestamp = obs['timestamp']
            if isinstance(obs_timestamp, str):
                obs_timestamp = pd.to_datetime(obs_timestamp)
            
            time_diff = abs((obs_timestamp - timestamp).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_obs = i
        
        if closest_obs is None or closest_obs < 5:  # Need at least 5 past observations
            print(f"Not enough observations before {timestamp}")
            return None
        
        # Extract past sequence
        window_size = 5
        input_seq = observations[closest_obs-window_size+1:closest_obs+1]
        
        # Ensure we have enough future observations for ground truth
        if closest_obs + prediction_horizon >= len(observations):
            print(f"Not enough future observations for {prediction_horizon}-step prediction")
            return None
        
        # Create feature vector
        target_class = self.targets[target_id]['class']
        features = self._create_features(
            input_seq, target_id, target_class, 
            include_terrain=True, 
            include_blue_forces=True, 
            blue_force_data=None  # Simple visualization, skip blue forces
        )
        
        # Make prediction
        # NOTE: This is a placeholder - the actual prediction method depends on your model
        # You would need to adapt this to your specific model implementation
        try:
            predicted_pos = model.predict([features])[0]
        except Exception as e:
            print(f"Error making prediction: {e}")
            print("Using simple linear extrapolation as a fallback")
            
            # Simple linear extrapolation as fallback
            last_pos = (input_seq[-1]['x_coord'], input_seq[-1]['y_coord'])
            prev_pos = (input_seq[-2]['x_coord'], input_seq[-2]['y_coord'])
            
            # Calculate velocity vector
            vx = last_pos[0] - prev_pos[0]
            vy = last_pos[1] - prev_pos[1]
            
            # Extrapolate
            predicted_pos = (last_pos[0] + vx * prediction_horizon, last_pos[1] + vy * prediction_horizon)
        
        # Get ground truth
        ground_truth = (
            observations[closest_obs + prediction_horizon]['x_coord'],
            observations[closest_obs + prediction_horizon]['y_coord']
        )
        
        # Visualize
        plt.figure(figsize=figsize)
        
        # Plot terrain if requested
        if show_terrain:
            # Create custom colormap for terrain
            terrain_colors = [self.TERRAIN_TYPES[i]['color'] for i in range(len(self.TERRAIN_TYPES))]
            terrain_cmap = mcolors.ListedColormap(terrain_colors)
            
            # Plot terrain
            plt.imshow(self.terrain_map.T, origin='lower', cmap=terrain_cmap, 
                      vmin=0, vmax=len(self.TERRAIN_TYPES)-1, alpha=0.7)
        
        # Convert past trajectory to grid coordinates
        grid_points = []
        for obs in input_seq:
            x, y = obs['x_coord'], obs['y_coord']
            grid_x, grid_y = self.data_to_grid(x, y)
            grid_points.append((grid_x, grid_y))
        
        # Extract x and y coordinates
        grid_xs, grid_ys = zip(*grid_points)
        
        # Convert prediction to grid coordinates
        pred_x, pred_y = self.data_to_grid(predicted_pos[0], predicted_pos[1])
        
        # Convert ground truth to grid coordinates
        truth_x, truth_y = self.data_to_grid(ground_truth[0], ground_truth[1])
        
        # Calculate error distance
        error_distance = math.sqrt(
            (predicted_pos[0] - ground_truth[0])**2 + 
            (predicted_pos[1] - ground_truth[1])**2
        )
        
        # Plot past trajectory
        plt.plot(grid_xs, grid_ys, '-', color='blue', linewidth=2, alpha=0.7)
        plt.scatter(grid_xs[-1], grid_ys[-1], marker='o', color='blue', s=100, edgecolors='black', zorder=10)
        
        # Plot prediction
        plt.scatter(pred_x, pred_y, marker='x', color='red', s=100, edgecolors='black', zorder=10)
        
        # Plot ground truth
        plt.scatter(truth_x, truth_y, marker='*', color='green', s=100, edgecolors='black', zorder=10)
        
        # Draw a line between prediction and ground truth
        plt.plot([pred_x, truth_x], [pred_y, truth_y], '--', color='black', alpha=0.5)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=2, label='Past Trajectory'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Current Position'),
            plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=8, label='Predicted Position'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=8, label='Actual Position')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add error information
        plt.text(0.05, 0.95, f"Error: {error_distance:.2f} meters", transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.title(f'Trajectory Prediction for Target {target_id} ({prediction_horizon}-step ahead)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_to_disk(self, output_dir='simulation_output'):
        """
        Save the simulation data to disk.
        
        Args:
            output_dir: Directory to save the data
            
        Returns:
            Dictionary of saved file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save terrain data if available
        terrain_path = None
        elevation_path = None
        
        if self.terrain_map is not None:
            terrain_path = os.path.join(output_dir, 'terrain_map.npy')
            np.save(terrain_path, self.terrain_map)
        
        if self.elevation_map is not None:
            elevation_path = os.path.join(output_dir, 'elevation_map.npy')
            np.save(elevation_path, self.elevation_map)
        
        # Save observations to CSV
        target_csv_path = None
        blue_force_csv_path = None
        
        if self.target_observations:
            target_df = pd.DataFrame(self.target_observations)
            target_csv_path = os.path.join(output_dir, 'target_observations.csv')
            target_df.to_csv(target_csv_path, index=False)
        
        if self.blue_force_observations:
            blue_force_df = pd.DataFrame(self.blue_force_observations)
            blue_force_csv_path = os.path.join(output_dir, 'blue_force_observations.csv')
            blue_force_df.to_csv(blue_force_csv_path, index=False)
        
        print(f"Simulation data saved to {output_dir}")
        
        return {
            'terrain_path': terrain_path,
            'elevation_path': elevation_path,
            'target_csv_path': target_csv_path,
            'blue_force_csv_path': blue_force_csv_path
        }
    
    @classmethod
    def load_from_disk(cls, input_dir):
        """
        Load a simulation from saved data.
        
        Args:
            input_dir: Directory with saved data
            
        Returns:
            BattlefieldSimulation instance
        """
        # Create a new simulation instance
        sim = cls()
        
        # Load terrain data if available
        terrain_path = os.path.join(input_dir, 'terrain_map.npy')
        elevation_path = os.path.join(input_dir, 'elevation_map.npy')
        
        if os.path.exists(terrain_path):
            sim.terrain_map = np.load(terrain_path)
            print(f"Loaded terrain map with shape {sim.terrain_map.shape}")
            
            # Update size if needed
            if sim.terrain_map.shape != sim.size:
                sim.size = sim.terrain_map.shape
        
        if os.path.exists(elevation_path):
            sim.elevation_map = np.load(elevation_path)
            print(f"Loaded elevation map with shape {sim.elevation_map.shape}")
        
        # Load observations from CSV
        target_csv_path = os.path.join(input_dir, 'target_observations.csv')
        blue_force_csv_path = os.path.join(input_dir, 'blue_force_observations.csv')
        
        if os.path.exists(target_csv_path):
            sim.load_observation_data(target_csv_path, blue_force_csv_path if os.path.exists(blue_force_csv_path) else None)
        
        return sim

# Example usage
if __name__ == "__main__":
    # NOTE: This process can be time-consuming and might take a while to run
    # Create a simulation
    sim = BattlefieldSimulation(size=(100, 100))
    
    # Generate some synthetic terrain if needed
    if not os.path.exists("simulation_data/elevation_map.npy"):
        print("Generating synthetic terrain...")
        # Generate terrain types
        terrain = np.ones(sim.size, dtype=int) * 4  # Default to grassland
        
        # Add some water bodies
        for _ in range(3):
            center_x = np.random.randint(10, 90)
            center_y = np.random.randint(10, 90)
            radius = np.random.randint(5, 15)
            
            for x in range(max(0, center_x - radius), min(100, center_x + radius + 1)):
                for y in range(max(0, center_y - radius), min(100, center_y + radius + 1)):
                    if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                        terrain[x, y] = 0  # Water
        
        # Add some forests
        for _ in range(5):
            center_x = np.random.randint(10, 90)
            center_y = np.random.randint(10, 90)
            width = np.random.randint(10, 30)
            height = np.random.randint(10, 30)
            
            for x in range(max(0, center_x - width//2), min(100, center_x + width//2 + 1)):
                for y in range(max(0, center_y - height//2), min(100, center_y + height//2 + 1)):
                    if np.random.random() < 0.7:  # Some randomness
                        terrain[x, y] = 3  # Forest
        
        # Add some urban areas
        for _ in range(2):
            center_x = np.random.randint(10, 90)
            center_y = np.random.randint(10, 90)
            size = np.random.randint(5, 15)
            
            for x in range(max(0, center_x - size), min(100, center_x + size + 1)):
                for y in range(max(0, center_y - size), min(100, center_y + size + 1)):
                    if np.random.random() < 0.8:  # Some randomness
                        terrain[x, y] = 1  # Urban
        
        # Add some roads
        for _ in range(3):
            # Horizontal road
            y = np.random.randint(10, 90)
            width = np.random.randint(1, 3)
            
            for x in range(100):
                for offset in range(-width//2, width//2 + 1):
                    road_y = y + offset
                    if 0 <= road_y < 100:
                        terrain[x, road_y] = 6  # Road
        
        for _ in range(3):
            # Vertical road
            x = np.random.randint(10, 90)
            width = np.random.randint(1, 3)
            
            for y in range(100):
                for offset in range(-width//2, width//2 + 1):
                    road_x = x + offset
                    if 0 <= road_x < 100:
                        terrain[road_x, y] = 6  # Road
        
        # Generate elevation
        elevation = np.zeros(sim.size, dtype=float)
        
        # Add some hills
        for _ in range(10):
            center_x = np.random.randint(10, 90)
            center_y = np.random.randint(10, 90)
            radius = np.random.randint(10, 30)
            height = np.random.uniform(100, 500)
            
            for x in range(max(0, center_x - radius), min(100, center_x + radius + 1)):
                for y in range(max(0, center_y - radius), min(100, center_y + radius + 1)):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance <= radius:
                        # Bell curve based on distance
                        elevation[x, y] += height * np.exp(-(distance**2) / (2 * (radius/2)**2))
        
        # Add some noise
        elevation += np.random.normal(0, 10, sim.size)
        
        # Smooth the elevation
        elevation = gaussian_filter(elevation, sigma=1.0)
        
        # Ensure water is at low elevation
        for x in range(100):
            for y in range(100):
                if terrain[x, y] == 0:  # Water
                    elevation[x, y] = max(0, elevation[x, y] - 50)
        
        # Save to disk
        os.makedirs("simulation_data", exist_ok=True)
        np.save("simulation_data/terrain_map.npy", terrain)
        np.save("simulation_data/elevation_map.npy", elevation)
        
        # Set in simulation
        sim.terrain_map = terrain
        sim.elevation_map = elevation
    else:
        # Load terrain data
        sim.load_terrain_data(
            terrain_data_path="simulation_data/terrain_map.npy",
            elevation_data_path="simulation_data/elevation_map.npy"
        )
    
    # Load observation data if available
    if os.path.exists("synthetic_data/target_observations.csv"):
        sim.load_observation_data(
            target_csv="synthetic_data/target_observations.csv",
            blue_force_csv="synthetic_data/blue_force_observations.csv"
        )
    
    # Visualize terrain
    sim.visualize_terrain(show_elevation=True)
    plt.savefig("terrain_visualization.png")
    plt.close()
    
    # Visualize entities
    if sim.target_observations:
        sim.visualize_entities(show_terrain=True)
        plt.savefig("entities_visualization.png")
        plt.close()
        
        # Visualize trajectories
        sim.visualize_trajectories(show_terrain=True)
        plt.savefig("trajectories_visualization.png")
        plt.close()
        
        # Build datasets for prediction
        datasets = sim.build_trajectory_datasets(
            test_ratio=0.2,
            window_size=5,
            prediction_horizons=[1, 3, 5, 10],
            include_terrain=True,
            include_blue_forces=True
        )
        
        # Print dataset statistics
        if datasets:
            print("\nDataset Statistics:")
            for horizon, data in datasets.items():
                print(f"  {horizon}:")
                print(f"    Training: {len(data['X_train'])} samples")
                print(f"    Testing: {len(data['X_test'])} samples")
