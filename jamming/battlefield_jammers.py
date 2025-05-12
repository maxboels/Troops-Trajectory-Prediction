import numpy as np
import pandas as pd
import random
import datetime
import os
import uuid
import math
from pathlib import Path

class BattlefieldJammerGenerator:
    """
    Generates RF jammers for a battlefield simulation environment.
    
    This class creates both static and mobile jammers with configurable
    characteristics and outputs their positions and effects over time.
    """
    
    # Define jammer types with their characteristics
    JAMMER_TYPES = {
        'static_barrage': {
            'power': 100,           # High power output (arbitrary units)
            'range': 2000,          # Large coverage area (meters)
            'angle': 360,           # Omnidirectional (degrees)
            'frequency_low': 30,    # 30 MHz
            'frequency_high': 300,  # 300 MHz (VHF band)
            'mobility': 'static',
        },
        'static_directional': {
            'power': 150,           # Very high power output
            'range': 3000,          # Long range (meters)
            'angle': 60,            # Directional beam (degrees)
            'frequency_low': 300,   # 300 MHz
            'frequency_high': 3000, # 3 GHz (UHF to microwave)
            'mobility': 'static',
        },
        'vehicle_tactical': {
            'power': 50,            # Medium power output
            'range': 1000,          # Medium range (meters)
            'angle': 180,           # Semi-directional (degrees)
            'frequency_low': 20,    # 20 MHz
            'frequency_high': 500,  # 500 MHz
            'mobility': 'mobile',
        },
        'portable_reactive': {
            'power': 25,            # Lower power output
            'range': 500,           # Shorter range (meters)
            'angle': 360,           # Omnidirectional (degrees)
            'frequency_low': 400,   # 400 MHz
            'frequency_high': 6000, # 6 GHz (covers WiFi, some radar)
            'mobility': 'mobile',
        },
        'drone_jammer': {
            'power': 30,            # Medium-low power
            'range': 800,           # Medium-short range (meters)
            'angle': 90,            # Directional cone (degrees)
            'frequency_low': 1000,  # 1 GHz
            'frequency_high': 6000, # 6 GHz (targets drone control frequencies)
            'mobility': 'mobile',
        }
    }
    
    def __init__(self, 
                 area_bounds=(0, 0, 10000, 10000),  # (min_x, min_y, max_x, max_y) in meters
                 start_time=None,
                 duration_hours=48,
                 seed=None,
                 existing_data_dir=None):
        """
        Initialize the battlefield jammer generator.
        
        Args:
            area_bounds: Tuple of (min_x, min_y, max_x, max_y) defining the area in meters
            start_time: Starting datetime for the simulation (default: now)
            duration_hours: Duration of the simulation in hours
            seed: Random seed for reproducibility
            existing_data_dir: Directory containing existing simulation data
        """
        self.min_x, self.min_y, self.max_x, self.max_y = area_bounds
        self.start_time = start_time if start_time else datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(hours=duration_hours)

        self.duration_hours = duration_hours
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize storage for generated jammers and observations
        self.jammers = {}
        self.jammer_observations = []
        self.jamming_effects = []
        
        # Load existing simulation data if provided
        self.existing_data = None
        if existing_data_dir:
            self.load_existing_data(existing_data_dir)
            
    def load_existing_data(self, data_dir):
        """Load existing target and blue force data"""
        self.existing_data = {}
        
        target_csv = os.path.join(data_dir, 'target_observations.csv')
        blue_force_csv = os.path.join(data_dir, 'blue_force_observations.csv')
        
        if os.path.exists(target_csv):
            self.existing_data['targets'] = pd.read_csv(target_csv)
            print(f"Loaded {len(self.existing_data['targets'])} target observations")
            
            # Get the start time from the data if not already set
            if 'timestamp' in self.existing_data['targets'].columns:
                self.existing_data['targets']['timestamp'] = pd.to_datetime(self.existing_data['targets']['timestamp'])
                self.start_time = self.existing_data['targets']['timestamp'].min()
                self.end_time = self.existing_data['targets']['timestamp'].max()
        
        if os.path.exists(blue_force_csv):
            self.existing_data['blue_forces'] = pd.read_csv(blue_force_csv)
            print(f"Loaded {len(self.existing_data['blue_forces'])} blue force observations")
            
            # Update time bounds if needed
            if 'timestamp' in self.existing_data['blue_forces'].columns:
                self.existing_data['blue_forces']['timestamp'] = pd.to_datetime(self.existing_data['blue_forces']['timestamp'])
                if self.existing_data['blue_forces']['timestamp'].min() < self.start_time:
                    self.start_time = self.existing_data['blue_forces']['timestamp'].min()
                if self.existing_data['blue_forces']['timestamp'].max() > self.end_time:
                    self.end_time = self.existing_data['blue_forces']['timestamp'].max()
    
    def _generate_random_location(self):
        """Generate a random location within the defined bounds"""
        x = random.uniform(self.min_x, self.max_x)
        y = random.uniform(self.min_y, self.max_y)
        return x, y
    
    def _calculate_strategic_location(self, strategic_type='center'):
        """
        Calculate a strategic location for a jammer based on type.
        
        Args:
            strategic_type: 'center', 'border', 'high_value', 'random'
        
        Returns:
            Tuple of (x, y) coordinates
        """
        if strategic_type == 'center':
            # Position the jammer near the center of the map
            center_x = (self.min_x + self.max_x) / 2
            center_y = (self.min_y + self.max_y) / 2
            offset_x = random.uniform(-0.2, 0.2) * (self.max_x - self.min_x)
            offset_y = random.uniform(-0.2, 0.2) * (self.max_y - self.min_y)
            return center_x + offset_x, center_y + offset_y
            
        elif strategic_type == 'border':
            # Position the jammer near a border
            border_side = random.choice(['top', 'right', 'bottom', 'left'])
            
            if border_side == 'top':
                return random.uniform(self.min_x, self.max_x), self.max_y - random.uniform(0, 500)
            elif border_side == 'right':
                return self.max_x - random.uniform(0, 500), random.uniform(self.min_y, self.max_y)
            elif border_side == 'bottom':
                return random.uniform(self.min_x, self.max_x), self.min_y + random.uniform(0, 500)
            else:  # left
                return self.min_x + random.uniform(0, 500), random.uniform(self.min_y, self.max_y)
        
        elif strategic_type == 'high_value' and self.existing_data is not None:
            # Position the jammer near a high concentration of targets or blue forces
            # This is a simplified approach - in a real system, you'd use more sophisticated analysis
            
            if 'targets' in self.existing_data:
                # Get a random target location
                targets = self.existing_data['targets']
                if len(targets) > 0:
                    random_idx = random.randint(0, len(targets) - 1)
                    base_x = targets.iloc[random_idx]['x_coord']
                    base_y = targets.iloc[random_idx]['y_coord']
                    offset_x = random.uniform(-300, 300)
                    offset_y = random.uniform(-300, 300)
                    return base_x + offset_x, base_y + offset_y
            
            # Fallback to random location
            return self._generate_random_location()
                    
        else:  # random or fallback
            return self._generate_random_location()
    
    def generate_static_jammers(self, num_jammers=5):
        """
        Generate static jammers placed at strategic locations.
        
        Args:
            num_jammers: Number of static jammers to generate
        """
        # Choose jammer types that are static
        static_jammer_types = [jt for jt, props in self.JAMMER_TYPES.items() 
                              if props['mobility'] == 'static']
        
        if not static_jammer_types:
            print("No static jammer types defined")
            return
        
        # Create random jammers
        for i in range(num_jammers):
            jammer_id = f"jammer-static-{i}"
            jammer_type = random.choice(static_jammer_types)
            
            # Choose a strategic location
            strategic_type = random.choice(['center', 'border', 'high_value', 'random'])
            position = self._calculate_strategic_location(strategic_type)
            
            # Get base properties from the jammer type
            base_props = self.JAMMER_TYPES[jammer_type].copy()
            
            # Add some randomization to properties
            power = base_props['power'] * random.uniform(0.8, 1.2)
            jammer_range = base_props['range'] * random.uniform(0.9, 1.1)
            angle = base_props['angle']
            direction = random.uniform(0, 360)  # Random direction for directional jammers
            
            # Randomize frequencies slightly
            freq_low = base_props['frequency_low'] * random.uniform(0.95, 1.05)
            freq_high = base_props['frequency_high'] * random.uniform(0.95, 1.05)
            
            # Create jammer entity
            self.jammers[jammer_id] = {
                'id': jammer_id,
                'type': jammer_type,
                'position': position,
                'power': power,
                'range': jammer_range,
                'angle': angle,
                'direction': direction,
                'freq_low': freq_low,
                'freq_high': freq_high,
                'mobility': 'static',
                'active': True,
                'start_time': self.start_time,
                'end_time': self.end_time,  # Static jammers are always on
            }
    
    def generate_mobile_jammers(self, num_jammers=5, attach_to_entities=True):
        """
        Generate mobile jammers that move with targets or blue forces.
        
        Args:
            num_jammers: Number of mobile jammers to generate
            attach_to_entities: Whether to attach jammers to existing entities
        """
        # Choose jammer types that are mobile
        mobile_jammer_types = [jt for jt, props in self.JAMMER_TYPES.items() 
                              if props['mobility'] == 'mobile']
        
        if not mobile_jammer_types:
            print("No mobile jammer types defined")
            return
        
        # Get entity IDs to attach jammers to if requested
        attachable_entities = []
        if attach_to_entities and self.existing_data is not None:
            # Prefer to attach to blue forces
            if 'blue_forces' in self.existing_data:
                blue_force_ids = self.existing_data['blue_forces']['id'].unique()
                attachable_entities.extend([('blue_force', id) for id in blue_force_ids])
            
            # Can also attach to targets (enemy jammers)
            if 'targets' in self.existing_data:
                # Filter to only attach to vehicles, not infantry/civilian
                vehicle_targets = self.existing_data['targets'][
                    self.existing_data['targets']['target_class'].isin(['light_vehicle', 'heavy_vehicle'])
                ]
                vehicle_ids = vehicle_targets['id'].unique()
                attachable_entities.extend([('target', id) for id in vehicle_ids])
        
        # Create mobile jammers
        for i in range(num_jammers):
            jammer_id = f"jammer-mobile-{i}"
            jammer_type = random.choice(mobile_jammer_types)
            
            # Determine if this jammer is attached to an entity
            attached_entity = None
            if attach_to_entities and attachable_entities:
                attached_entity = random.choice(attachable_entities)
                # Remove this entity to avoid attaching multiple jammers to the same entity
                attachable_entities.remove(attached_entity)
            
            # Get base properties from the jammer type
            base_props = self.JAMMER_TYPES[jammer_type].copy()
            
            # Add some randomization to properties
            power = base_props['power'] * random.uniform(0.8, 1.2)
            jammer_range = base_props['range'] * random.uniform(0.9, 1.1)
            angle = base_props['angle']
            direction = random.uniform(0, 360)  # Random initial direction
            
            # Randomize frequencies slightly
            freq_low = base_props['frequency_low'] * random.uniform(0.95, 1.05)
            freq_high = base_props['frequency_high'] * random.uniform(0.95, 1.05)
            
            # Initial position (will be overridden if attached to an entity)
            position = self._generate_random_location()
            
            # Create jammer entity
            self.jammers[jammer_id] = {
                'id': jammer_id,
                'type': jammer_type,
                'position': position,
                'power': power,
                'range': jammer_range,
                'angle': angle,
                'direction': direction,
                'freq_low': freq_low,
                'freq_high': freq_high,
                'mobility': 'mobile',
                'attached_entity': attached_entity,
                'active': True,
                'start_time': self.start_time,
                # Mobile jammers might have operational periods
                'end_time': self.start_time + datetime.timedelta(
                    hours=random.uniform(self.duration_hours/4, self.duration_hours)
                ),
            }
    
    def generate_jammer_observations(self, observation_interval_minutes=5):
        """
        Generate jammer observations over time.
        
        Args:
            observation_interval_minutes: Time between observations
        """
        # Set up time parameters
        current_time = self.start_time
        time_delta = datetime.timedelta(minutes=observation_interval_minutes)
        
        print(f"Generating jammer observations from {self.start_time} to {self.end_time}")
        
        # Keep track of entity positions for mobile jammers
        entity_positions = {}
        
        # Generate observations for all time steps
        while current_time <= self.end_time:
            # Update entity positions from existing data at this timestamp
            if self.existing_data is not None:
                # Update from targets
                if 'targets' in self.existing_data:
                    targets_at_time = self.existing_data['targets'][
                        self.existing_data['targets']['timestamp'] == current_time
                    ]
                    for _, row in targets_at_time.iterrows():
                        entity_id = row['id']
                        entity_positions[('target', entity_id)] = (row['x_coord'], row['y_coord'])
                
                # Update from blue forces
                if 'blue_forces' in self.existing_data:
                    blue_forces_at_time = self.existing_data['blue_forces'][
                        self.existing_data['blue_forces']['timestamp'] == current_time
                    ]
                    for _, row in blue_forces_at_time.iterrows():
                        entity_id = row['id']
                        entity_positions[('blue_force', entity_id)] = (row['x_coord'], row['y_coord'])
            
            # Update each jammer and create observations
            for jammer_id, jammer in self.jammers.items():
                # Skip if jammer is not active at this time
                if current_time < jammer['start_time'] or current_time > jammer['end_time']:
                    continue
                
                # Update jammer position for mobile jammers
                if jammer['mobility'] == 'mobile':
                    if jammer['attached_entity'] is not None and jammer['attached_entity'] in entity_positions:
                        # Update position based on attached entity
                        jammer['position'] = entity_positions[jammer['attached_entity']]
                    else:
                        # Random movement if not attached or entity not found
                        x, y = jammer['position']
                        move_dist = random.uniform(50, 200)  # meters per time step
                        move_angle = random.uniform(0, 2 * np.pi)
                        new_x = x + move_dist * np.cos(move_angle)
                        new_y = y + move_dist * np.sin(move_angle)
                        
                        # Keep within bounds
                        new_x = max(self.min_x, min(self.max_x, new_x))
                        new_y = max(self.min_y, min(self.max_y, new_y))
                        jammer['position'] = (new_x, new_y)
                
                # Record observation
                self.jammer_observations.append({
                    'id': jammer_id,
                    'timestamp': current_time,
                    'x_coord': jammer['position'][0],
                    'y_coord': jammer['position'][1],
                    'jammer_type': jammer['type'],
                    'power': jammer['power'],
                    'range': jammer['range'],
                    'angle': jammer['angle'],
                    'direction': jammer['direction'],
                    'freq_low': jammer['freq_low'],
                    'freq_high': jammer['freq_high'],
                    'mobility': jammer['mobility'],
                    'attached_entity_type': jammer['attached_entity'][0] if jammer.get('attached_entity') else None,
                    'attached_entity_id': jammer['attached_entity'][1] if jammer.get('attached_entity') else None,
                })
                
                # Randomly update direction for mobile directional jammers
                if jammer['mobility'] == 'mobile' and jammer['angle'] < 360:
                    jammer['direction'] = (jammer['direction'] + random.uniform(-30, 30)) % 360
            
            # Move to next time step
            current_time += time_delta
    
    def calculate_jamming_effects(self):
        """
        Calculate the effects of jammers on entities in the simulation.
        
        For each target and blue force observation, calculate:
        - Which jammers are affecting it
        - Signal degradation based on distance, power, etc.
        """
        if not self.existing_data or not self.jammer_observations:
            print("No existing data or jammer observations to calculate effects")
            return
        
        # Process targets
        if 'targets' in self.existing_data:
            for _, target_row in self.existing_data['targets'].iterrows():
                target_time = target_row['timestamp']
                target_x = target_row['x_coord']
                target_y = target_row['y_coord']
                
                # Find jammers active at this time
                active_jammers = [
                    jam for jam in self.jammer_observations 
                    if jam['timestamp'] == target_time
                ]
                
                # Calculate jamming effects for each active jammer
                for jammer in active_jammers:
                    jammer_x = jammer['x_coord']
                    jammer_y = jammer['y_coord']
                    
                    # Calculate distance between target and jammer
                    distance = math.sqrt((target_x - jammer_x)**2 + (target_y - jammer_y)**2)
                    
                    # Skip if target is outside jammer range
                    if distance > jammer['range']:
                        continue
                    
                    # For directional jammers, check if target is within the jamming cone
                    if jammer['angle'] < 360:
                        # Calculate angle from jammer to target
                        target_angle = math.degrees(math.atan2(target_y - jammer_y, target_x - jammer_x))
                        target_angle = (target_angle + 360) % 360
                        
                        # Calculate angular difference
                        angle_diff = abs(target_angle - jammer['direction'])
                        angle_diff = min(angle_diff, 360 - angle_diff)
                        
                        # Skip if target is outside jamming cone
                        if angle_diff > jammer['angle'] / 2:
                            continue
                    
                    # Calculate signal degradation
                    # - Inverse square law for signal strength
                    # - Normalized to jammer range
                    signal_strength = 1.0 - min(1.0, (distance / jammer['range'])**2)
                    
                    # Scale by jammer power
                    jamming_effect = signal_strength * (jammer['power'] / 100.0)
                    
                    # Record the jamming effect
                    self.jamming_effects.append({
                        'timestamp': target_time,
                        'affected_id': target_row['id'],
                        'affected_type': 'target',
                        'jammer_id': jammer['id'],
                        'distance': distance,
                        'jamming_effect': jamming_effect,
                        'original_detection_confidence': target_row.get('detection_confidence', 1.0),
                        'degraded_detection_confidence': max(
                            0.3, target_row.get('detection_confidence', 1.0) * (1.0 - jamming_effect)
                        ),
                    })
        
        # Process blue forces
        if 'blue_forces' in self.existing_data:
            for _, blue_row in self.existing_data['blue_forces'].iterrows():
                blue_time = blue_row['timestamp']
                blue_x = blue_row['x_coord']
                blue_y = blue_row['y_coord']
                
                # Find jammers active at this time
                active_jammers = [
                    jam for jam in self.jammer_observations 
                    if jam['timestamp'] == blue_time
                ]
                
                # Calculate jamming effects for each active jammer
                for jammer in active_jammers:
                    jammer_x = jammer['x_coord']
                    jammer_y = jammer['y_coord']
                    
                    # Calculate distance
                    distance = math.sqrt((blue_x - jammer_x)**2 + (blue_y - jammer_y)**2)
                    
                    # Skip if blue force is outside jammer range
                    if distance > jammer['range']:
                        continue
                    
                    # For directional jammers, check if blue force is within the jamming cone
                    if jammer['angle'] < 360:
                        # Calculate angle from jammer to blue force
                        blue_angle = math.degrees(math.atan2(blue_y - jammer_y, blue_x - jammer_x))
                        blue_angle = (blue_angle + 360) % 360
                        
                        # Calculate angular difference
                        angle_diff = abs(blue_angle - jammer['direction'])
                        angle_diff = min(angle_diff, 360 - angle_diff)
                        
                        # Skip if blue force is outside jamming cone
                        if angle_diff > jammer['angle'] / 2:
                            continue
                    
                    # Calculate signal degradation
                    # - Inverse square law for signal strength
                    # - Normalized to jammer range
                    signal_strength = 1.0 - min(1.0, (distance / jammer['range'])**2)
                    
                    # Scale by jammer power
                    jamming_effect = signal_strength * (jammer['power'] / 100.0)
                    
                    # Record the jamming effect
                    self.jamming_effects.append({
                        'timestamp': blue_time,
                        'affected_id': blue_row['id'],
                        'affected_type': 'blue_force',
                        'jammer_id': jammer['id'],
                        'distance': distance,
                        'jamming_effect': jamming_effect,
                        'original_comms_quality': 1.0,  # Assuming perfect comms by default
                        'degraded_comms_quality': max(0.2, 1.0 - jamming_effect),
                    })
    
    def save_to_csv(self, output_dir='synthetic_data'):
        """
        Save the generated jammer data to CSV files.
        
        Args:
            output_dir: Directory to save the files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert observations to DataFrames
        jammer_df = pd.DataFrame(self.jammer_observations)
        
        # Save to CSV
        jammer_csv_path = os.path.join(output_dir, 'jammer_observations.csv')
        jammer_df.to_csv(jammer_csv_path, index=False)
        
        # Save jamming effects if calculated
        if self.jamming_effects:
            effects_df = pd.DataFrame(self.jamming_effects)
            effects_csv_path = os.path.join(output_dir, 'jamming_effects.csv')
            effects_df.to_csv(effects_csv_path, index=False)
            
            print(f"Generated {len(jammer_df)} jammer observations and {len(effects_df)} jamming effects")
            return {
                'jammer_csv': jammer_csv_path,
                'effects_csv': effects_csv_path
            }
        else:
            print(f"Generated {len(jammer_df)} jammer observations")
            return {
                'jammer_csv': jammer_csv_path
            }
    
    def generate_all_data(self, num_static_jammers=5, num_mobile_jammers=5):
        """
        Generate all jammer data with default parameters.
        
        Args:
            num_static_jammers: Number of static jammers to generate
            num_mobile_jammers: Number of mobile jammers to generate
            
        Returns:
            Dictionary of file paths to the generated CSV files
        """
        print(f"Generating {num_static_jammers} static jammers...")
        self.generate_static_jammers(num_jammers=num_static_jammers)
        
        print(f"Generating {num_mobile_jammers} mobile jammers...")
        self.generate_mobile_jammers(num_jammers=num_mobile_jammers)
        
        print("Generating jammer observations...")
        self.generate_jammer_observations()
        
        print("Calculating jamming effects...")
        self.calculate_jamming_effects()
        
        return self.save_to_csv()


# Example usage
if __name__ == "__main__":
    # Initialize the jammer generator
    generator = BattlefieldJammerGenerator(
        area_bounds=(0, 0, 10000, 10000),
        start_time=datetime.datetime(2023, 5, 1, 0, 0, 0),
        duration_hours=48,
        seed=42,
        existing_data_dir="synthetic_data"
    )
    
    # Generate all jammer data
    csv_files = generator.generate_all_data(
        num_static_jammers=5,
        num_mobile_jammers=7
    )
    
    print("Generated files:")
    for key, path in csv_files.items():
        print(f"  - {key}: {path}")