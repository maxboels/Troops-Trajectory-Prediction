import pandas as pd
import numpy as np
import datetime
import random
import os
import uuid
from pathlib import Path

class SyntheticDataGenerator:
    """
    Generator for synthetic military target tracking data.
    
    Generates realistic target and blue force movement patterns 
    with timestamps, classifications, and coordinated movements.
    """
    
    # Define target classes with their characteristics
    TARGET_CLASSES = {
        'infantry': {'speed': 5, 'pause_probability': 0.2, 'direction_change_probability': 0.1},
        'light_vehicle': {'speed': 15, 'pause_probability': 0.1, 'direction_change_probability': 0.15},
        'heavy_vehicle': {'speed': 10, 'pause_probability': 0.15, 'direction_change_probability': 0.1},
        'uav': {'speed': 40, 'pause_probability': 0.05, 'direction_change_probability': 0.2},
        'civilian': {'speed': 3, 'pause_probability': 0.3, 'direction_change_probability': 0.25}
    }
    
    # Define blue force classes
    BLUE_FORCE_CLASSES = {
        'infantry_squad': {'speed': 4, 'pause_probability': 0.15},
        'mechanized_patrol': {'speed': 20, 'pause_probability': 0.1},
        'recon_team': {'speed': 8, 'pause_probability': 0.2},
        'command_post': {'speed': 1, 'pause_probability': 0.9},
        'uav_surveillance': {'speed': 30, 'pause_probability': 0.05}
    }
    
    def __init__(self, 
                 area_bounds=(0, 0, 10000, 10000),  # (min_x, min_y, max_x, max_y) in meters
                 start_time=None,
                 duration_hours=48,
                 seed=None):
        """
        Initialize the synthetic data generator.
        
        Args:
            area_bounds: Tuple of (min_x, min_y, max_x, max_y) defining the area in meters
            start_time: Starting datetime for the simulation (default: now)
            duration_hours: Duration of the simulation in hours
            seed: Random seed for reproducibility
        """
        self.min_x, self.min_y, self.max_x, self.max_y = area_bounds
        self.start_time = start_time if start_time else datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(hours=duration_hours)
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize storage for generated data
        self.target_observations = []
        self.blue_force_observations = []
        
        # Initialize dictionaries to track entity states
        self.targets = {}
        self.blue_forces = {}
    
    def _generate_random_location(self):
        """Generate a random location within the defined bounds"""
        x = random.uniform(self.min_x, self.max_x)
        y = random.uniform(self.min_y, self.max_y)
        return x, y
    
    def _calculate_new_position(self, current_pos, heading, speed, time_delta_hours):
        """Calculate new position based on heading, speed, and time delta"""
        # Convert heading to radians
        heading_rad = np.radians(heading)
        
        # Calculate distance traveled
        distance = speed * time_delta_hours * 1000  # Convert to meters
        
        # Calculate new position
        dx = distance * np.cos(heading_rad)
        dy = distance * np.sin(heading_rad)
        
        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy
        
        # Ensure position stays within bounds (reflecting off boundaries)
        if new_x < self.min_x:
            new_x = 2 * self.min_x - new_x
            heading = (180 - heading) % 360
        elif new_x > self.max_x:
            new_x = 2 * self.max_x - new_x
            heading = (180 - heading) % 360
            
        if new_y < self.min_y:
            new_y = 2 * self.min_y - new_y
            heading = (360 - heading) % 360
        elif new_y > self.max_y:
            new_y = 2 * self.max_y - new_y
            heading = (360 - heading) % 360
        
        return (new_x, new_y), heading
    
    def generate_targets(self, num_targets=20, observation_interval_minutes=15):
        """
        Generate target entities and their movement observations.
        
        Args:
            num_targets: Number of target entities to generate
            observation_interval_minutes: Time between observations in minutes
        """
        # Create random targets
        for _ in range(num_targets):
            target_id = str(uuid.uuid4())
            target_class = random.choice(list(self.TARGET_CLASSES.keys()))
            initial_position = self._generate_random_location()
            initial_heading = random.uniform(0, 360)
            
            self.targets[target_id] = {
                'id': target_id,
                'class': target_class,
                'position': initial_position,
                'heading': initial_heading,
                'active': True,
                'characteristics': self.TARGET_CLASSES[target_class]
            }
        
        # Generate observations over time
        current_time = self.start_time
        time_delta = datetime.timedelta(minutes=observation_interval_minutes)
        time_delta_hours = observation_interval_minutes / 60.0
        
        # Group targets with similar starting points for convoy-like behavior
        convoy_groups = []
        remaining_targets = list(self.targets.keys())
        
        # Form 2-4 convoys with 2-5 targets each
        num_convoys = random.randint(2, 4)
        for _ in range(num_convoys):
            if len(remaining_targets) <= 1:
                break
                
            convoy_size = min(random.randint(2, 5), len(remaining_targets))
            convoy_lead = random.choice(remaining_targets)
            convoy_members = [convoy_lead]
            remaining_targets.remove(convoy_lead)
            
            # Add members to convoy
            for _ in range(convoy_size - 1):
                if not remaining_targets:
                    break
                member = random.choice(remaining_targets)
                convoy_members.append(member)
                remaining_targets.remove(member)
            
            # Set similar positions for convoy members
            lead_pos = self.targets[convoy_lead]['position']
            for member in convoy_members:
                if member != convoy_lead:
                    # Offset slightly from lead
                    offset_x = random.uniform(-100, 100)
                    offset_y = random.uniform(-100, 100)
                    self.targets[member]['position'] = (lead_pos[0] + offset_x, lead_pos[1] + offset_y)
                    # Same heading as lead
                    self.targets[member]['heading'] = self.targets[convoy_lead]['heading']
            
            convoy_groups.append(convoy_members)
        
        # Generate movement over time
        while current_time <= self.end_time:
            # Update each target
            for target_id, target in self.targets.items():
                if not target['active']:
                    continue
                    
                # Determine if target is in a convoy
                in_convoy = False
                convoy_lead = None
                
                for convoy in convoy_groups:
                    if target_id in convoy:
                        in_convoy = True
                        convoy_lead = convoy[0]
                        break
                
                # For convoy followers, adjust position to follow lead
                if in_convoy and target_id != convoy_lead:
                    lead_target = self.targets[convoy_lead]
                    # Get lead position with slight offset
                    lead_pos = lead_target['position']
                    offset_x = random.uniform(-80, 80)
                    offset_y = random.uniform(-80, 80)
                    
                    # Set position and heading based on lead
                    target['position'] = (lead_pos[0] + offset_x, lead_pos[1] + offset_y)
                    target['heading'] = lead_target['heading']
                else:
                    # Regular movement logic for non-convoy targets or convoy leads
                    characteristics = target['characteristics']
                    
                    # Check if target pauses
                    if random.random() < characteristics['pause_probability']:
                        # Target stays in place
                        pass
                    else:
                        # Check if target changes direction
                        if random.random() < characteristics['direction_change_probability']:
                            # Change heading by -60 to +60 degrees
                            target['heading'] = (target['heading'] + random.uniform(-60, 60)) % 360
                        
                        # Move target
                        new_position, new_heading = self._calculate_new_position(
                            target['position'], 
                            target['heading'], 
                            characteristics['speed'], 
                            time_delta_hours
                        )
                        
                        target['position'] = new_position
                        target['heading'] = new_heading
                
                # Record observation (with some probability of missed detections)
                if random.random() < 0.9:  # 90% chance of detection
                    self.target_observations.append({
                        'id': target_id,
                        'timestamp': current_time,
                        'x_coord': target['position'][0],
                        'y_coord': target['position'][1],
                        'target_class': target['class']
                    })
            
            # Move to next time step
            current_time += time_delta
    
    def generate_blue_forces(self, num_forces=10, observation_interval_minutes=5):
        """
        Generate blue force entities and their movement observations.
        
        Args:
            num_forces: Number of blue force entities to generate
            observation_interval_minutes: Time between observations in minutes
        """
        # Create blue forces with different patrol patterns
        for i in range(num_forces):
            force_id = f"blue-{i}"
            force_class = random.choice(list(self.BLUE_FORCE_CLASSES.keys()))
            
            # For command posts, place them strategically
            if force_class == 'command_post':
                # Place command posts away from the edges
                buffer = (self.max_x - self.min_x) * 0.2
                x = random.uniform(self.min_x + buffer, self.max_x - buffer)
                y = random.uniform(self.min_y + buffer, self.max_y - buffer)
                initial_position = (x, y)
                patrol_type = 'stationary'
                patrol_radius = 50  # Very small movement radius
            else:
                initial_position = self._generate_random_location()
                patrol_type = random.choice(['circular', 'linear', 'area'])
                
                # Patrol parameters based on type
                if patrol_type == 'circular':
                    patrol_radius = random.uniform(500, 2000)
                    patrol_center = initial_position
                elif patrol_type == 'linear':
                    patrol_length = random.uniform(2000, 5000)
                    patrol_heading = random.uniform(0, 360)
                    patrol_endpoint = self._calculate_new_position(
                        initial_position, patrol_heading, patrol_length/1000, 1
                    )[0]
                else:  # area patrol
                    patrol_area_size = random.uniform(1000, 3000)
                    patrol_center = initial_position
            
            self.blue_forces[force_id] = {
                'id': force_id,
                'class': force_class,
                'position': initial_position,
                'heading': random.uniform(0, 360),
                'patrol_type': patrol_type,
                'characteristics': self.BLUE_FORCE_CLASSES[force_class]
            }
            
            # Add patrol-specific properties
            if patrol_type == 'circular':
                self.blue_forces[force_id]['patrol_radius'] = patrol_radius
                self.blue_forces[force_id]['patrol_center'] = patrol_center
                self.blue_forces[force_id]['patrol_angle'] = 0
            elif patrol_type == 'linear':
                self.blue_forces[force_id]['patrol_start'] = initial_position
                self.blue_forces[force_id]['patrol_end'] = patrol_endpoint
                self.blue_forces[force_id]['patrol_direction'] = 1  # 1: start->end, -1: end->start
            elif patrol_type == 'area':
                self.blue_forces[force_id]['patrol_center'] = patrol_center
                self.blue_forces[force_id]['patrol_area_size'] = patrol_area_size
                self.blue_forces[force_id]['waypoint'] = self._generate_area_waypoint(
                    patrol_center, patrol_area_size
                )
        
        # Generate observations over time
        current_time = self.start_time
        time_delta = datetime.timedelta(minutes=observation_interval_minutes)
        time_delta_hours = observation_interval_minutes / 60.0
        
        while current_time <= self.end_time:
            # Update each blue force
            for force_id, force in self.blue_forces.items():
                characteristics = force['characteristics']
                
                # Check if force pauses
                if random.random() < characteristics['pause_probability']:
                    # Force stays in place
                    pass
                else:
                    # Move based on patrol type
                    if force['patrol_type'] == 'stationary':
                        # Small random movement around fixed position
                        offset_x = random.uniform(-20, 20)
                        offset_y = random.uniform(-20, 20)
                        force['position'] = (force['position'][0] + offset_x, force['position'][1] + offset_y)
                    
                    elif force['patrol_type'] == 'circular':
                        # Move in a circular pattern
                        center = force['patrol_center']
                        radius = force['patrol_radius']
                        angle = force['patrol_angle']
                        
                        # Calculate new position on circle
                        x = center[0] + radius * np.cos(np.radians(angle))
                        y = center[1] + radius * np.sin(np.radians(angle))
                        force['position'] = (x, y)
                        
                        # Update angle for next iteration (degrees)
                        speed_factor = characteristics['speed'] / 10
                        angle_increment = 10 * speed_factor  # degrees per time step
                        force['patrol_angle'] = (angle + angle_increment) % 360
                        
                        # Update heading to be tangent to circle
                        force['heading'] = (angle + 90) % 360
                    
                    elif force['patrol_type'] == 'linear':
                        # Move back and forth along a line
                        start = force['patrol_start']
                        end = force['patrol_end']
                        direction = force['patrol_direction']
                        
                        # Calculate vector from start to end
                        vec_x = end[0] - start[0]
                        vec_y = end[1] - start[1]
                        distance = np.sqrt(vec_x**2 + vec_y**2)
                        
                        # Normalize and scale by speed
                        movement_distance = characteristics['speed'] * time_delta_hours * 1000
                        move_x = vec_x / distance * movement_distance * direction
                        move_y = vec_y / distance * movement_distance * direction
                        
                        # Update position
                        new_x = force['position'][0] + move_x
                        new_y = force['position'][1] + move_y
                        force['position'] = (new_x, new_y)
                        
                        # Check if we need to reverse direction
                        if direction == 1:  # Moving from start to end
                            dist_to_end = np.sqrt((new_x - end[0])**2 + (new_y - end[1])**2)
                            if dist_to_end < movement_distance:
                                force['patrol_direction'] = -1
                                force['heading'] = (force['heading'] + 180) % 360
                        else:  # Moving from end to start
                            dist_to_start = np.sqrt((new_x - start[0])**2 + (new_y - start[1])**2)
                            if dist_to_start < movement_distance:
                                force['patrol_direction'] = 1
                                force['heading'] = (force['heading'] + 180) % 360
                    
                    elif force['patrol_type'] == 'area':
                        # Move to random waypoints in an area
                        waypoint = force['waypoint']
                        current = force['position']
                        
                        # Calculate vector to waypoint
                        vec_x = waypoint[0] - current[0]
                        vec_y = waypoint[1] - current[1]
                        distance = np.sqrt(vec_x**2 + vec_y**2)
                        
                        # If close to waypoint, select a new one
                        if distance < 100:
                            force['waypoint'] = self._generate_area_waypoint(
                                force['patrol_center'], force['patrol_area_size']
                            )
                            waypoint = force['waypoint']
                            vec_x = waypoint[0] - current[0]
                            vec_y = waypoint[1] - current[1]
                            distance = np.sqrt(vec_x**2 + vec_y**2)
                        
                        # Update heading
                        force['heading'] = np.degrees(np.arctan2(vec_y, vec_x)) % 360
                        
                        # Normalize and scale by speed
                        movement_distance = min(distance, characteristics['speed'] * time_delta_hours * 1000)
                        move_x = vec_x / distance * movement_distance
                        move_y = vec_y / distance * movement_distance
                        
                        # Update position
                        new_x = current[0] + move_x
                        new_y = current[1] + move_y
                        force['position'] = (new_x, new_y)
                
                # Record observation (blue forces have 100% detection rate)
                self.blue_force_observations.append({
                    'id': force_id,
                    'timestamp': current_time,
                    'x_coord': force['position'][0],
                    'y_coord': force['position'][1],
                    'force_class': force['class']
                })
            
            # Move to next time step
            current_time += time_delta
    
    def _generate_area_waypoint(self, center, area_size):
        """Generate a random waypoint within an area"""
        x_offset = random.uniform(-area_size/2, area_size/2)
        y_offset = random.uniform(-area_size/2, area_size/2)
        
        x = center[0] + x_offset
        y = center[1] + y_offset
        
        # Ensure within bounds
        x = max(self.min_x, min(self.max_x, x))
        y = max(self.min_y, min(self.max_y, y))
        
        return (x, y)
    
    def save_to_csv(self, output_dir='synthetic_data'):
        """
        Save the generated observations to CSV files.
        
        Args:
            output_dir: Directory to save the files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert observations to DataFrames
        target_df = pd.DataFrame(self.target_observations)
        blue_force_df = pd.DataFrame(self.blue_force_observations)
        
        # Add additional metadata
        target_df['detection_confidence'] = [random.uniform(0.7, 1.0) for _ in range(len(target_df))]
        target_df['sensor_id'] = [random.choice(['radar-1', 'camera-2', 'thermal-3', 'drone-4']) 
                                for _ in range(len(target_df))]
        
        blue_force_df['status'] = [random.choice(['active', 'active', 'active', 'resupply', 'maintenance']) 
                                 for _ in range(len(blue_force_df))]
        blue_force_df['report_type'] = [random.choice(['regular', 'regular', 'regular', 'sitrep', 'contact']) 
                                     for _ in range(len(blue_force_df))]
        
        # Save to CSV
        target_csv_path = os.path.join(output_dir, 'target_observations.csv')
        blue_force_csv_path = os.path.join(output_dir, 'blue_force_observations.csv')
        
        target_df.to_csv(target_csv_path, index=False)
        blue_force_df.to_csv(blue_force_csv_path, index=False)
        
        # Create a combined CSV with all data
        # First, rename columns in blue force dataframe to match target dataframe
        blue_force_df_renamed = blue_force_df.rename(columns={'force_class': 'entity_class'})
        target_df_renamed = target_df.rename(columns={'target_class': 'entity_class'})
        
        # Add entity type column
        blue_force_df_renamed['entity_type'] = 'blue_force'
        target_df_renamed['entity_type'] = 'target'
        
        # Combine dataframes, keeping only common columns
        common_columns = ['id', 'timestamp', 'x_coord', 'y_coord', 'entity_class', 'entity_type']
        combined_df = pd.concat([
            target_df_renamed[common_columns],
            blue_force_df_renamed[common_columns]
        ])
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')
        
        # Save combined CSV
        combined_csv_path = os.path.join(output_dir, 'combined_observations.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        
        print(f"Generated {len(target_df)} target observations and {len(blue_force_df)} blue force observations.")
        print(f"Files saved to: {output_dir}")
        
        return {
            'target_csv': target_csv_path,
            'blue_force_csv': blue_force_csv_path,
            'combined_csv': combined_csv_path
        }
    
    def generate_all_data(self, num_targets=20, num_blue_forces=10):
        """
        Generate all data with default parameters.
        
        Args:
            num_targets: Number of target entities to generate
            num_blue_forces: Number of blue force entities to generate
            
        Returns:
            Dictionary of file paths to the generated CSV files
        """
        print(f"Generating {num_targets} targets...")
        self.generate_targets(num_targets=num_targets)
        
        print(f"Generating {num_blue_forces} blue forces...")
        self.generate_blue_forces(num_forces=num_blue_forces)
        
        return self.save_to_csv()

def analyze_generated_data(csv_paths):
    """
    Analyze the generated data and print summary statistics.
    
    Args:
        csv_paths: Dictionary of CSV file paths from the generator
    """
    # Load the data
    target_df = pd.read_csv(csv_paths['target_csv'])
    blue_force_df = pd.read_csv(csv_paths['blue_force_csv'])
    combined_df = pd.read_csv(csv_paths['combined_csv'])
    
    # Print basic stats
    print("\n=== Data Summary ===")
    print(f"Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"Total observations: {len(combined_df)}")
    print(f"  - Target observations: {len(target_df)}")
    print(f"  - Blue force observations: {len(blue_force_df)}")
    
    # Unique entities
    unique_targets = target_df['id'].nunique()
    unique_blue_forces = blue_force_df['id'].nunique()
    print(f"\nUnique entities: {unique_targets + unique_blue_forces}")
    print(f"  - Targets: {unique_targets}")
    print(f"  - Blue forces: {unique_blue_forces}")
    
    # Entity classes
    print("\nTarget classes:")
    for cls, count in target_df['target_class'].value_counts().items():
        print(f"  - {cls}: {count} observations")
    
    print("\nBlue force classes:")
    for cls, count in blue_force_df['force_class'].value_counts().items():
        print(f"  - {cls}: {count} observations")
    
    # Spatial distribution
    x_min, y_min = combined_df['x_coord'].min(), combined_df['y_coord'].min()
    x_max, y_max = combined_df['x_coord'].max(), combined_df['y_coord'].max()
    print(f"\nSpatial coverage: ({x_min:.1f}, {y_min:.1f}) to ({x_max:.1f}, {y_max:.1f})")
    
    # Observation frequency
    obs_counts = combined_df.groupby('id').size()
    print(f"\nObservations per entity: Min={obs_counts.min()}, Avg={obs_counts.mean():.1f}, Max={obs_counts.max()}")
    
    # Additional metadata
    print("\nSensor distribution:")
    for sensor, count in target_df['sensor_id'].value_counts().items():
        print(f"  - {sensor}: {count} observations")
    
    print("\nBlue force status:")
    for status, count in blue_force_df['status'].value_counts().items():
        print(f"  - {status}: {count} observations")

def generate_demonstration_data():
    """Generate a demonstration dataset with default parameters"""
    # Create a generator with a smaller area for testing
    generator = SyntheticDataGenerator(
        area_bounds=(0, 0, 5000, 5000),
        start_time=datetime.datetime(2023, 5, 1, 0, 0, 0),
        duration_hours=24,
        seed=42
    )
    
    # Generate data
    csv_paths = generator.generate_all_data(num_targets=15, num_blue_forces=8)
    
    # Analyze the data
    analyze_generated_data(csv_paths)
    
    return csv_paths


if __name__ == "__main__":
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        area_bounds=(0, 0, 10000, 10000),  # 10km x 10km area
        start_time=datetime.datetime(2023, 5, 1, 0, 0, 0),
        duration_hours=48,  # 2 days of data
        seed=42  # For reproducibility
    )

    # Generate data
    generator.generate_targets(num_targets=20, observation_interval_minutes=15)
    generator.generate_blue_forces(num_forces=10, observation_interval_minutes=5)

    # Save to CSV files
    csv_files = generator.save_to_csv(output_dir="synthetic_data")

    print("Generated files:")
    for key, path in csv_files.items():
        print(f"  - {key}: {path}")