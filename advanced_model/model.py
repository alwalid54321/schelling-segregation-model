"""
Advanced Schelling Segregation Model implementation.
"""

import numpy as np
import random
from .agent import Agent
import time
from collections import defaultdict



class AdvancedSchellingModel:
    """
    Advanced implementation of the Schelling segregation model with:
    - Heterogeneous agent characteristics (income, education, age)
    - Dynamic homophily thresholds based on agent attributes
    - Social networks between agents
    - Geographic barriers and premium locations
    - Multiple neighborhood factors influencing happiness
    """
    
    def __init__(self, width=50, height=50, density=0.8, global_homophily=0.3, 
                 num_agent_types=2, pattern_type="alternating", shape="square",
                 use_networks=True, geographic_features=True, income_segregation=True,
                 use_hubs=True):
        """
        Initialize the advanced Schelling model.
        
        Args:
            width, height: Dimensions of the grid
            density: Proportion of cells to be filled with agents
            global_homophily: Base homophily threshold for all agents
            num_agent_types: Number of different agent types
            pattern_type: Initial pattern of agents
            shape: Shape of the grid (square, circle, human)
            use_networks: Whether to use social networks between agents
            geographic_features: Whether to include geographic features in the grid
            income_segregation: Whether income should influence location preferences
        """
        self.width = width
        self.height = height
        self.density = density
        self.global_homophily = global_homophily
        self.num_agent_types = num_agent_types
        self.pattern_type = pattern_type
        self.shape = shape
        self.use_networks = use_networks
        self.geographic_features = geographic_features
        self.income_segregation = income_segregation
        self.use_hubs = use_hubs
        
        # Create grid and agents
        self.grid = [[None for _ in range(height)] for _ in range(width)]
        self.agents = {}  # Dictionary of all agents by ID
        self.barriers = []  # List of barrier positions (x, y)
        self.premium_locations = []  # List of premium location positions (x, y)
        
        # Simulation state
        self.is_running = False
        self.steps = 0
        self.total_moves = 0
        self.unhappy_agents = []
        
        # Metrics
        self.happiness_data = []
        self.segregation_data = []
        self.avg_moves_data = []
        self.type_distributions = []  # Distribution of agent types over time
        self.income_segregation_index = []  # Measure of income segregation
        
        # Hub metrics
        self.hub_segregation_data = []  # Segregation in activity hubs
        self.hub_income_segregation_data = []  # Income segregation in hubs
        
        # Create shape mask and geographic features
        self.mask = self._create_shape_mask()
        if self.geographic_features:
            self._create_geographic_features()
        
        # Initialize the grid with agents
        self.initialize_grid()
        
        # Make all agents initially unhappy
        self._make_all_agents_unhappy()
        
        # Create social networks if enabled
        if self.use_networks:
            self._create_social_networks()
        
        # Create urban activity hubs if enabled
        if self.use_hubs:
            self.initialize_hubs()
            self.assign_agents_to_hubs()
        
        # Collect initial metrics
        self.collect_data()
    
    def _create_shape_mask(self):
        """Create a mask for the grid based on the specified shape."""
        mask = np.ones((self.width, self.height), dtype=bool)
        
        if self.shape == "square":
            # Square is default (all cells available)
            pass
        
        elif self.shape == "circle":
            # Create a circular mask
            center_x, center_y = self.width // 2, self.height // 2
            radius = min(center_x, center_y) - 2  # Leave a small border
            
            for x in range(self.width):
                for y in range(self.height):
                    # If outside the circle, mask it out
                    if (x - center_x)**2 + (y - center_y)**2 > radius**2:
                        mask[x, y] = False
        
        elif self.shape == "human":
            # Create a simple human-like shape (head and body)
            center_x, center_y = self.width // 2, self.height // 2
            head_radius = min(center_x, center_y) // 3
            head_center_y = center_y - head_radius
            
            # Start with all masked out
            mask.fill(False)
            
            # Create head (circle)
            for x in range(self.width):
                for y in range(self.height):
                    # If inside the head circle
                    if (x - center_x)**2 + (y - head_center_y)**2 <= head_radius**2:
                        mask[x, y] = True
            
            # Create body (rectangle)
            body_width = head_radius * 2
            body_height = self.height // 2
            body_top = head_center_y + head_radius
            body_left = center_x - body_width // 2
            
            for x in range(body_left, body_left + body_width):
                for y in range(body_top, body_top + body_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
            
            # Create arms (rectangles)
            arm_width = self.width // 4
            arm_height = head_radius
            arm_top = body_top + head_radius
            
            # Left arm
            for x in range(body_left - arm_width, body_left):
                for y in range(arm_top, arm_top + arm_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
            
            # Right arm
            for x in range(body_left + body_width, body_left + body_width + arm_width):
                for y in range(arm_top, arm_top + arm_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
            
            # Create legs (rectangles)
            leg_width = head_radius
            leg_height = self.height // 4
            leg_top = body_top + body_height
            
            # Left leg
            left_leg_left = center_x - leg_width - leg_width // 2
            for x in range(left_leg_left, left_leg_left + leg_width):
                for y in range(leg_top, leg_top + leg_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
            
            # Right leg
            right_leg_left = center_x + leg_width // 2
            for x in range(right_leg_left, right_leg_left + leg_width):
                for y in range(leg_top, leg_top + leg_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
        
        return mask
    
    def _create_geographic_features(self):
        """Create geographic features like barriers and premium locations."""
        # Create barriers (e.g., rivers, highways)
        num_barriers = random.randint(1, 3)
        for _ in range(num_barriers):
            # Decide if horizontal or vertical barrier
            if random.random() < 0.5:
                # Horizontal barrier
                y = random.randint(5, self.height - 5)
                width = random.randint(self.width // 2, self.width - 5)
                start_x = random.randint(0, self.width - width)
                
                for x in range(start_x, start_x + width):
                    self.barriers.append((x, y))
                    # Small chance to have gaps in the barrier
                    if random.random() < 0.1:
                        self.barriers.pop()
            else:
                # Vertical barrier
                x = random.randint(5, self.width - 5)
                height = random.randint(self.height // 2, self.height - 5)
                start_y = random.randint(0, self.height - height)
                
                for y in range(start_y, start_y + height):
                    self.barriers.append((x, y))
                    # Small chance to have gaps in the barrier
                    if random.random() < 0.1:
                        self.barriers.pop()
        
        # Create premium locations (e.g., lakes, parks, downtown)
        num_premium = random.randint(2, 4)
        for _ in range(num_premium):
            center_x = random.randint(5, self.width - 5)
            center_y = random.randint(5, self.height - 5)
            radius = random.randint(2, 5)
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        x, y = center_x + dx, center_y + dy
                        if 0 <= x < self.width and 0 <= y < self.height:
                            self.premium_locations.append((x, y))
        
        # Remove any overlaps between barriers and premium locations
        self.premium_locations = [pos for pos in self.premium_locations if pos not in self.barriers]
        
        # Update mask to account for barriers
        for x, y in self.barriers:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.mask[x, y] = False
