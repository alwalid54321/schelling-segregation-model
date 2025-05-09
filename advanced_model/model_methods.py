"""
Additional methods for the Advanced Schelling Segregation Model.
"""

import random
import numpy as np
from .agent import Agent


def initialize_grid(self):
    """Initialize the grid with agents based on the specified pattern."""
    agent_id = 0
    
    # Choose initialization method based on pattern_type
    if self.pattern_type == "random":
        self._initialize_random(agent_id)
    elif self.pattern_type == "checkerboard":
        self._initialize_checkerboard(agent_id)
    elif self.pattern_type == "alternating":
        self._initialize_alternating(agent_id)
    elif self.pattern_type == "stripes":
        self._initialize_stripes(agent_id)
    elif self.pattern_type == "clusters":
        self._initialize_clusters(agent_id)
    elif self.pattern_type == "income_stratified":
        self._initialize_income_stratified(agent_id)
    else:
        raise ValueError(f"Unknown pattern type: {self.pattern_type}")


def _initialize_random(self, agent_id_start=0):
    """Initialize with a random distribution of agent types."""
    agent_id = agent_id_start
    
    for x in range(self.width):
        for y in range(self.height):
            # Only place agents where the mask allows and not on barriers
            if self.mask[x, y] and (x, y) not in self.barriers and random.random() < self.density:
                # Create agent with random type
                agent_type = random.randint(0, self.num_agent_types - 1)
                
                # Create agent with some randomized characteristics
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    x=x, y=y,
                    income=random.normalvariate(50, 20),
                    education=random.normalvariate(12, 4),
                    age=random.normalvariate(35, 15),
                    tolerance=random.uniform(0.2, 0.6)
                )
                agent_id += 1
                
                # Add agent to grid and dictionary
                self.grid[x][y] = agent
                self.agents[agent.id] = agent


def _initialize_alternating(self, agent_id_start=0):
    """Initialize with a strict alternating pattern of agent types."""
    agent_id = agent_id_start
    current_type = 0
    
    for x in range(self.width):
        for y in range(self.height):
            # Only place agents where the mask allows and not on barriers
            if self.mask[x, y] and (x, y) not in self.barriers and random.random() < self.density:
                # Create agent with alternating type
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=current_type,
                    x=x, y=y,
                    income=random.normalvariate(50, 20),
                    education=random.normalvariate(12, 4),
                    age=random.normalvariate(35, 15),
                    tolerance=random.uniform(0.2, 0.6)
                )
                agent_id += 1
                
                # Add agent to grid and dictionary
                self.grid[x][y] = agent
                self.agents[agent.id] = agent
                
                # Switch to next type for perfect alternation
                current_type = (current_type + 1) % self.num_agent_types


def _initialize_checkerboard(self, agent_id_start=0):
    """Initialize with a checkerboard pattern."""
    agent_id = agent_id_start
    
    for x in range(self.width):
        for y in range(self.height):
            # Only place agents where the mask allows and not on barriers
            if self.mask[x, y] and (x, y) not in self.barriers and random.random() < self.density:
                # Create agent with checkerboard pattern type
                agent_type = (x + y) % self.num_agent_types
                
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    x=x, y=y,
                    income=random.normalvariate(50, 20),
                    education=random.normalvariate(12, 4),
                    age=random.normalvariate(35, 15),
                    tolerance=random.uniform(0.2, 0.6)
                )
                agent_id += 1
                
                # Add agent to grid and dictionary
                self.grid[x][y] = agent
                self.agents[agent.id] = agent


def _initialize_stripes(self, agent_id_start=0):
    """Initialize with horizontal stripes of agent types."""
    agent_id = agent_id_start
    
    for x in range(self.width):
        for y in range(self.height):
            # Only place agents where the mask allows and not on barriers
            if self.mask[x, y] and (x, y) not in self.barriers and random.random() < self.density:
                # Create stripes based on y coordinate
                stripe_width = max(1, self.height // (self.num_agent_types * 2))
                agent_type = (y // stripe_width) % self.num_agent_types
                
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    x=x, y=y,
                    income=random.normalvariate(50, 20),
                    education=random.normalvariate(12, 4),
                    age=random.normalvariate(35, 15),
                    tolerance=random.uniform(0.2, 0.6)
                )
                agent_id += 1
                
                # Add agent to grid and dictionary
                self.grid[x][y] = agent
                self.agents[agent.id] = agent


def _initialize_clusters(self, agent_id_start=0):
    """Initialize with clusters of similar agent types."""
    agent_id = agent_id_start
    
    # Create cluster centers
    centers = []
    for i in range(self.num_agent_types):
        # Create multiple centers for each type
        for _ in range(3):  # 3 clusters per type
            centers.append({
                'x': random.randint(0, self.width - 1),
                'y': random.randint(0, self.height - 1),
                'type': i
            })
    
    for x in range(self.width):
        for y in range(self.height):
            # Only place agents where the mask allows and not on barriers
            if self.mask[x, y] and (x, y) not in self.barriers and random.random() < self.density:
                # Find the closest center and use its type
                min_dist = float('inf')
                chosen_type = 0
                
                for center in centers:
                    dist = (x - center['x'])**2 + (y - center['y'])**2
                    if dist < min_dist:
                        min_dist = dist
                        chosen_type = center['type']
                
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=chosen_type,
                    x=x, y=y,
                    income=random.normalvariate(50, 20),
                    education=random.normalvariate(12, 4),
                    age=random.normalvariate(35, 15),
                    tolerance=random.uniform(0.2, 0.6)
                )
                agent_id += 1
                
                # Add agent to grid and dictionary
                self.grid[x][y] = agent
                self.agents[agent.id] = agent


def _initialize_income_stratified(self, agent_id_start=0):
    """Initialize with income-based stratification (premium areas for high income)."""
    agent_id = agent_id_start
    
    for x in range(self.width):
        for y in range(self.height):
            # Only place agents where the mask allows and not on barriers
            if self.mask[x, y] and (x, y) not in self.barriers and random.random() < self.density:
                # Determine agent type (random)
                agent_type = random.randint(0, self.num_agent_types - 1)
                
                # Determine income based on location (center = higher income)
                x_center_dist = abs(x - self.width/2) / (self.width/2)
                y_center_dist = abs(y - self.height/2) / (self.height/2)
                center_dist = (x_center_dist**2 + y_center_dist**2)**0.5 / 1.414
                
                # Adjust income based on distance from center and premium locations
                base_income = 70 - 40 * center_dist
                
                # Premium locations boost income
                if (x, y) in self.premium_locations:
                    base_income += 20
                
                # Add some randomness
                income = random.normalvariate(base_income, 10)
                
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    x=x, y=y,
                    income=income,
                    education=random.normalvariate(12, 4),
                    age=random.normalvariate(35, 15),
                    tolerance=random.uniform(0.2, 0.6)
                )
                agent_id += 1
                
                # Add agent to grid and dictionary
                self.grid[x][y] = agent
                self.agents[agent.id] = agent


def _make_all_agents_unhappy(self):
    """Make all agents initially unhappy."""
    self.unhappy_agents = []
    for agent_id, agent in self.agents.items():
        agent.is_happy = False
        self.unhappy_agents.append(agent)


def _create_social_networks(self):
    """Create social networks between agents."""
    # Each agent has a chance to connect with others
    # Connections are more likely between agents of the same type
    for agent_id, agent in self.agents.items():
        # Determine number of connections for this agent (random)
        num_connections = random.randint(1, 10)
        
        # Sort other agents by similarity and proximity
        other_agents = []
        for other_id, other in self.agents.items():
            if other_id != agent_id:
                # Calculate distance
                distance = ((agent.x - other.x)**2 + (agent.y - other.y)**2)**0.5
                
                # Calculate similarity
                type_match = 1 if agent.type == other.type else 0
                
                # Higher score = more likely to connect
                score = type_match * 2 + (1 / (distance + 1))
                other_agents.append((other, score))
        
        # Sort by score (highest first)
        other_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Create connections
        for other, _ in other_agents[:num_connections]:
            agent.connect_with(other)
            other.connect_with(agent)


def get_neighbors(self, x, y, radius=1):
    """Get the neighbors of a cell within the specified radius."""
    neighbors = []
    
    # Check cells in square around (x,y) with given radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue  # Skip the center cell
            
            nx = (x + dx) % self.width  # Wrap around the grid
            ny = (y + dy) % self.height
            
            # Consider barriers when getting neighbors
            if self.geographic_features and (nx, ny) in self.barriers:
                continue  # Skip barrier cells
            
            if self.grid[nx][ny] is not None:
                neighbors.append(self.grid[nx][ny])
    
    return neighbors


def step(self):
    """Run one step of the model."""
    self.unhappy_agents = []
    
    # Check happiness of all agents
    for agent_id, agent in self.agents.items():
        neighbors = self.get_neighbors(agent.x, agent.y)
        agent.calculate_happiness(neighbors, self.global_homophily)
        
        if not agent.is_happy:
            self.unhappy_agents.append(agent)
    
    # Move unhappy agents
    self.move_unhappy_agents()
    
    # Increase attachment for agents that didn't move
    for agent_id, agent in self.agents.items():
        if agent.is_happy:
            agent.increase_attachment()
    
    # Collect metrics
    self.collect_data()
    
    self.steps += 1
    
    # Return the grid for animation
    return self.get_grid_values()
