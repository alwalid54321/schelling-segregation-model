"""
Model module for Mesa-based Advanced Schelling Segregation Model
"""

import numpy as np
import random
from mesa import Model
from mesa.time import RandomActivation
print("Import successful!")
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from agent import SchellingAgent



class SchellingModel(Model):
    """
    Advanced Schelling segregation model with:
    - Multiple agent types with heterogeneous characteristics
    - Geographic barriers and premium locations
    - Hub-based interactions (work, education, leisure)
    - Social networks between agents
    - Shape masks (square, circle, human)
    """
    
    def __init__(
        self, width=50, height=50, density=0.8, homophily=0.3, 
        num_agent_types=2, pattern_type="alternating", shape="square",
        use_networks=True, geographic_features=True, income_segregation=True,
        use_hubs=True):
        """
        Initialize a new Schelling segregation model.
        
        Args:
            width, height: Grid dimensions
            density: Proportion of cells to populate with agents
            homophily: Base homophily threshold for agent happiness
            num_agent_types: Number of different agent types
            pattern_type: Initial distribution of agents (random, alternating, etc.)
            shape: Shape of the inhabitable grid (square, circle, human)
            use_networks: Whether to create social networks between agents
            geographic_features: Whether to add geographic barriers and premium areas
            income_segregation: Whether income should affect location preferences
            use_hubs: Whether to create activity hubs (work, education, leisure)
        """
        super().__init__()
        
        # Model parameters
        self.width = width
        self.height = height
        self.density = density
        self.homophily = homophily
        self.num_agent_types = num_agent_types
        self.pattern_type = pattern_type
        self.shape = shape
        self.use_networks = use_networks
        self.geographic_features = geographic_features
        self.income_segregation = income_segregation
        self.use_hubs = use_hubs
        
        # Initialize model components
        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        
        # Initialize shape, barriers, premium locations
        self.mask = self._create_shape_mask()
        self.barriers = set()
        self.premium_locations = set()
        
        if self.geographic_features:
            self._create_geographic_features()
        
        # Metrics
        self.total_moves = 0
        self.unhappy_agents = []
        
        # Create agents according to the specified pattern
        self._initialize_agents()
        
        # Create social networks if enabled
        if self.use_networks:
            self._create_social_networks()
        
        # Create hubs if enabled
        if self.use_hubs:
            self.initialize_hubs()
            self.assign_agents_to_hubs()
        
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Happiness": lambda m: self.get_happiness_ratio(),
                "Segregation": lambda m: self.calculate_segregation_index(),
                "Income_Segregation": lambda m: self.calculate_income_segregation_index(),
                "Moves": lambda m: self.total_moves,
                "Unhappy": lambda m: len(self.unhappy_agents)
            },
            agent_reporters={
                "Type": lambda a: a.type,
                "Income": lambda a: a.income, 
                "Happiness": lambda a: a.is_happy,
                "Tolerance": lambda a: a.tolerance
            }
        )
        
        # Collect initial data
        self.running = True
        self.datacollector.collect(self)
    
    def _create_shape_mask(self):
        """Create a boolean mask for the inhabitable area based on the shape."""
        mask = np.ones((self.width, self.height), dtype=bool)
        
        if self.shape == "circle":
            # Create a circular mask
            center_x, center_y = self.width // 2, self.height // 2
            radius = min(center_x, center_y) - 2
            
            for x in range(self.width):
                for y in range(self.height):
                    if ((x - center_x) ** 2 + (y - center_y) ** 2) > radius ** 2:
                        mask[x, y] = False
        
        elif self.shape == "human":
            # Create a simple human-like shape (head and body)
            center_x, center_y = self.width // 2, self.height // 2
            head_radius = min(center_x, center_y) // 3
            
            for x in range(self.width):
                for y in range(self.height):
                    # The "head" is a circle at the top
                    in_head = ((x - center_x) ** 2 + (y - (center_y - head_radius)) ** 2) <= head_radius ** 2
                    
                    # The "body" is an oval below the head
                    body_width = head_radius * 1.5
                    body_height = head_radius * 2.5
                    x_rel, y_rel = x - center_x, y - center_y
                    in_body = (x_rel / body_width) ** 2 + ((y_rel + head_radius // 2) / body_height) ** 2 <= 1
                    
                    if not (in_head or in_body):
                        mask[x, y] = False
        
        return mask
    
    def _create_geographic_features(self):
        """Create geographic barriers (rivers, highways) and premium locations."""
        # Create a river (horizontal or vertical)
        is_horizontal = self.random.choice([True, False])
        if is_horizontal:
            river_y = self.random.randint(self.height // 4, 3 * self.height // 4)
            for x in range(self.width):
                if x % 5 != 0:  # Leave some bridges
                    self.barriers.add((x, river_y))
        else:
            river_x = self.random.randint(self.width // 4, 3 * self.width // 4)
            for y in range(self.height):
                if y % 5 != 0:  # Leave some bridges
                    self.barriers.add((river_x, y))
        
        # Create premium locations (parks, waterfronts)
        # Central premium area (like a park)
        center_x, center_y = self.width // 2, self.height // 2
        for x in range(center_x - 3, center_x + 3):
            for y in range(center_y - 3, center_y + 3):
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.premium_locations.add((x, y))
    
    def is_cell_empty(self, pos):
        """Check if a cell is empty and within the mask."""
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if not self.mask[x, y]:
            return False
        return self.grid.is_cell_empty(pos)
    
    def is_cell_barrier(self, pos):
        """Check if a cell is a barrier."""
        return pos in self.barriers
    
    def get_location_quality(self, pos):
        """
        Calculate the quality of a location.
        Premium locations and central positions have higher quality.
        """
        x, y = pos
        
        # Check if it's a premium location
        if pos in self.premium_locations:
            return 0.9  # Premium locations are high quality
            
        # Calculate distance from center as a factor of quality
        center_x, center_y = self.width // 2, self.height // 2
        max_distance = ((self.width/2)**2 + (self.height/2)**2)**0.5
        distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
        
        # Normalize distance to 0-1 range and invert (closer to center = higher quality)
        distance_factor = 1 - (distance / max_distance)
        
        return distance_factor * 0.7  # Scale to leave room for premium locations
        
    def _initialize_agents(self):
        """Initialize agents according to the specified pattern."""
        if self.pattern_type == "random":
            self._initialize_random()
        elif self.pattern_type == "alternating":
            self._initialize_alternating()
        elif self.pattern_type == "checkerboard":
            self._initialize_checkerboard()
        elif self.pattern_type == "stripes":
            self._initialize_stripes()
        elif self.pattern_type == "clusters":
            self._initialize_clusters()
        elif self.pattern_type == "income_stratified":
            self._initialize_income_stratified()
        else:
            self._initialize_random()  # Default to random
            
    def _initialize_random(self):
        """Initialize with a random distribution of agent types."""
        # Place agents randomly within the mask and not on barriers
        for x in range(self.width):
            for y in range(self.height):
                # Check if cell is valid for agent placement
                if self.mask[x, y] and (x, y) not in self.barriers and self.random.random() < self.density:
                    # Randomly select agent type
                    agent_type = self.random.randint(0, self.num_agent_types - 1)
                    
                    # Create agent with randomized characteristics
                    agent = SchellingAgent(
                        pos=(x, y),
                        model=self,
                        agent_type=agent_type,
                        income=self.random.normalvariate(50, 20),
                        education=self.random.normalvariate(12, 4),
                        age=self.random.normalvariate(35, 15),
                        tolerance=self.random.uniform(0.2, 0.6)
                    )
                    
                    # Add agent to grid and scheduler
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
    
    def _initialize_alternating(self):
        """Initialize with alternating agent types."""
        current_type = 0
        
        for x in range(self.width):
            for y in range(self.height):
                if self.mask[x, y] and (x, y) not in self.barriers and self.random.random() < self.density:
                    # Create agent with alternating type
                    agent = SchellingAgent(
                        pos=(x, y),
                        model=self,
                        agent_type=current_type,
                        income=self.random.normalvariate(50, 20),
                        education=self.random.normalvariate(12, 4),
                        age=self.random.normalvariate(35, 15),
                        tolerance=self.random.uniform(0.2, 0.6)
                    )
                    
                    # Add agent to grid and scheduler
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
                    
                    # Switch to next type for perfect alternation
                    current_type = (current_type + 1) % self.num_agent_types
    
    def _initialize_checkerboard(self):
        """Initialize with a checkerboard pattern of agent types."""
        for x in range(self.width):
            for y in range(self.height):
                if self.mask[x, y] and (x, y) not in self.barriers and self.random.random() < self.density:
                    # Assign type based on checkerboard pattern
                    agent_type = (x + y) % self.num_agent_types
                    
                    agent = SchellingAgent(
                        pos=(x, y),
                        model=self,
                        agent_type=agent_type,
                        income=self.random.normalvariate(50, 20),
                        education=self.random.normalvariate(12, 4),
                        age=self.random.normalvariate(35, 15),
                        tolerance=self.random.uniform(0.2, 0.6)
                    )
                    
                    # Add agent to grid and scheduler
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
    
    def _initialize_stripes(self):
        """Initialize with horizontal stripes of agent types."""
        stripe_height = max(1, self.height // (self.num_agent_types * 2))
        
        for x in range(self.width):
            for y in range(self.height):
                if self.mask[x, y] and (x, y) not in self.barriers and self.random.random() < self.density:
                    # Assign type based on horizontal stripes
                    agent_type = (y // stripe_height) % self.num_agent_types
                    
                    agent = SchellingAgent(
                        pos=(x, y),
                        model=self,
                        agent_type=agent_type,
                        income=self.random.normalvariate(50, 20),
                        education=self.random.normalvariate(12, 4),
                        age=self.random.normalvariate(35, 15),
                        tolerance=self.random.uniform(0.2, 0.6)
                    )
                    
                    # Add agent to grid and scheduler
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
    
    def _initialize_clusters(self):
        """Initialize with clusters of similar agent types."""
        # Create several cluster centers for each type
        clusters_per_type = 3
        cluster_centers = []
        
        for agent_type in range(self.num_agent_types):
            for _ in range(clusters_per_type):
                # Find a valid location for the cluster center
                while True:
                    x = self.random.randint(0, self.width - 1)
                    y = self.random.randint(0, self.height - 1)
                    if self.mask[x, y] and (x, y) not in self.barriers:
                        cluster_centers.append((x, y, agent_type))
                        break
        
        # Place agents with higher probability near their type's cluster centers
        for x in range(self.width):
            for y in range(self.height):
                if self.mask[x, y] and (x, y) not in self.barriers and self.random.random() < self.density:
                    # Calculate distances to each cluster center
                    distances = []
                    for cx, cy, ctype in cluster_centers:
                        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                        distances.append((dist, ctype))
                    
                    # Sort by distance (closest first)
                    distances.sort()
                    
                    # Assign type with probability influenced by distance to cluster centers
                    if self.random.random() < 0.7:  # 70% chance to use the closest cluster's type
                        agent_type = distances[0][1]
                    else:  # 30% chance to be random
                        agent_type = self.random.randint(0, self.num_agent_types - 1)
                    
                    agent = SchellingAgent(
                        pos=(x, y),
                        model=self,
                        agent_type=agent_type,
                        income=self.random.normalvariate(50, 20),
                        education=self.random.normalvariate(12, 4),
                        age=self.random.normalvariate(35, 15),
                        tolerance=self.random.uniform(0.2, 0.6)
                    )
                    
                    # Add agent to grid and scheduler
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
    
    def _initialize_income_stratified(self):
        """Initialize with income-based stratification."""
        center_x, center_y = self.width // 2, self.height // 2
        max_distance = ((self.width/2)**2 + (self.height/2)**2)**0.5
        
        for x in range(self.width):
            for y in range(self.height):
                if self.mask[x, y] and (x, y) not in self.barriers and self.random.random() < self.density:
                    # Calculate distance from center
                    dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
                    dist_normalized = dist / max_distance  # 0 to 1
                    
                    # Assign type randomly
                    agent_type = self.random.randint(0, self.num_agent_types - 1)
                    
                    # Assign income based on distance from center (higher income near center)
                    income = 90 - 70 * dist_normalized + self.random.normalvariate(0, 10)
                    income = max(0, min(100, income))
                    
                    # Premium locations boost income
                    if (x, y) in self.premium_locations:
                        income = min(100, income + 10)
                    
                    agent = SchellingAgent(
                        pos=(x, y),
                        model=self,
                        agent_type=agent_type,
                        income=income,
                        education=self.random.normalvariate(12, 4),
                        age=self.random.normalvariate(35, 15),
                        tolerance=self.random.uniform(0.2, 0.6)
                    )
                    
                    # Add agent to grid and scheduler
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
    
    def _create_social_networks(self):
        """Create social networks between agents."""
        agents_list = self.schedule.agents
        
        # Each agent forms connections with other agents
        for agent in agents_list:
            # Determine number of connections for this agent
            num_connections = self.random.randint(1, 10)
            
            # Sort other agents by similarity and proximity
            other_agents = []
            for other in agents_list:
                if other.unique_id != agent.unique_id:
                    # Calculate distance
                    x1, y1 = agent.pos
                    x2, y2 = other.pos
                    distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                    
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
    
    def initialize_hubs(self):
        """Initialize activity hubs in the model."""
        # Import ActivityHub here to avoid circular imports
        from .hub import ActivityHub
        
        self.hubs = []
        
        # Hub types and characteristics
        hub_types = ['work', 'education', 'leisure', 'shopping', 'transit']
        
        # Number of hubs of each type
        hub_counts = {
            'work': 3,
            'education': 2,
            'leisure': 4,
            'shopping': 3,
            'transit': 2
        }
        
        # Hub characteristics by type
        hub_characteristics = {
            'work': {'radius': 5, 'capacity': 30, 'importance': 0.8},
            'education': {'radius': 4, 'capacity': 25, 'importance': 0.7},
            'leisure': {'radius': 3, 'capacity': 20, 'importance': 0.5},
            'shopping': {'radius': 3, 'capacity': 25, 'importance': 0.4},
            'transit': {'radius': 2, 'capacity': 40, 'importance': 0.6}
        }
        
        # Generate hubs
        hub_id = 0
        for hub_type in hub_types:
            count = hub_counts[hub_type]
            chars = hub_characteristics[hub_type]
            
            for _ in range(count):
                # Find a valid location for the hub
                while True:
                    x = self.random.randint(0, self.width - 1)
                    y = self.random.randint(0, self.height - 1)
                    
                    # Hubs should be on valid locations and not too close to other hubs
                    if not self.mask[x, y] or (x, y) in self.barriers:
                        continue
                        
                    # Check if too close to other hubs
                    too_close = False
                    for other_hub in self.hubs:
                        dist = ((x - other_hub.x)**2 + (y - other_hub.y)**2)**0.5
                        if dist < chars['radius'] + other_hub.radius:
                            too_close = True
                            break
                            
                    if not too_close:
                        break
                
                # Create the hub
                hub = ActivityHub(
                    hub_id=hub_id,
                    hub_type=hub_type,
                    x=x,
                    y=y,
                    radius=chars['radius'],
                    capacity=chars['capacity'],
                    importance=chars['importance'],
                    model=self
                )
                
                self.hubs.append(hub)
                hub_id += 1
        
        # Initialize metrics
        self.hub_segregation_data = []
        self.hub_income_segregation_data = []
    
    def assign_agents_to_hubs(self):
        """Assign agents to appropriate hubs based on preferences and demographics."""
        if not hasattr(self, 'hubs') or not self.hubs:
            return
            
        # Clear existing assignments
        for hub in self.hubs:
            hub.agents.clear()
            hub.agent_types.clear()
            hub.agent_incomes = []
        
        # Group hubs by type
        hubs_by_type = defaultdict(list)
        for hub in self.hubs:
            hubs_by_type[hub.type].append(hub)
        
        # Assign agents to hubs
        for agent in self.schedule.agents:
            # Work hubs - working age agents (18-65) get assigned to work hubs
            if 18 <= agent.age <= 65:
                work_hubs = hubs_by_type['work']
                if work_hubs and not all(hub.is_full() for hub in work_hubs):
                    # Assign to closest non-full work hub
                    potential_hubs = [hub for hub in work_hubs if not hub.is_full()]
                    if potential_hubs:
                        closest_hub = min(potential_hubs, key=lambda h: ((h.x - agent.pos[0])**2 + (h.y - agent.pos[1])**2))
                        closest_hub.add_agent(agent)
            
            # Education hubs - younger agents get assigned to education hubs
            if agent.age < 30:
                education_hubs = hubs_by_type['education']
                priority = 1.0 if agent.age < 18 else 0.5  # Higher priority for children
                if education_hubs and self.random.random() < priority:
                    # Assign to a non-full education hub if available
                    potential_hubs = [hub for hub in education_hubs if not hub.is_full()]
                    if potential_hubs:
                        chosen_hub = self.random.choice(potential_hubs)
                        chosen_hub.add_agent(agent)
            
            # Leisure hubs - everyone has some chance to be assigned to leisure hubs
            leisure_hubs = hubs_by_type['leisure']
            if leisure_hubs and self.random.random() < 0.7:  # 70% chance
                # Assign to a non-full leisure hub if available
                potential_hubs = [hub for hub in leisure_hubs if not hub.is_full()]
                if potential_hubs:
                    chosen_hub = self.random.choice(potential_hubs)
                    chosen_hub.add_agent(agent)
            
            # Shopping hubs - chance increases with income
            shopping_hubs = hubs_by_type['shopping']
            if shopping_hubs and self.random.random() < 0.3 + 0.4 * (agent.income / 100):
                # Assign to a non-full shopping hub if available
                potential_hubs = [hub for hub in shopping_hubs if not hub.is_full()]
                if potential_hubs:
                    chosen_hub = self.random.choice(potential_hubs)
                    chosen_hub.add_agent(agent)
            
            # Transit hubs - higher probability for working-age agents
            transit_hubs = hubs_by_type['transit']
            transit_prob = 0.6 if 18 <= agent.age <= 65 else 0.3
            if transit_hubs and self.random.random() < transit_prob:
                # Assign to a non-full transit hub if available
                potential_hubs = [hub for hub in transit_hubs if not hub.is_full()]
                if potential_hubs:
                    chosen_hub = self.random.choice(potential_hubs)
                    chosen_hub.add_agent(agent)
        
        # Calculate segregation metrics for all hubs
        for hub in self.hubs:
            hub.calculate_hub_segregation(self.num_agent_types)
    
    def get_hub_influence(self, agent, x, y):
        """Calculate how hubs influence an agent's happiness at a location."""
        if not hasattr(self, 'hubs') or not self.hubs:
            return 0.0
        
        influence = 0.0
        agent_hubs = [hub for hub in self.hubs if agent.unique_id in hub.agents]
        
        if not agent_hubs:
            return 0.0
        
        for hub in agent_hubs:
            # Distance to hub
            dist = ((hub.x - x)**2 + (hub.y - y)**2)**0.5
            
            # Closer to hub = better
            proximity = max(0, 1 - (dist / (hub.radius * 2)))
            
            # Hub type preferences based on agent demographics
            type_preference = 0.0
            
            if hub.type == 'work' and 18 <= agent.age <= 65:
                # Working-age agents prefer to be near work
                type_preference = 0.8
            elif hub.type == 'education' and agent.age < 30:
                # Young agents prefer to be near education
                type_preference = 0.9 if agent.age < 18 else 0.5
            elif hub.type == 'leisure':
                # Everyone likes leisure, but preferences vary
                type_preference = 0.5
            elif hub.type == 'shopping':
                # Shopping importance scales with income
                type_preference = 0.3 + (agent.income / 100) * 0.4
            elif hub.type == 'transit':
                # Transit more important for working-age
                type_preference = 0.7 if 18 <= agent.age <= 65 else 0.4
            
            # Calculate hub match (how well the hub matches agent preferences)
            # Agents prefer hubs with more of their own type
            type_match = 0.0
            if hub.agent_types:
                agent_type_count = hub.agent_types.get(agent.type, 0)
                agent_type_percent = agent_type_count / len(hub.agents) if hub.agents else 0
                type_match = agent_type_percent
            
            # Combine factors
            hub_influence = (
                proximity * 0.4 +             # Being close is good
                type_preference * 0.3 +       # Hub matches agent needs
                type_match * 0.3              # Hub has similar agents
            ) * hub.importance                # Important hubs matter more
            
            influence += hub_influence
        
        # Normalize based on number of hubs
        if agent_hubs:
            influence /= len(agent_hubs)
            
        return influence * 0.3  # Scale the overall influence
    
    def update_hub_metrics(self):
        """Update metrics related to hubs for data collection."""
        if not hasattr(self, 'hubs') or not self.hubs:
            return
            
        # Reassign agents to hubs occasionally (not every step)
        if self.schedule.steps % 5 == 0:
            self.assign_agents_to_hubs()
        
        # Calculate overall hub segregation metrics
        hub_segregation = np.mean([hub.segregation_index for hub in self.hubs]) if self.hubs else 0
        hub_income_segregation = np.mean([hub.income_segregation for hub in self.hubs]) if self.hubs else 0
        
        # Store for data collection
        self.hub_segregation_data.append(hub_segregation)
        self.hub_income_segregation_data.append(hub_income_segregation)
    
    def step(self):
        """Run one step of the model."""
        # First, all agents decide whether they are happy or unhappy
        self.schedule.step()
        
        # Get list of unhappy agents
        unhappy_agents = [agent for agent in self.schedule.agents if not agent.is_happy]
        self.unhappy_agents = unhappy_agents
        
        # Move unhappy agents
        for agent in unhappy_agents:
            agent.move_to_new_location()
            self.total_moves += 1
        
        # Update hub metrics if hubs are enabled
        if self.use_hubs:
            self.update_hub_metrics()
        
        # Collect data
        self.datacollector.collect(self)
        
        # Are all agents happy?
        if not unhappy_agents:
            print(f"All agents happy after {self.schedule.steps} steps!")
    
    def get_happiness_ratio(self):
        """Calculate the proportion of happy agents."""
        happy_count = sum(1 for agent in self.schedule.agents if agent.is_happy)
        total_count = len(self.schedule.agents)
        return happy_count / total_count if total_count > 0 else 0
    
    def calculate_segregation_index(self):
        """Calculate a segregation index based on agent neighborhoods."""
        # Get all agents and their neighbors
        agent_neighborhoods = [
            (agent, agent.get_nearby_agents()) 
            for agent in self.schedule.agents
        ]
        
        # Calculate segregation for each neighborhood
        neighborhood_segregation = []
        for agent, neighbors in agent_neighborhoods:
            if not neighbors:
                continue
                
            # Count agents of each type in the neighborhood
            type_counts = defaultdict(int)
            for neighbor in neighbors:
                type_counts[neighbor.type] += 1
                
            # Find the most common type
            max_type_count = max(type_counts.values()) if type_counts else 0
            total_neighbors = len(neighbors)
            
            # Calculate segregation (percentage of neighbors of the most common type)
            segregation = max_type_count / total_neighbors if total_neighbors > 0 else 0
            
            # Adjust for random expectation (perfect integration would be 1/num_types for each type)
            random_expectation = 1 / self.num_agent_types
            adjusted_segregation = (segregation - random_expectation) / (1 - random_expectation)
            adjusted_segregation = max(0, min(1, adjusted_segregation))
            
            neighborhood_segregation.append(adjusted_segregation)
        
        # Return average segregation across all neighborhoods
        return np.mean(neighborhood_segregation) if neighborhood_segregation else 0
    
    def calculate_income_segregation_index(self):
        """Calculate income segregation based on income variance within neighborhoods."""
        if not self.income_segregation:
            return 0.0
            
        # Get all agents and their neighbors
        agent_neighborhoods = [
            (agent, agent.get_nearby_agents()) 
            for agent in self.schedule.agents
        ]
        
        # Calculate overall income standard deviation
        all_incomes = [agent.income for agent in self.schedule.agents]
        global_std = np.std(all_incomes) if all_incomes else 1.0
        
        # Calculate income segregation for each neighborhood
        neighborhood_income_segregation = []
        for agent, neighbors in agent_neighborhoods:
            if not neighbors or len(neighbors) < 2:
                continue
                
            # Get incomes of neighbors
            neighbor_incomes = [neighbor.income for neighbor in neighbors]
            
            # Calculate local income standard deviation
            local_std = np.std(neighbor_incomes)
            
            # Calculate income segregation (smaller local std = more segregation)
            # Normalize to 0-1 range
            income_segregation = 1 - (local_std / global_std) if global_std > 0 else 0
            income_segregation = max(0, min(1, income_segregation))
            
            neighborhood_income_segregation.append(income_segregation)
        
        # Return average income segregation across all neighborhoods
        return np.mean(neighborhood_income_segregation) if neighborhood_income_segregation else 0
