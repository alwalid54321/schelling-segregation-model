"""
Hub-based features for the Advanced Schelling Segregation Model.

This module implements urban activity hubs that influence agent movement and
segregation patterns, based on recent research showing that segregation emerges
not just in residential areas but in various domains of daily life.
"""

import random
import numpy as np
from collections import defaultdict


class ActivityHub:
    """
    Represents an urban activity hub where agents interact.
    
    Examples include:
    - Work hubs (offices, industrial areas)
    - Education hubs (schools, universities)
    - Leisure hubs (parks, entertainment venues)
    - Shopping hubs (malls, markets)
    - Transit hubs (stations, airports)
    """
    
    def __init__(self, hub_id, hub_type, x, y, radius, capacity, importance):
        """
        Initialize an activity hub.
        
        Args:
            hub_id: Unique identifier for the hub
            hub_type: Type of hub (work, education, leisure, shopping, transit)
            x, y: Coordinates of the hub center
            radius: Influence radius of the hub
            capacity: Maximum number of agents that can interact in the hub
            importance: Hub importance (affects how much it influences agent decisions)
        """
        self.id = hub_id
        self.type = hub_type
        self.x = x
        self.y = y
        self.radius = radius
        self.capacity = capacity
        self.importance = importance
        
        # Tracks agents currently in the hub
        self.agents = set()
        
        # Tracks demographic makeup of the hub
        self.agent_types = defaultdict(int)
        self.agent_incomes = []
        
        # Hub segregation metrics
        self.segregation_index = 0.0
        self.income_segregation = 0.0
        
        # Tracks hub evolution over time
        self.history = {
            'agents': [],
            'segregation': [],
            'income_segregation': []
        }
    
    def is_in_range(self, x, y):
        """Check if a position is within the hub's influence radius."""
        return ((x - self.x)**2 + (y - self.y)**2) <= self.radius**2
    
    def add_agent(self, agent):
        """Add an agent to the hub."""
        self.agents.add(agent.id)
        self.agent_types[agent.type] += 1
        self.agent_incomes.append(agent.income)
        
    def remove_agent(self, agent):
        """Remove an agent from the hub."""
        if agent.id in self.agents:
            self.agents.remove(agent.id)
            self.agent_types[agent.type] -= 1
            # Remove income (this is approximate since we don't track which income belongs to which agent)
            if self.agent_incomes:
                self.agent_incomes.pop()
    
    def calculate_hub_segregation(self, num_agent_types):
        """Calculate segregation metrics for the hub."""
        if len(self.agents) <= 1:
            self.segregation_index = 0.0
            self.income_segregation = 0.0
            return
        
        # Type segregation - how dominated is the hub by one type?
        total_agents = len(self.agents)
        max_type_count = max(self.agent_types.values())
        type_balance = max_type_count / total_agents if total_agents > 0 else 0
        
        # Adjusted for number of types - perfect balance would be 1/num_types
        perfect_balance = 1 / num_agent_types
        self.segregation_index = (type_balance - perfect_balance) / (1 - perfect_balance)
        self.segregation_index = max(0, min(1, self.segregation_index))
        
        # Income segregation - variance in incomes compared to overall population
        if self.agent_incomes:
            self.income_segregation = np.std(self.agent_incomes) / 50  # Normalize to 0-1 range
            self.income_segregation = min(1, self.income_segregation)
        else:
            self.income_segregation = 0
        
        # Record metrics in history
        self.history['agents'].append(len(self.agents))
        self.history['segregation'].append(self.segregation_index)
        self.history['income_segregation'].append(self.income_segregation)
    
    @property
    def is_full(self):
        """Check if the hub is at capacity."""
        return len(self.agents) >= self.capacity
    
    @property
    def dominant_type(self):
        """Get the dominant agent type in the hub."""
        if not self.agent_types:
            return None
        return max(self.agent_types.items(), key=lambda x: x[1])[0] if self.agent_types else None
    
    @property
    def diversity_score(self):
        """Calculate hub diversity score (higher = more diverse)."""
        if not self.agents:
            return 0
            
        total = len(self.agents)
        type_proportions = [count/total for count in self.agent_types.values()]
        
        # Shannon entropy as diversity measure
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in type_proportions)
        max_entropy = np.log(len(self.agent_types)) if self.agent_types else 0
        
        # Normalize to 0-1
        return entropy / max_entropy if max_entropy > 0 else 0


def add_hub_methods_to_model(model_class):
    """
    Add hub-related methods to the model class.
    
    Args:
        model_class: The model class to extend
    """
    
    def initialize_hubs(self):
        """Initialize activity hubs in the model."""
        self.hubs = []
        self.hub_types = ['work', 'education', 'leisure', 'shopping', 'transit']
        
        # Number of hubs of each type
        hub_counts = {
            'work': random.randint(1, 3),
            'education': random.randint(2, 4),
            'leisure': random.randint(2, 5),
            'shopping': random.randint(2, 4),
            'transit': random.randint(1, 3)
        }
        
        hub_id = 0
        
        # Create hubs of each type
        for hub_type, count in hub_counts.items():
            for _ in range(count):
                # Find a valid location for the hub (not on a barrier)
                while True:
                    x = random.randint(5, self.width - 5)
                    y = random.randint(5, self.height - 5)
                    if self.mask[x, y] and (x, y) not in self.barriers:
                        break
                
                # Create hub with varying properties based on type
                if hub_type == 'work':
                    radius = random.randint(5, 8)
                    capacity = random.randint(30, 50)
                    importance = random.uniform(0.7, 0.9)
                elif hub_type == 'education':
                    radius = random.randint(4, 7)
                    capacity = random.randint(20, 40)
                    importance = random.uniform(0.6, 0.8)
                elif hub_type == 'leisure':
                    radius = random.randint(3, 6)
                    capacity = random.randint(15, 30)
                    importance = random.uniform(0.4, 0.6)
                elif hub_type == 'shopping':
                    radius = random.randint(4, 6)
                    capacity = random.randint(20, 35)
                    importance = random.uniform(0.5, 0.7)
                else:  # transit
                    radius = random.randint(3, 5)
                    capacity = random.randint(25, 45)
                    importance = random.uniform(0.6, 0.7)
                
                hub = ActivityHub(
                    hub_id=hub_id,
                    hub_type=hub_type,
                    x=x, y=y,
                    radius=radius,
                    capacity=capacity,
                    importance=importance
                )
                hub_id += 1
                
                self.hubs.append(hub)
    
    def assign_agents_to_hubs(self):
        """Assign agents to appropriate hubs based on preferences and demographics."""
        # Clear existing hub assignments
        for hub in self.hubs:
            hub.agents.clear()
            hub.agent_types.clear()
            hub.agent_incomes.clear()
        
        # Assign agents to hubs based on their characteristics
        for agent_id, agent in self.agents.items():
            # Each agent is assigned to multiple hubs based on their demographics
            # Work hubs - based on age and education
            if 18 <= agent.age <= 65:
                work_hubs = [h for h in self.hubs if h.type == 'work' and not h.is_full]
                if work_hubs:
                    # Higher education = preference for larger work hubs
                    work_hubs.sort(key=lambda h: h.capacity * (agent.education / 20), reverse=True)
                    for hub in work_hubs[:1]:  # Assign to one work hub
                        hub.add_agent(agent)
            
            # Education hubs - based on age
            if agent.age < 30:
                edu_chance = 1.0 if agent.age < 18 else (30 - agent.age) / 12
                if random.random() < edu_chance:
                    edu_hubs = [h for h in self.hubs if h.type == 'education' and not h.is_full]
                    if edu_hubs:
                        # Sort based on agent age
                        if agent.age < 18:
                            # Younger agents prefer smaller education hubs (schools)
                            edu_hubs.sort(key=lambda h: h.capacity)
                        else:
                            # Older students prefer larger education hubs (universities)
                            edu_hubs.sort(key=lambda h: h.capacity, reverse=True)
                        
                        for hub in edu_hubs[:1]:  # Assign to one education hub
                            hub.add_agent(agent)
            
            # Leisure hubs - everyone goes to these with varying frequency
            leisure_chance = random.uniform(0.3, 0.8)  # Chance of using leisure hubs
            if random.random() < leisure_chance:
                leisure_hubs = [h for h in self.hubs if h.type == 'leisure' and not h.is_full]
                if leisure_hubs:
                    # Higher income = preference for more important leisure hubs
                    leisure_hubs.sort(key=lambda h: h.importance * (agent.income / 50), reverse=True)
                    for hub in leisure_hubs[:random.randint(1, 2)]:  # Assign to 1-2 leisure hubs
                        hub.add_agent(agent)
            
            # Shopping hubs - based on income primarily
            shop_hubs = [h for h in self.hubs if h.type == 'shopping' and not h.is_full]
            if shop_hubs:
                # Higher income agents prefer more important shopping hubs
                shop_hubs.sort(key=lambda h: h.importance * (agent.income / 50), reverse=True)
                for hub in shop_hubs[:random.randint(1, 2)]:  # Assign to 1-2 shopping hubs
                    hub.add_agent(agent)
            
            # Transit hubs - based on location and work status
            transit_chance = 0.7 if 18 <= agent.age <= 65 else 0.4
            if random.random() < transit_chance:
                transit_hubs = [h for h in self.hubs if h.type == 'transit' and not h.is_full]
                if transit_hubs:
                    # Prefer transit hubs closer to home
                    transit_hubs.sort(key=lambda h: ((h.x - agent.x)**2 + (h.y - agent.y)**2)**0.5)
                    for hub in transit_hubs[:1]:  # Assign to one transit hub
                        hub.add_agent(agent)
        
        # Calculate segregation metrics for all hubs
        for hub in self.hubs:
            hub.calculate_hub_segregation(self.num_agent_types)
    
    def get_hub_influence(self, agent, x, y):
        """
        Calculate how hubs influence an agent's happiness at a location.
        
        Args:
            agent: The agent
            x, y: The location to evaluate
            
        Returns:
            A happiness modifier based on hubs
        """
        # No influence if hub features are disabled
        if not self.use_hubs:
            return 0.0
        
        influence = 0.0
        agent_hubs = [hub for hub in self.hubs if agent.id in hub.agents]
        
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
        # Reassign agents to hubs occasionally (not every step)
        if self.steps % 5 == 0:
            self.assign_agents_to_hubs()
        
        # Calculate overall hub segregation metrics
        hub_segregation = np.mean([hub.segregation_index for hub in self.hubs]) if self.hubs else 0
        hub_income_segregation = np.mean([hub.income_segregation for hub in self.hubs]) if self.hubs else 0
        
        # Store for data collection
        self.hub_segregation_data.append(hub_segregation)
        self.hub_income_segregation_data.append(hub_income_segregation)

    # Add these methods to the model class
    model_class.initialize_hubs = initialize_hubs
    model_class.assign_agents_to_hubs = assign_agents_to_hubs
    model_class.get_hub_influence = get_hub_influence
    model_class.update_hub_metrics = update_hub_metrics
