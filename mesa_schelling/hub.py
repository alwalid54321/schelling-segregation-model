"""
Hub module for Mesa-based Advanced Schelling model.
Implements activity hubs where agents interact (work, education, leisure, etc.)
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
    
    def __init__(self, hub_id, hub_type, x, y, radius, capacity, importance, model):
        """
        Initialize an activity hub.
        
        Args:
            hub_id: Unique identifier for the hub
            hub_type: Type of hub (work, education, leisure, shopping, transit)
            x, y: Coordinates of the hub center
            radius: Influence radius of the hub
            capacity: Maximum number of agents that can interact in the hub
            importance: Hub importance (affects how much it influences agent decisions)
            model: Reference to the model
        """
        self.id = hub_id
        self.type = hub_type
        self.x = x
        self.y = y
        self.radius = radius
        self.capacity = capacity
        self.importance = importance
        self.model = model
        
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
        self.agents.add(agent.unique_id)
        self.agent_types[agent.type] += 1
        self.agent_incomes.append(agent.income)
        
    def remove_agent(self, agent):
        """Remove an agent from the hub."""
        if agent.unique_id in self.agents:
            self.agents.remove(agent.unique_id)
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
    
    def is_full(self):
        """Check if the hub is at capacity."""
        return len(self.agents) >= self.capacity
    
    def dominant_type(self):
        """Get the dominant agent type in the hub."""
        if not self.agents:
            return None
        return max(self.agent_types.items(), key=lambda x: x[1])[0]
    
    def diversity_score(self):
        """Calculate hub diversity score (higher = more diverse)."""
        if not self.agents:
            return 0.0
            
        total = len(self.agents)
        # Calculate Shannon entropy
        entropy = 0.0
        for count in self.agent_types.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p)
                
        # Normalize entropy to 0-1
        max_entropy = np.log(len(self.agent_types)) if self.agent_types else 1.0
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return diversity
