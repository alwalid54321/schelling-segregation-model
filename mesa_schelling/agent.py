"""
Agent module for Mesa-based Advanced Schelling model
"""

import random
import numpy as np
from mesa import Agent


class SchellingAgent(Agent):
    """
    Advanced Schelling segregation agent with complex characteristics:
    - Income, education, and age affect behavior
    - Dynamic tolerance based on attributes and experiences
    - Location attachment that increases over time
    - Social network connections
    """
    
    def __init__(self, pos, model, agent_type=None, income=None, education=None, 
                 age=None, tolerance=None):
        """
        Create a new Schelling agent.
        
        Args:
            pos: The agent's coordinates on the grid
            model: The model instance
            agent_type: The agent's type (determines its group membership)
            income: Agent's income level (0-100)
            education: Years of education (0-20)
            age: Agent's age (0-100)
            tolerance: Base tolerance for neighbors of different types
        """
        super().__init__(model.next_id(), model)
        self.pos = pos
        self.type = agent_type if agent_type is not None else self.random.randint(0, model.num_agent_types - 1)
        self.is_happy = False
        self.moves = 0
        self.happiness_history = []
        self.location_history = [pos]
        
        # Advanced characteristics
        self.income = income if income is not None else self.random.normalvariate(50, 20)
        self.education = education if education is not None else self.random.normalvariate(12, 4)
        self.age = age if age is not None else self.random.normalvariate(35, 15)
        self.base_tolerance = tolerance if tolerance is not None else self.random.uniform(0.2, 0.6)
        
        # Normalize values to reasonable ranges
        self.income = max(0, min(100, self.income))
        self.education = max(0, min(25, self.education))
        self.age = max(0, min(100, self.age))
        
        # Social network and location attachment
        self.connections = set()
        self.location_attachment = 0
    
    @property
    def tolerance(self):
        """
        Dynamic tolerance based on agent characteristics and experiences.
        Higher education increases tolerance, older age slightly decreases it.
        Recent experiences affect tolerance.
        """
        # Education effect (higher education = more tolerance)
        education_factor = 0.01 * (self.education - 12)  # +/- 1% per year from avg
        
        # Age effect (older = slightly less tolerance)
        age_factor = -0.001 * max(0, self.age - 35)  # -0.1% per year above avg
        
        # Recent happiness history effect
        if len(self.happiness_history) > 5:
            recent_experiences = self.happiness_history[-5:]
            experience_factor = 0.02 * (sum(recent_experiences) / len(recent_experiences) - 0.5)
        else:
            experience_factor = 0
        
        # Calculate modified tolerance with bounds
        modified_tolerance = self.base_tolerance - education_factor - age_factor - experience_factor
        return max(0.1, min(0.9, modified_tolerance))
    
    @property
    def location_preference(self):
        """
        Agent's preference for premium/central locations based on income.
        Higher income agents prefer premium locations.
        """
        return self.income / 100  # Normalize to 0-1 range
    
    def connect_with(self, other_agent):
        """Add another agent to this agent's social network."""
        self.connections.add(other_agent.unique_id)
        
    def increase_attachment(self):
        """Increase attachment to current location."""
        self.location_attachment = min(1.0, self.location_attachment + 0.1)
        
    def decrease_attachment(self):
        """Reset attachment when moving."""
        self.location_attachment = 0
    
    def get_nearby_agents(self):
        """Get agents in the neighborhood."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=1
        )
        return neighbors
    
    def calculate_happiness(self):
        """
        Calculate agent's happiness based on:
        - Similarity ratio to neighbors
        - Social network connections in neighborhood
        - Location quality
        - Attachment to current location
        """
        neighbors = self.get_nearby_agents()
        
        if not neighbors:
            self.is_happy = True
            happiness_score = 1.0
            self.happiness_history.append(happiness_score)
            return happiness_score
        
        # Count similar neighbors
        similar = sum(1 for n in neighbors if n.type == self.type)
        total = len(neighbors)
        
        # Basic similarity ratio
        similarity_ratio = similar / total if total > 0 else 0
        
        # ENHANCED: Calculate a similarity preference score that rewards having MORE similar neighbors
        # This creates stronger clustering as agents prefer maximum similarity
        similarity_preference = similarity_ratio ** 1.5  # Exponential reward for higher similarity
        
        # Adjustment for connections in neighborhood
        connected_neighbors = sum(1 for n in neighbors if n.unique_id in self.connections)
        connection_bonus = 0.05 * (connected_neighbors / total if total > 0 else 0)
        
        # Location quality adjustment
        location_quality = self.model.get_location_quality(self.pos)
        location_match = 1 - abs(self.location_preference - location_quality)
        location_factor = 0.1 * location_match
        
        # Hub influence
        hub_influence = 0.0
        if hasattr(self.model, 'get_hub_influence'):
            hub_influence = self.model.get_hub_influence(self, self.pos[0], self.pos[1])
        
        # Attachment to current location
        attachment_factor = 0.1 * self.location_attachment
        
        # Calculate final happiness score
        happiness_score = (similarity_preference * 0.6 +  # Enhanced similarity preference
                          connection_bonus * 0.1 +       # Bonus for knowing neighbors
                          location_factor * 0.1 +        # Location preference
                          attachment_factor * 0.1 +      # Attachment to current location
                          hub_influence * 0.1)           # Hub influence
        
        # Record happiness history
        self.happiness_history.append(happiness_score)
        
        # An agent is happy if the similarity ratio exceeds their personal
        # tolerance threshold OR if they are generally very happy
        meets_threshold = similarity_ratio >= self.tolerance
        very_happy = happiness_score >= 0.8
        self.is_happy = meets_threshold or very_happy
        
        return happiness_score
    
    def evaluate_potential_location(self, pos):
        """Evaluate how happy the agent would be at a potential new location."""
        # Get potential neighbors
        potential_neighbors = self.model.grid.get_neighbors(
            pos, moore=True, include_center=False, radius=1
        )
        
        if not potential_neighbors:
            return 0.7  # No neighbors is decent but not ideal
        
        # Count similar neighbors
        similar = sum(1 for n in potential_neighbors if n.type == self.type)
        total = len(potential_neighbors)
        
        # Basic similarity ratio
        similarity_ratio = similar / total if total > 0 else 0
        
        # ENHANCED: Calculate a similarity preference score with exponential scaling
        similarity_preference = similarity_ratio ** 1.5  # Reward higher concentrations
        
        # Adjustment for connections in neighborhood
        connected_neighbors = sum(1 for n in potential_neighbors if n.unique_id in self.connections)
        connection_bonus = 0.05 * (connected_neighbors / total if total > 0 else 0)
        
        # Location quality adjustment
        location_quality = self.model.get_location_quality(pos)
        location_match = 1 - abs(self.location_preference - location_quality)
        location_factor = 0.1 * location_match
        
        # Add a bonus for neighborhoods with very high similarity (80%+ same type)
        high_similarity_bonus = 0.2 if similarity_ratio > 0.8 else 0.0
        
        # Hub influence
        hub_influence = 0.0
        if hasattr(self.model, 'get_hub_influence'):
            hub_influence = self.model.get_hub_influence(self, pos[0], pos[1])
        
        # Calculate potential happiness score
        potential_score = (similarity_preference * 0.6 +  # Enhanced similarity preference
                          high_similarity_bonus * 0.1 +  # Bonus for highly similar neighborhoods
                          connection_bonus * 0.1 +       # Bonus for knowing neighbors
                          location_factor * 0.1 +        # Location preference
                          hub_influence * 0.1)           # Hub influence
        
        return potential_score
        
    def step(self):
        """
        Decide whether to move. If unhappy, the agent will look for a better location.
        """
        self.calculate_happiness()
        
    def move_to_new_location(self):
        """
        Find and move to a better location if unhappy.
        Uses a smart relocation strategy to find locations that maximize happiness.
        """
        if self.is_happy:
            self.increase_attachment()
            return False
        
        # Get all empty locations
        empty_cells = [(x, y) for x in range(self.model.grid.width) 
                         for y in range(self.model.grid.height)
                         if self.model.is_cell_empty((x, y)) and 
                         not self.model.is_cell_barrier((x, y))]
        
        if not empty_cells:
            self.is_happy = True  # No empty cells, so stay put
            return False
            
        # Evaluate a sample of potential locations
        sample_size = min(20, len(empty_cells))  # Limit sample size for performance
        candidate_locations = self.random.sample(empty_cells, sample_size)
        
        # Score each location
        scored_locations = []
        current_score = self.calculate_happiness()
        
        for pos in candidate_locations:
            score = self.evaluate_potential_location(pos)
            # Only consider locations that would make the agent happier
            if score > current_score:
                scored_locations.append((pos, score))
        
        # Pick the best location, or randomly if none are better
        if scored_locations:
            scored_locations.sort(key=lambda x: x[1], reverse=True)
            new_pos = scored_locations[0][0]
        else:
            # If no better location, just pick randomly from all empty cells
            new_pos = self.random.choice(empty_cells)
        
        # Move to the new location
        self.model.grid.move_agent(self, new_pos)
        self.location_history.append(new_pos)
        self.moves += 1
        self.decrease_attachment()
        
        return True
