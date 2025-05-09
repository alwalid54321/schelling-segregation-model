"""
Agent classes for the advanced Schelling Segregation Model.
"""

import numpy as np
import random


class Agent:
    """Base agent class with common properties and behaviors."""
    
    def __init__(self, agent_id, agent_type, x, y, income=None, education=None, age=None, tolerance=None):
        self.id = agent_id
        self.type = agent_type
        self.x = x
        self.y = y
        self.is_happy = False
        self.moves = 0
        self.happiness_history = []
        
        # Advanced characteristics that influence behavior
        self.income = income if income is not None else random.normalvariate(50, 20)
        self.education = education if education is not None else random.normalvariate(12, 4)
        self.age = age if age is not None else random.normalvariate(35, 15)
        
        # Tolerance is personal homophily threshold - can vary by agent
        # Higher education could lead to more tolerance (lower threshold)
        self.base_tolerance = tolerance if tolerance is not None else random.uniform(0.2, 0.6)
        
        # Memory of previous locations
        self.location_history = [(x, y)]
        
        # Social network - agents this agent has connections with
        self.connections = set()
        
        # Attachment to current location (increases with time spent in location)
        self.location_attachment = 0
    
    @property
    def tolerance(self):
        """
        Agent's tolerance for living near different types.
        This can change dynamically based on experiences.
        """
        # Higher education increases tolerance (lowers homophily threshold)
        education_factor = 0.01 * (self.education - 12)  # +/- 1% per year from avg
        
        # Age effect - older people might be slightly less tolerant
        age_factor = -0.001 * max(0, self.age - 35)  # -0.1% per year above avg
        
        # Recent happiness history affects tolerance
        if len(self.happiness_history) > 5:
            recent_experiences = self.happiness_history[-5:]
            experience_factor = 0.02 * (sum(recent_experiences) / len(recent_experiences) - 0.5)
        else:
            experience_factor = 0
        
        # Calculate modified tolerance, ensuring it stays in valid range
        modified_tolerance = self.base_tolerance - education_factor - age_factor - experience_factor
        return max(0.1, min(0.9, modified_tolerance))
    
    @property
    def location_preference(self):
        """
        Preference for locations based on income (affluent areas).
        Higher values mean preference for central/premium locations.
        """
        return self.income / 100  # Normalize to 0-1 range

    def connect_with(self, other_agent):
        """Add another agent to this agent's social network."""
        self.connections.add(other_agent.id)
        
    def increase_attachment(self):
        """Increase attachment to current location."""
        self.location_attachment = min(1.0, self.location_attachment + 0.1)
        
    def decrease_attachment(self):
        """Reset attachment when moving."""
        self.location_attachment = 0
        
    def add_location_to_history(self, x, y):
        """Record a new location in the agent's history."""
        self.location_history.append((x, y))
        self.x = x
        self.y = y
        self.moves += 1
        
    def calculate_happiness(self, neighbors, global_homophily):
        """
        Calculate happiness based on neighbors and personal preferences.
        Returns a happiness score and whether the agent is happy.
        """
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
        # This creates stronger clustering as agents prefer maximum similarity, not just meeting a threshold
        similarity_preference = similarity_ratio ** 1.5  # Exponential reward for higher similarity
        
        # Adjustment for connections in neighborhood
        connected_neighbors = sum(1 for n in neighbors if n.id in self.connections)
        connection_bonus = 0.05 * (connected_neighbors / total if total > 0 else 0)
        
        # Adjustment for location preference
        x_center_dist = abs(self.x - 25) / 25  # Distance from center X (normalized)
        y_center_dist = abs(self.y - 25) / 25  # Distance from center Y (normalized)
        center_dist = (x_center_dist**2 + y_center_dist**2)**0.5 / 1.414  # Normalized
        
        location_match = 1 - abs(self.location_preference - (1 - center_dist))
        location_factor = 0.1 * location_match
        
        # Attachment to current location reduces likelihood of moving
        attachment_factor = 0.1 * self.location_attachment
        
        # Calculate final happiness score
        happiness_score = (similarity_preference * 0.7 +  # Enhanced similarity preference
                          connection_bonus +            # Bonus for knowing neighbors
                          location_factor +             # Location preference
                          attachment_factor)            # Attachment to current location
        
        # Record happiness history for learning
        self.happiness_history.append(happiness_score)
        
        # An agent is happy if meeting the minimum threshold OR having high overall happiness
        # But now will be happier with MORE similar neighbors beyond just the threshold
        meets_minimum = similarity_ratio >= self.tolerance
        has_high_happiness = happiness_score >= 0.8
        self.is_happy = meets_minimum or has_high_happiness
        
        return happiness_score
    
    def evaluate_potential_location(self, x, y, potential_neighbors):
        """
        Evaluate how happy the agent would be at a potential new location.
        Returns an estimated happiness score.
        """
        if not potential_neighbors:
            return 0.8  # No neighbors is okay but not ideal
        
        # Count similar neighbors
        similar = sum(1 for n in potential_neighbors if n.type == self.type)
        total = len(potential_neighbors)
        
        # Basic similarity ratio
        similarity_ratio = similar / total if total > 0 else 0
        
        # ENHANCED: Calculate a similarity preference score that rewards having MORE similar neighbors
        # This creates stronger clustering as agents seek maximum similarity
        similarity_preference = similarity_ratio ** 1.5  # Exponential reward for higher similarity
        
        # Adjustment for connections in neighborhood
        connected_neighbors = sum(1 for n in potential_neighbors if n.id in self.connections)
        connection_bonus = 0.05 * (connected_neighbors / total if total > 0 else 0)
        
        # Adjustment for location preference
        x_center_dist = abs(x - 25) / 25  # Distance from center X (normalized)
        y_center_dist = abs(y - 25) / 25  # Distance from center Y (normalized)
        center_dist = (x_center_dist**2 + y_center_dist**2)**0.5 / 1.414  # Normalized
        
        location_match = 1 - abs(self.location_preference - (1 - center_dist))
        location_factor = 0.1 * location_match
        
        # Add a bonus for neighborhoods with very high similarity (80%+ same type)
        high_similarity_bonus = 0.2 if similarity_ratio > 0.8 else 0.0
        
        # Calculate potential happiness score with enhanced preference for similar agents
        potential_score = (similarity_preference * 0.7 +  # Enhanced similarity is most important
                           high_similarity_bonus +        # Bonus for highly similar neighborhoods
                           connection_bonus +             # Bonus for knowing neighbors
                           location_factor)               # Location preference
        
        return potential_score
