# agent.py
from mesa import Agent

class SchellingAgent(Agent):
    def __init__(self, unique_id, model, agent_type):
        # In Mesa 3.1.5, Agent.__init__ expects model as the first parameter
        # and it generates a unique_id automatically
        super().__init__(model)
        # Override the auto-generated unique_id with our provided one
        self.unique_id = unique_id
        self.agent_type = agent_type
        self.is_happy = True
        self.moves_made = 0

    def step(self):
        similar = 0
        total = 0
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        for neighbor in neighbors:
            if isinstance(neighbor, SchellingAgent):
                total += 1
                if neighbor.agent_type == self.agent_type:
                    similar += 1

        # Determine happiness
        if total > 0 and (similar / total) < self.model.homophily:
            self.is_happy = False
            if self not in self.model.unhappy_agents:
                self.model.unhappy_agents.append(self)
        else:
            self.is_happy = True
    
    def try_to_swap(self):
        """Try to swap positions with another agent to improve happiness"""
        # Get all possible agents to swap with
        possible_swaps = [agent for agent in self.model.agents 
                         if agent.unique_id != self.unique_id]
        
        # Shuffle the list of possible agents to swap with
        self.model.random.shuffle(possible_swaps)
        
        current_happiness = self.calculate_happiness_at(self.pos)
        
        # Try to find a swap that improves happiness
        for other_agent in possible_swaps:
            # Calculate current happiness of both agents
            other_current_happiness = other_agent.calculate_happiness_at(other_agent.pos)
            
            # Calculate happiness after potential swap
            my_new_happiness = self.calculate_happiness_at(other_agent.pos)
            other_new_happiness = other_agent.calculate_happiness_at(self.pos)
            
            # Only swap if it improves happiness for at least one agent
            # without making the other agent unhappier
            if ((my_new_happiness > current_happiness and other_new_happiness >= other_current_happiness) or
                (other_new_happiness > other_current_happiness and my_new_happiness >= current_happiness)):
                # Perform the swap
                self.model.grid.swap_positions(self, other_agent)
                self.moves_made += 1
                other_agent.moves_made += 1
                self.model.total_moves += 1
                return True
        
        return False
    
    def calculate_happiness_at(self, pos):
        """Calculate how happy this agent would be at the given position"""
        similar = 0
        total = 0
        
        # Get neighbors at the position
        neighbors = self.model.grid.get_neighbors(
            pos, moore=True, include_center=False
        )
        
        # Count similar neighbors
        for neighbor in neighbors:
            if isinstance(neighbor, SchellingAgent):
                total += 1
                if neighbor.agent_type == self.agent_type:
                    similar += 1
        
        # Calculate happiness score
        happiness_score = similar / total if total > 0 else 0
        return happiness_score