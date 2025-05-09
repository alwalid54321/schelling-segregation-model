# agent.py
from mesa import Agent

class SchellingAgent(Agent):
    def __init__(self, unique_id, model, agent_type):
        # In Mesa 3.x, Agent.__init__ only takes self
        Agent.__init__(self)
        self.unique_id = unique_id
        self.model = model
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
            self.model.unhappy_agents.append(self)
        else:
            self.is_happy = True
