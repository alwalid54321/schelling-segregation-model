# model.py
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import SchellingAgent


def compute_segregation(model):
    """
    Calculate average fraction of same-type neighbors across all agents.
    """
    total_frac = 0
    count = 0
    for agent in model.agents:
        neighbors = model.grid.get_neighbors(
            agent.pos, moore=True, include_center=False
        )
        same = sum(
            1 for n in neighbors
            if isinstance(n, SchellingAgent) and n.agent_type == agent.agent_type
        )
        occupied = sum(
            1 for n in neighbors if isinstance(n, SchellingAgent)
        )
        if occupied > 0:
            total_frac += same / occupied
            count += 1
    return total_frac / count if count else 0


def compute_happiness(model):
    """
    Calculate fraction of happy agents.
    """
    total = len(model.agents)
    unhappy = len(model.unhappy_agents)
    return (total - unhappy) / total if total else 0


class SchellingModel(Model):
    """
    Schelling segregation model for Mesa 3.x with full grid and agent swapping.
    """

    def __init__(
        self, width=20, height=20,
        density=1.0, homophily=0.3,
        neighbor_radius=1, num_agent_types=2,
        seed=None
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.density = density  # Set to 1.0 for full grid
        self.homophily = homophily
        self.num_agent_types = num_agent_types
        self.steps = 0
        self.max_steps = 100
        self.unhappy_agents = []
        self.total_moves = 0

        # Set up grid and populate
        self.grid = MultiGrid(width, height, torus=True)
        self._populate_grid()

        # Add swap_positions method to the grid
        def swap_positions(grid, agent1, agent2):
            """Swap the positions of two agents"""
            pos1, pos2 = agent1.pos, agent2.pos
            
            # Remove agents from their current positions
            grid.remove_agent(agent1)
            grid.remove_agent(agent2)
            
            # Place agents in new positions
            grid.place_agent(agent1, pos2)
            grid.place_agent(agent2, pos1)
            
            # Agent positions are updated automatically by place_agent
        
        # Add the swap method to the grid
        self.grid.swap_positions = lambda a1, a2: swap_positions(self.grid, a1, a2)

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Segregation": compute_segregation,
                "Happiness": compute_happiness,
                "TotalMoves": lambda m: m.total_moves,
                "Step": lambda m: m.steps
            },
            agent_reporters={
                "IsHappy": lambda a: a.is_happy,
                "AgentType": lambda a: a.agent_type,
                "MovesMade": lambda a: a.moves_made
            }
        )
        self.datacollector.collect(self)

    def _populate_grid(self):
        agent_id = 0
        for x in range(self.width):
            for y in range(self.height):
                if self.random.random() < self.density:
                    agent_type = self.random.randint(0, self.num_agent_types - 1)
                    agent = SchellingAgent(agent_id, self, agent_type)
                    self.grid.place_agent(agent, (x, y))
                    agent_id += 1

    def step(self):
        self.steps += 1
        self.unhappy_agents = []

        # Activate all agents to determine happiness
        for agent in self.agents:
            agent.step()

        # Collect metrics
        self.datacollector.collect(self)

        # Make unhappy agents try to swap with others
        if self.unhappy_agents:
            self.random.shuffle(self.unhappy_agents)
            for agent in self.unhappy_agents:
                if agent.is_happy:  # Skip if agent became happy from another swap
                    continue
                
                # Try to swap, which internally handles the happiness calculations
                agent.try_to_swap()

    def run_model(self, steps=None):
        steps = steps or self.max_steps
        for _ in range(steps):
            self.step()
            if not self.unhappy_agents:
                break