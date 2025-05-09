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
    Schelling segregation model for Mesa 3.x.
    """

    def __init__(
        self, width=20, height=20,
        density=0.8, homophily=0.3,
        neighbor_radius=1, num_agent_types=2,
        seed=None
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.density = density
        self.homophily = homophily
        self.steps = 0
        self.max_steps = 100
        self.unhappy_agents = []
        self.total_moves = 0

        # Set up grid and populate
        self.grid = MultiGrid(width, height, torus=True)
        self._populate_grid()

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
                    agent_type = self.random.randint(0, 1)
                    agent = SchellingAgent(agent_id, self, agent_type)
                    self.grid.place_agent(agent, (x, y))
                    agent_id += 1

    def step(self):
        self.steps += 1
        self.unhappy_agents = []

        # Activate all agents
        for agent in self.agents:
            agent.step()

        # Collect metrics
        self.datacollector.collect(self)

        # Relocate unhappy agents
        if self.unhappy_agents:
            self.relocate_agents()

    def relocate_agents(self):
        empty_cells = [
            (x, y) for (contents, x, y) in self.grid.coord_iter() if not contents
        ]
        for agent in self.unhappy_agents:
            if not empty_cells:
                break
            new_pos = self.random.choice(empty_cells)
            old_pos = agent.pos
            self.grid.move_agent(agent, new_pos)
            agent.moves_made += 1
            self.total_moves += 1
            empty_cells.remove(new_pos)
            empty_cells.append(old_pos)

    def run_model(self, steps=None):
        steps = steps or self.max_steps
        for _ in range(steps):
            self.step()
            if not self.unhappy_agents:
                break
