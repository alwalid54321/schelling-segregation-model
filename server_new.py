# server.py
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from model import SchellingModel
from agent import SchellingAgent


def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "Layer": 0, "r": 0.5}
    portrayal["Color"] = "red" if agent.agent_type == 0 else "blue"
    return portrayal

# Grid visualization
grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)

# Happiness chart
happiness_chart = ChartModule([
    {"Label": "Happiness", "Color": "#0000FF"}
])

server = ModularServer(
    SchellingModel,
    [grid, happiness_chart],
    "Schelling Segregation Model",
    {
        "width": 20,
        "height": 20,
        "density": UserSettableParameter("slider", "Agent Density", 0.8, 0.1, 1.0, 0.05),
        "homophily": UserSettableParameter("slider", "Homophily Threshold", 0.3, 0.0, 1.0, 0.05),
    }
)
server.port = 8521

if __name__ == '__main__':
    server.launch()
