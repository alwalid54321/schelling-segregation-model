# server.py
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from model import SchellingModel
from agent import SchellingAgent


def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle", 
        "Filled": "true", 
        "Layer": 0, 
        "r": 0.8 if agent.is_happy else 0.4
    }
    
    # Use different colors for different agent types
    colors = ["#FF4500", "#4169E1", "#32CD32", "#FFD700", "#9932CC"]
    if agent.agent_type < len(colors):
        portrayal["Color"] = colors[agent.agent_type]
    else:
        portrayal["Color"] = "#000000"  # Default black for any other types
        
    # Add a symbol for unhappy agents
    if not agent.is_happy:
        portrayal["text"] = "✗"
        portrayal["text_color"] = "white"
    
    return portrayal

# Grid visualization
grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)

# Happiness chart
happiness_chart = ChartModule([
    {"Label": "Happiness", "Color": "#00FF00"}
])

# Segregation chart
segregation_chart = ChartModule([
    {"Label": "Segregation", "Color": "#FF0000"}
])

server = ModularServer(
    SchellingModel,
    [grid, happiness_chart, segregation_chart],
    "Schelling Segregation Model with Agent Swapping",
    {
        "width": 20,
        "height": 20,
        "density": UserSettableParameter("slider", "Agent Density", 1.0, 0.5, 1.0, 0.05),
        "homophily": UserSettableParameter("slider", "Homophily Threshold", 0.3, 0.0, 1.0, 0.05),
        "num_agent_types": UserSettableParameter("slider", "Agent Types", 2, 2, 5, 1)
    }
)
server.port = 8521

if __name__ == '__main__':
    server.launch()