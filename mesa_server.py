"""
Mesa 3.1.5 compatible server for the Schelling Segregation Model
"""
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from model import SchellingModel
from agent import SchellingAgent

# Define how to visualize the model
def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.8 if agent.is_happy else 0.4,
        "Layer": 0
    }
    
    # Colors for different agent types
    colors = ["#FF4500", "#4169E1", "#32CD32", "#FFD700", "#9932CC"]
    if agent.agent_type < len(colors):
        portrayal["Color"] = colors[agent.agent_type]
    else:
        portrayal["Color"] = "#000000"  # Default black for any other types
    
    # Different appearance for unhappy agents
    if not agent.is_happy:
        portrayal["r"] = 0.4
        portrayal["text"] = "✗"
        portrayal["text_color"] = "white"
    
    return portrayal

# Create visualization elements
canvas_element = CanvasGrid(
    agent_portrayal, 20, 20, 500, 500
)

# Charts for data collection
happiness_chart = ChartModule(
    [{"Label": "Happiness", "Color": "#00FF00"}]
)

segregation_chart = ChartModule(
    [{"Label": "Segregation", "Color": "#FF0000"}]
)

# Create the server
model_params = {
    "width": 20,
    "height": 20,
    "density": UserSettableParameter(
        "slider", "Agent Density", 1.0, 0.5, 1.0, 0.05
    ),
    "homophily": UserSettableParameter(
        "slider", "Homophily Threshold", 0.3, 0.0, 1.0, 0.05
    ),
    "num_agent_types": UserSettableParameter(
        "slider", "Agent Types", 2, 2, 5, 1
    )
}

server = ModularServer(
    SchellingModel,
    [canvas_element, happiness_chart, segregation_chart],
    "Schelling Segregation Model with Agent Swapping",
    model_params
)

server.port = 8521

if __name__ == "__main__":
    server.launch()
