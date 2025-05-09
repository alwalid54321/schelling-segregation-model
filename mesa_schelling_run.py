"""
Run script for Mesa-based Advanced Schelling Segregation Model
"""
import mesa
import numpy as np
from mesa_schelling.model import SchellingModel
from mesa_schelling.visualization import agent_portrayal

# For visualization
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

# Create a text element to display model info
class InfoTextElement(mesa.visualization.TextElement):
    def render(self, model):
        happy_agents = sum(1 for agent in model.schedule.agents if getattr(agent, 'is_happy', False))
        total_agents = len(model.schedule.agents)
        segregation = model.calculate_segregation_index() if hasattr(model, 'calculate_segregation_index') else 0
        
        text = f"Happy agents: {happy_agents}/{total_agents} ({happy_agents/total_agents:.1%})<br>"
        text += f"Segregation index: {segregation:.2f}"
        return text

def run_mesa_model():
    """Set up and run the Mesa model server"""
    # Create grid visualization with appropriate cell size to see agents clearly
    grid_size = 50
    canvas_size = 500
    grid = CanvasGrid(agent_portrayal, grid_size, grid_size, canvas_size, canvas_size)
    
    # Create info text element
    info = InfoTextElement()
    
    # Create chart for tracking happiness and segregation
    chart = mesa.visualization.ChartModule(
        [
            {"Label": "happy_agents", "Color": "Green"},
            {"Label": "segregation", "Color": "Red"}
        ],
        data_collector_name="datacollector"
    )
    
    # Model parameters
    model_params = {
        "width": grid_size,
        "height": grid_size,
        "density": UserSettableParameter("slider", "Population Density", 1.0, 0.1, 1.0, 0.1),
        "homophily": UserSettableParameter("slider", "Homophily", 0.3, 0.0, 1.0, 0.1),
        "num_agent_types": UserSettableParameter("slider", "Agent Types", 2, 2, 5, 1),
        "pattern_type": UserSettableParameter(
            "choice", "Initial Pattern", "alternating",
            choices=["random", "alternating", "checkerboard", "clusters", "income_stratified"]
        )
    }
    
    # Create and launch the server
    server = ModularServer(
        SchellingModel,
        [grid, info, chart],
        "Mesa Schelling Model",
        model_params
    )
    
    # Start the server
    server.port = 8521  # Choose another port if this one is taken
    server.launch()

if __name__ == "__main__":
    run_mesa_model()
