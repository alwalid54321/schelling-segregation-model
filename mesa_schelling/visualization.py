"""
Visualization module for Mesa-based Schelling model
"""

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from .model import SchellingModel
from collections import defaultdict


class HappinessElement(TextElement):
    """Text element displaying the happiness ratio."""
    
    def render(self, model):
        happy_count = sum(1 for agent in model.schedule.agents if agent.is_happy)
        total_count = len(model.schedule.agents)
        ratio = happy_count / total_count if total_count > 0 else 0
        
        if total_count > 0:
            return f"Happy agents: {happy_count}/{total_count} ({ratio:.1%})"
        else:
            return "No agents in simulation"


class SegregationElement(TextElement):
    """Text element displaying segregation indices."""
    
    def render(self, model):
        segregation = model.calculate_segregation_index()
        income_segregation = model.calculate_income_segregation_index() if model.income_segregation else 0
        
        text = f"Type segregation: {segregation:.2f}"
        if model.income_segregation:
            text += f" | Income segregation: {income_segregation:.2f}"
            
        if hasattr(model, 'hub_segregation_data') and model.hub_segregation_data:
            hub_segregation = model.hub_segregation_data[-1]
            text += f" | Hub segregation: {hub_segregation:.2f}"
            
        return text


class HubElement(TextElement):
    """Text element displaying hub information."""
    
    def render(self, model):
        if not hasattr(model, 'hubs') or not model.hubs:
            return "Hubs: Disabled"
            
        hub_counts = defaultdict(int)
        for hub in model.hubs:
            hub_counts[hub.type] += 1
            
        text = "Hubs: "
        text += ", ".join(f"{count} {hub_type}" for hub_type, count in hub_counts.items())
        return text


def agent_portrayal(agent):
    """
    Determine how to portray an agent on the grid.
    
    Returns a dictionary defining the agent portrayal in the visualization.
    """
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.8,
        "Layer": 0,
        "x": agent.pos[0],
        "y": agent.pos[1]
    }
    
    # Set color based on agent type
    if agent.type == 0:
        color = "#3366CC"  # Blue
    elif agent.type == 1:
        color = "#DC3912"  # Red
    elif agent.type == 2:
        color = "#FF9900"  # Orange
    elif agent.type == 3:
        color = "#109618"  # Green
    else:
        color = "#990099"  # Purple
    
    # Modify color based on happiness
    if not agent.is_happy:
        # Make unhappy agents lighter
        color_value = int(color[1:], 16)
        r = ((color_value >> 16) & 255) + 40
        g = ((color_value >> 8) & 255) + 40
        b = (color_value & 255) + 40
        
        r = min(255, r)
        g = min(255, g)
        b = min(255, b)
        
        color = f"#{r:02x}{g:02x}{b:02x}"
    
    portrayal["Color"] = color
    
    # Add a border to indicate income if income_segregation is enabled
    if agent.model.income_segregation:
        # Higher income = thicker border
        income_scaled = (agent.income / 100) * 2.5
        portrayal["stroke_color"] = "#000000"
        portrayal["stroke"] = income_scaled
    
    return portrayal


def hub_portrayal(hub, model):
    """Create a portrayal for a hub."""
    portrayal = {
        "Shape": "rect",
        "w": hub.radius * 1.5,
        "h": hub.radius * 1.5,
        "Filled": "true",
        "Layer": 1,
        "x": hub.x,
        "y": hub.y,
        "Color": "#000000",  # Base color black
        "stroke_color": "#ffffff",
        "stroke": 2
    }
    
    # Color based on hub type
    if hub.type == "work":
        portrayal["Color"] = "rgba(255, 0, 0, 0.3)"  # Red
    elif hub.type == "education":
        portrayal["Color"] = "rgba(0, 0, 255, 0.3)"  # Blue
    elif hub.type == "leisure":
        portrayal["Color"] = "rgba(0, 255, 0, 0.3)"  # Green
    elif hub.type == "shopping":
        portrayal["Color"] = "rgba(255, 165, 0, 0.3)"  # Orange
    elif hub.type == "transit":
        portrayal["Color"] = "rgba(128, 0, 128, 0.3)"  # Purple
    
    return portrayal


def barrier_portrayal(x, y, model):
    """Create a portrayal for a barrier."""
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 1,
        "x": x,
        "y": y,
        "Color": "#000000"  # Black for barriers
    }


def premium_location_portrayal(x, y, model):
    """Create a portrayal for a premium location."""
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": x,
        "y": y,
        "Color": "rgba(255, 215, 0, 0.3)"  # Gold for premium locations
    }


def get_portrayal_maker(model):
    """Create a function that will generate portrayals for all grid cells."""
    def portrayal_maker(cell):
        # First, check for barriers and premium locations
        x, y = cell[0], cell[1]
        if (x, y) in model.barriers:
            return barrier_portrayal(x, y, model)
        if (x, y) in model.premium_locations:
            return premium_location_portrayal(x, y, model)
        
        # Check for hubs
        if hasattr(model, 'hubs') and model.hubs:
            for hub in model.hubs:
                if hub.x == x and hub.y == y:
                    return hub_portrayal(hub, model)
        
        # Check for agents
        agent = model.grid.get_cell_list_contents([(x, y)])
        if agent:
            return agent_portrayal(agent[0])
        
        # Empty cell
        return None
    
    return portrayal_maker
