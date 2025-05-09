# Advanced Model Package - Enhanced Schelling Implementation

## Overview

The `advanced_model` package provides an enhanced, modular implementation of the Schelling segregation model with additional features, metrics, and visualization capabilities. This package separates functionality into distinct modules for better organization and extensibility.

## Package Structure

The package consists of the following modules:

1. **`model.py`:** Extended model with additional metrics and features
2. **`agent.py`:** Enhanced agent with more complex behavior
3. **`movement.py`:** Specialized movement algorithms
4. **`model_methods.py`:** Helper methods for model functionality
5. **`hub_features.py`:** Implementation of "hub" structures in the model
6. **`visualization.py`:** Specialized visualization for the advanced model

## Key Features

### Enhanced Model Capabilities

- **Dynamic Parameters:** Model parameters can change during simulation
- **Advanced Metrics:** Additional metrics for analyzing segregation
- **Event System:** Tracks and reports significant model events
- **State History:** Maintains history of model states for analysis
- **External Data Integration:** Can import real-world data for initialization

### Advanced Agent Behavior

- **Complex Preferences:** Agents can have more nuanced neighborhood preferences
- **Memory:** Agents can remember previous locations and neighbors
- **Learning:** Agents can adapt their preferences over time
- **Status Effects:** Agents can have different status levels affecting behavior
- **Network Formation:** Agents can form persistent connections with others

### Hub Features

The hub system introduces central locations that:

- Attract certain types of agents
- Provide benefits to nearby agents
- Create natural gathering points
- Influence movement patterns
- Generate more complex emergent behavior

### Specialized Movement Algorithms

- **Utility-Based Movement:** Agents calculate expected utility of potential locations
- **Strategic Placement:** More sophisticated placement strategies
- **Path Planning:** Agents can plan multi-step movements
- **Group Movement:** Coordinated movement of agent groups
- **Constrained Movement:** Movement limited by geographical or social barriers

## Usage

To use the advanced model:

```python
from advanced_model.model import AdvancedSchellingModel

model = AdvancedSchellingModel(
    width=20, 
    height=20,
    homophily=0.3,
    proportions=[0.5, 0.5],
    hub_count=2,
    learning_rate=0.01
)

# Run the simulation
for i in range(100):
    model.step()
    
# Analyze results
results = model.get_analysis_results()
```

## Advanced Visualization

The package includes specialized visualization for the advanced model:

```python
from advanced_model.visualization import AdvancedVisualization

viz = AdvancedVisualization(model)
viz.show()
```

The visualization includes:

- Hub visualization
- Agent network display
- Status level indicators
- Movement pattern tracking
- Enhanced metrics panels

## Integration with Base Model

The advanced model maintains compatibility with the base Schelling model while extending it:

- Can use the same agent types with enhanced behavior
- Supports the same basic parameters
- Can be used with standard Mesa tools and data collectors
- Allows gradual migration from basic to advanced features

## Technical Details

### Hub Implementation

Hubs are implemented as special locations on the grid that:

```python
class Hub:
    def __init__(self, pos, influence_radius, attraction_strength):
        self.pos = pos
        self.influence_radius = influence_radius
        self.attraction_strength = attraction_strength
        
    def calculate_influence(self, agent_pos):
        # Calculate distance-based influence on agent
        distance = self.calculate_distance(agent_pos)
        if distance > self.influence_radius:
            return 0
        return self.attraction_strength * (1 - distance/self.influence_radius)
```

### Advanced Movement Logic

The advanced movement system uses utility calculations:

```python
def calculate_location_utility(agent, position):
    # Base utility from neighbor similarity
    base_utility = calculate_similarity_utility(agent, position)
    
    # Add hub influence
    hub_utility = sum(hub.calculate_influence(position) for hub in model.hubs)
    
    # Add history effect (preference for previously visited locations)
    history_utility = calculate_history_utility(agent, position)
    
    return base_utility + hub_utility + history_utility
```

## Performance Considerations

The advanced model includes several optimizations:

- Spatial indexing for efficient neighbor lookups
- Caching of frequently computed values
- Selective updates of agent states
- Parallel processing for large simulations
- Memory-efficient history tracking
