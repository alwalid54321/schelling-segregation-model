# model.py - Core Schelling Segregation Model

## Overview

`model.py` contains the implementation of the Schelling segregation model using the Mesa framework. This file defines the `SchellingModel` class, which is the core component that drives the simulation dynamics.

## Model Characteristics

### Unique Implementation Features

- **Fully Populated Grid:** Unlike traditional Schelling models, this implementation maintains a fully populated grid with no empty spaces.
- **Agent Swapping:** Instead of moving to empty cells, agents swap positions with other agents.
- **Happiness-Improving Swap Mechanism:** Agents only swap if at least one agent gets happier and the other doesn't get unhappier.

### Core Parameters

- **Width/Height:** Dimensions of the grid
- **Homophily Threshold:** Minimum desired fraction of similar neighbors
- **Agent Type Proportions:** Relative proportions of different agent types

## Technical Implementation

### SchellingModel Class

The `SchellingModel` class extends Mesa's `Model` class and implements:

1. **Initialization:** Sets up the grid, schedules, and creates agents
2. **Step Method:** Advances the model by one step, allowing agents to swap positions
3. **Data Collection:** Monitors metrics like segregation index and happiness

### Grid Structure

- Uses Mesa's `MultiGrid` to allow multiple agents in the same cell (though in practice, it maintains one agent per cell)
- Implements grid methods for finding neighbors and calculating agent density

### Agent Management

- Creates and places agents according to specified proportions
- Maintains agent lists for easy access and manipulation
- Provides methods for finding potential swap partners

### Data Collection System

The model uses Mesa's `DataCollector` to track:

- **Segregation Index:** Measures the overall level of segregation
- **Average Happiness:** Tracks agent satisfaction over time
- **Type Distribution:** Monitors the proportion of different agent types

## Key Methods

1. **`__init__`:** Sets up model parameters, grid, and agents
2. **`step`:** Advances the simulation by one step
3. **`get_segregation_index`:** Calculates the current segregation level
4. **`get_average_happiness`:** Computes mean happiness across all agents
5. **`find_swap_partner`:** Identifies potential partners for agent swapping
6. **`swap_agents`:** Exchanges positions between two agents

## Data Collection Functions

1. **`calculate_segregation`:** Computes the segregation index for the data collector
2. **`calculate_happiness`:** Computes the average happiness for the data collector

## Usage

The model can be instantiated with:

```python
model = SchellingModel(
    width=20,
    height=20,
    homophily=0.3,
    proportions=[0.5, 0.5]
)
```

To advance the simulation one step:

```python
model.step()
```

To retrieve collected data:

```python
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()
```

## Integration Points

- **Agent Integration:** Works with the `SchellingAgent` class from `agent.py`
- **Visualization Integration:** Provides data and state for visualization components
- **Server Integration:** Can be used with Mesa's server for web-based visualization

## Performance Considerations

- **Efficient Neighbor Calculation:** Optimized methods for finding neighbors
- **Swap Partner Selection:** Efficient algorithms for identifying potential swap partners
- **Grid Operations:** Minimized iteration over the entire grid when possible

## Mathematical Foundation

The segregation index is calculated as:

1. For each agent, compute the fraction of similar neighbors
2. Compare this fraction to a random distribution baseline
3. Calculate the deviation from this baseline
4. Average across all agents

This provides a measure of segregation that accounts for:
- The actual distribution of agent types
- The spatial arrangement of agents
- The expected values in a random distribution
