# agent.py - Schelling Model Agent Implementation

## Overview

`agent.py` defines the `SchellingAgent` class, which represents individual agents in the Schelling segregation model. These agents make decisions about movement based on the composition of their neighborhoods and their preferences for similar neighbors.

## Agent Characteristics

### Key Features

- **Type-Based Behavior:** Each agent belongs to a specific type, which influences its neighborhood preferences
- **Happiness Calculation:** Agents evaluate their satisfaction based on the proportion of similar neighbors
- **Swapping Logic:** Agents can swap positions with other agents to increase their happiness
- **Neighbor Awareness:** Agents can examine their local neighborhood composition

### Properties

- **`agent_type`:** The type of the agent (typically represented as an integer)
- **`pos`:** Current position on the grid
- **`model`:** Reference to the parent model
- **`happy`:** Boolean indicating if the agent is satisfied with its current neighborhood
- **`happiness`:** Numerical measure of the agent's satisfaction (0.0 to 1.0)

## Technical Implementation

### SchellingAgent Class

The `SchellingAgent` class extends Mesa's `Agent` class and implements:

1. **Initialization:** Sets up the agent's type and initial position
2. **Step Method:** Determines if the agent should attempt to swap positions
3. **Happiness Calculation:** Computes the agent's happiness based on its neighborhood

### Neighborhood Analysis

The agent can analyze its neighborhood to:

- Count neighbors of the same and different types
- Calculate the proportion of similar neighbors
- Determine if it meets the homophily threshold

### Swapping Behavior

When unhappy with its current position, an agent will:

1. Look for potential swap partners
2. Evaluate if swapping would improve happiness
3. Execute a swap if it benefits at least one agent without harming the other

## Key Methods

1. **`__init__`:** Initializes the agent with a specific type and position
2. **`step`:** Determines if the agent should attempt to move
3. **`calculate_happiness`:** Computes the agent's happiness based on its neighborhood
4. **`is_happy`:** Determines if the agent is satisfied with its current position
5. **`try_to_move`:** Attempts to find a swap partner and execute a swap

## Decision-Making Logic

The agent's decision-making process follows these steps:

1. **Evaluate current happiness:**
   - Calculate the proportion of similar neighbors
   - Compare this to the homophily threshold

2. **If unhappy, seek improvement:**
   - Find potential swap partners (typically unhappy agents)
   - Calculate post-swap happiness for both agents
   - Execute swap if it improves overall happiness

3. **After moving, recalculate happiness:**
   - Update happiness based on new neighborhood
   - Store this for the next decision cycle

## Integration with Model

The agent class integrates with the model through:

- **Model Reference:** Maintains a reference to the parent model
- **Grid Interaction:** Uses the model's grid to find neighbors and move
- **Parameter Access:** Accesses model parameters like homophily threshold
- **Swap Coordination:** Coordinates swaps with other agents through the model

## Usage

The agent is typically instantiated by the model:

```python
agent = SchellingAgent(unique_id, model, agent_type, pos)
```

Agent behavior is driven by the model's step method, which calls each agent's step method:

```python
def step(self):
    if not self.is_happy():
        self.try_to_move()
    self.model.update_agent_status(self)
```

## Implementation Details

### Happiness Calculation

```python
def calculate_happiness(self):
    neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
    if not neighbors:
        return 0  # No neighbors means neutral happiness
    
    same_type_count = sum(1 for neighbor in neighbors if neighbor.agent_type == self.agent_type)
    return same_type_count / len(neighbors)
```

### Swap Decision Logic

```python
def should_swap(self, other_agent):
    # Calculate current happiness
    my_happiness = self.happiness
    other_happiness = other_agent.happiness
    
    # Calculate potential future happiness after swap
    my_future_happiness = self.calculate_future_happiness(other_agent.pos)
    other_future_happiness = other_agent.calculate_future_happiness(self.pos)
    
    # Swap if it makes at least one agent happier without making the other unhappier
    return ((my_future_happiness > my_happiness and other_future_happiness >= other_happiness) or
            (other_future_happiness > other_happiness and my_future_happiness >= my_happiness))
```

## Performance Considerations

- **Efficient Neighborhood Analysis:** Optimized methods for analyzing local neighborhoods
- **Selective Swapping:** Only consider swaps that have potential to improve happiness
- **Caching:** Store happiness values to avoid redundant calculations
