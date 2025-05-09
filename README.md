# Schelling Segregation Model Simulation

## Overview

This project implements Thomas Schelling's segregation model, which demonstrates how individual preferences regarding neighbors can lead to segregation patterns, even when individuals are not explicitly prejudiced. The model shows how even a mild preference for having neighbors of the same type can lead to segregation over time.

Our implementation extends the classic Schelling model with advanced features, interactive visualizations, and multiple analysis methods to explore segregation dynamics in depth.

## Project Structure

The project is organized into several components:

### Core Components

- **model.py**: Contains the `SchellingModel` class that extends Mesa's `Model` class. This handles the model initialization, step functions, and data collection.
- **agent.py**: Defines the `SchellingAgent` class with agent behaviors, happiness calculations, and movement rules.
- **run.py**: Simple script to run the model with default parameters.

### Visualization Components

- **fixed_viz.py**: Our initial enhanced visualization using Matplotlib, with interactive controls and multiple view modes.
- **new_viz.py**: A completely rewritten visualization with improved architecture, performance, and features.
- **simplified_viz.py**: A more streamlined visualization focused on stability and core functionality.
- **schelling_matplotlib.py**: Basic Matplotlib visualization of the model.

### Server Components

- **server.py**: Mesa server implementation for web-based visualization.
- **mesa_server.py**: Enhanced Mesa server with additional analytics.

### Advanced Model Package

- **advanced_model/**: A modular implementation with enhanced features:
  - **model.py**: Extended model with additional metrics
  - **agent.py**: Enhanced agent with more complex behavior
  - **movement.py**: Specialized movement algorithms
  - **model_methods.py**: Helper methods for model functionality
  - **hub_features.py**: Implementation of "hub" structures in the model
  - **visualization.py**: Specialized visualization for the advanced model

### Alternative Implementations

- **agent_new.py**, **model_new.py**, **server_new.py**: Alternative implementations exploring different model designs and optimizations.

### Runners

- **mesa_schelling_run.py**: Script to run the Mesa version of the model.
- **run_advanced_schelling.py**: Script to run the advanced model version.

### Documentation

Detailed documentation is available in the `docs/` directory:

- **[fixed_viz.md](docs/fixed_viz.md)**: Comprehensive guide to the enhanced Matplotlib visualization with interactive controls and multiple view modes.
- **[new_viz.md](docs/new_viz.md)**: Details about the completely rewritten visualization with improved architecture and enhanced features.
- **[simplified_viz.md](docs/simplified_viz.md)**: Documentation for the streamlined visualization focused on stability and core functionality.
- **[mesa_server.md](docs/mesa_server.md)**: Guide to the web-based visualization server with real-time updates and interactive controls.
- **[model.md](docs/model.md)**: In-depth explanation of the core Schelling model implementation, including its unique fully-populated grid approach.
- **[agent.md](docs/agent.md)**: Detailed documentation of agent behavior, happiness calculation, and the swapping mechanism.
- **[advanced_model.md](docs/advanced_model.md)**: Guide to the enhanced modular implementation with hub features and specialized movement algorithms.
- **[runners.md](docs/runners.md)**: Information about all the runner scripts and entry points for running different versions of the simulation.

## Key Features

### 1. Unique Model Characteristics

- **Fully Populated Grid**: Unlike traditional Schelling models, our implementation maintains a fully populated grid with no empty spaces, creating a more realistic and visually dense simulation.
- **Agent Swapping**: Instead of moving to empty cells, agents swap positions with other agents, requiring coordination between agents.
- **Happiness-Improving Swap Mechanism**: Agents only swap if at least one agent gets happier and the other doesn't get unhappier, introducing a cooperative dynamic.

### 2. Multiple Initialization Patterns

The model supports several initial agent distribution patterns:

- **Random**: Agents are distributed randomly across the grid
- **Alternating**: Agents are arranged in an alternating checkerboard pattern
- **Clusters**: Agents are grouped into type-specific clusters
- **Stripes**: Agents are arranged in vertical stripes by type

### 3. Visualization Modes

Our visualizations offer multiple ways to analyze the model:

- **Normal View**: Standard grid view showing agent positions and types
- **Network View**: Shows connections between similar agents, highlighting community structures
- **Happiness View**: Heat map visualization of agent happiness levels

### 4. Interactive Controls

The visualization includes interactive controls for:

- Adjusting homophily preference
- Setting grid width and height
- Changing agent type proportions
- Selecting initialization patterns
- Stepping through simulation or running continuously
- Adjusting simulation speed
- Saving and loading model states

### 5. Analytics

The model captures various metrics during simulation:

- **Segregation Index**: Measures the overall level of segregation
- **Average Happiness**: Tracks agent satisfaction over time
- **Cluster Analysis**: Identifies and measures agent clusters

### 6. Save/Load Functionality

- Save model states to disk for later analysis
- Load saved models to continue simulation or compare scenarios

### 7. Multiple Visualization Implementations

We provide several visualization options to suit different needs:

- **fixed_viz.py**: Our fully-featured interactive visualization with all controls and visualization modes
- **new_viz.py**: A complete rewrite with improved architecture and performance for complex simulations
- **simplified_viz.py**: A streamlined version prioritizing stability for reliable visualization
- **mesa_server.py**: Web-based visualization accessible through any browser
- **schelling_matplotlib.py**: Simple visualization for basic analysis and performance testing

## How It Works

### Model Mechanics

1. **Initialization**:
   - The grid is populated according to the selected pattern
   - Each cell contains exactly one agent
   - Agents are assigned one of the available types based on specified proportions

2. **Happiness Calculation**:
   - Each agent calculates its happiness based on the proportion of similar neighbors
   - An agent is "happy" if the proportion of similar neighbors meets or exceeds the homophily threshold

3. **Movement Rules**:
   - During each step, unhappy agents attempt to swap positions with another agent
   - A swap occurs only if it improves at least one agent's happiness without decreasing the other's
   - This creates a gradual optimization process that leads to segregation patterns

4. **Metrics Calculation**:
   - After each step, the model calculates and records various metrics
   - These metrics are displayed in real-time plots alongside the grid

### Running the Simulation

To run the basic model with visualization:

```
python run.py
```

To run the advanced model:

```
python run_advanced_schelling.py
```

To run the web-based visualization:

```
python mesa_server.py
```

## Results and Insights

### Observed Phenomena

The Schelling model demonstrates several key insights about segregation:

1. **Emergent Segregation**: Even with relatively low homophily preferences (e.g., 30%), clear segregation patterns emerge over time.

2. **Phase Transitions**: The model exhibits phase transitions where small changes in homophily preference lead to dramatic changes in overall segregation.

3. **Pattern Effects**: Different initialization patterns lead to different segregation dynamics:
   - Random initialization typically leads to many small clusters
   - Alternating patterns create interesting fractal-like segregation structures
   - Cluster initialization tends to reinforce and expand existing clusters
   - Stripe patterns often evolve into more block-like structures

4. **Equilibrium States**: The model eventually reaches stable equilibrium states where most or all agents are satisfied with their neighborhoods.

### Visual Results

The visualization produces several types of outputs:

1. **Grid Visualization**: Shows the spatial distribution of agents by type
2. **Network Visualization**: Reveals community structures and connections
3. **Time Series Plots**: Track segregation index and happiness over time

Sample outputs can be found in `schelling_results.png` and `schelling_results_simple.png`.

## Dependencies

The project requires the following Python packages:

- mesa
- matplotlib
- numpy
- networkx (for network visualization)
- pickle (for save/load functionality)

All dependencies can be installed via:

```
pip install -r requirements.txt
```

## Choosing the Right Visualization

Each visualization implementation has specific strengths:

- **fixed_viz.py**: Best for general exploration with a balance of features and performance
- **new_viz.py**: Ideal for complex analysis with advanced visualization needs
- **simplified_viz.py**: Best for stability and reliable performance
- **mesa_server.py**: Perfect for sharing visualizations with others
- **schelling_matplotlib.py**: Optimal for performance testing and simple analysis

See [docs/runners.md](docs/runners.md) for detailed information on how to run each visualization.

## Advanced Model Features

The advanced model package (`advanced_model/`) extends the base implementation with:

- **Hub Structures**: Central locations that influence agent movement
- **Complex Agent Behavior**: More sophisticated decision-making
- **Enhanced Metrics**: Additional analytical tools
- **Specialized Visualization**: Custom visualization for advanced features

See [docs/advanced_model.md](docs/advanced_model.md) for complete details.

## Future Enhancements

Possible future enhancements include:

1. More complex agent behaviors and preferences
2. Additional metrics for segregation analysis
3. 3D visualization capabilities
4. Agent learning and adaptation
5. Integration with real-world demographic data

## References

- Schelling, T. C. (1971). "Dynamic Models of Segregation." Journal of Mathematical Sociology, 1, 143-186.
- Wilensky, U. (1997). NetLogo Segregation model. Center for Connected Learning and Computer-Based Modeling, Northwestern University. http://ccl.northwestern.edu/netlogo/models/Segregation
- Mesa Documentation: https://mesa.readthedocs.io/
