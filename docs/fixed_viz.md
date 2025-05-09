# fixed_viz.py - Enhanced Matplotlib Visualization

## Overview

`fixed_viz.py` provides an enhanced visualization for the Schelling Segregation Model using Matplotlib. This visualization includes interactive controls, multiple view modes, and real-time metrics tracking.

## Key Features

- **Interactive Controls:** Buttons, sliders, and radio buttons for adjusting model parameters
- **Multiple Visualization Modes:** Standard grid view, network view, and happiness heatmap
- **Real-time Metrics:** Tracks and displays segregation index and happiness over time
- **Save/Load Functionality:** Allows saving and loading model states
- **Pattern Initialization:** Supports multiple initial agent distribution patterns

## Technical Implementation

### Class Structure

The visualization is implemented as a class `SchellingVisualization` with the following key methods:

1. **`__init__`**: Initializes the visualization, creates the figure, axes, and initial model
2. **`setup_controls`**: Sets up interactive UI elements (buttons, sliders, radio buttons)
3. **`setup_plot`**: Initializes the grid plot and metrics plots
4. **`update_plot`**: Updates the visualization after each model step
5. **`run_step`**: Advances the model one step and updates the visualization
6. **`run_animation`**: Continuously runs the model with animation
7. **Pattern initialization methods:** Methods for creating different initial agent distributions

### Visualization Components

1. **Grid Visualization:**
   - Uses `imshow` to display the grid
   - Colors represent different agent types
   - Uses custom colormaps for better visibility

2. **Metrics Plots:**
   - Line plots showing segregation index and average happiness over time
   - Updates in real-time as the model runs

3. **Network Visualization:**
   - Shows connections between similar agents
   - Uses a force-directed layout algorithm

4. **UI Components:**
   - Buttons: Step, Run/Stop, Reset, Save, Load
   - Sliders: Homophily threshold, proportions of agent types
   - Radio buttons: Pattern selection, visualization mode

### Pattern Initialization

The visualization handles pattern initialization through dedicated methods:

1. **`_initialize_random_pattern`**: Randomly places agents on the grid
2. **`_initialize_alternating_pattern`**: Creates a checkerboard pattern
3. **`_initialize_cluster_pattern`**: Groups agents into type-specific clusters
4. **`_initialize_stripe_pattern`**: Arranges agents in vertical stripes

### Data Management

- **Data Arrays:** Maintains arrays for x-values, happiness data, and segregation data
- **Synchronization:** Ensures data arrays are correctly synchronized to prevent shape mismatch errors
- **State Tracking:** Tracks the current pattern, animation state, and parameter values

## Usage

The visualization can be instantiated with various parameters to customize the initial state:

```python
viz = SchellingVisualization(
    width=20,
    height=20,
    homophily=0.3,
    proportions=[0.5, 0.5],
    initial_pattern="random"
)
```

To display the visualization:

```python
viz.show()
```

## Technical Challenges

1. **Grid Access:** Mesa's MultiGrid requires specific access patterns using `coord_iter()`
2. **Agent Properties:** Ensures correct access to agent properties (e.g., `agent.agent_type`)
3. **Data Synchronization:** Carefully manages data arrays to prevent shape mismatch errors
4. **Matplotlib Integration:** Properly manages Matplotlib's event loop and UI components

## Compatibility Notes

- Works with Mesa's `MultiGrid` and the `SchellingAgent` implementation
- Compatible with matplotlib 3.5+ (includes handling for deprecated colormaps)
- Requires numpy for array operations
- Uses pickle for save/load functionality
