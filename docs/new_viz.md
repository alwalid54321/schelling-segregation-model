# new_viz.py - Rewritten Schelling Model Visualization

## Overview

`new_viz.py` is a complete rewrite of the Schelling model visualization system with an improved architecture, better performance, and enhanced features. This version was created to address issues in the original visualization and provide a more robust implementation.

## Key Features

- **Improved Architecture:** Clear separation of concerns between model and visualization
- **Enhanced Performance:** Optimized for large grids and extended simulations
- **Robust Error Handling:** Comprehensive error management to prevent crashes
- **Expanded Visualization Options:** Additional view modes and analysis tools
- **Improved UI Layout:** Better spacing and organization of controls

## Technical Implementation

### Class Structure

The visualization is implemented using a cleaner architecture with clear separation between:

1. **Model Management:** Handles model creation, stepping, and data collection
2. **UI Components:** Manages all interactive controls
3. **Visualization:** Renders the grid and metrics
4. **Data Collection:** Gathers and processes model metrics

Key classes and methods:

- **`SchellingVisualizer`**: Main class that coordinates all components
- **`ModelManager`**: Handles model lifecycle and data collection
- **`UIController`**: Manages all UI elements and user interactions
- **`GridRenderer`**: Handles grid visualization with multiple display modes
- **`MetricsPlotter`**: Manages time series plots of model metrics

### Visualization Modes

1. **Standard Grid View:** Color-coded visualization of agent types
2. **Network View:** Shows connections between agents of the same type
3. **Happiness View:** Heat map showing agent happiness levels
4. **Cluster Analysis:** Visualization highlighting agent clusters
5. **Movement Tracking:** Shows recent agent movements

### Enhanced Controls

- **Parameter Controls:** Sliders for all model parameters with immediate feedback
- **View Controls:** Radio buttons for selecting visualization mode
- **Simulation Controls:** Buttons for step, run/pause, reset, save/load
- **Analysis Controls:** Options for different metrics and analysis methods

### Pattern Management

Improved pattern initialization with:

- More precise control over pattern parameters
- Preview functionality for patterns before applying
- Ability to combine patterns for complex distributions
- Custom pattern creation and editing

### Advanced Features

1. **Batch Simulation:** Run multiple simulations with parameter variations
2. **Parameter Sensitivity Analysis:** Automatically test different parameter combinations
3. **Statistical Analysis:** Compute statistics across multiple simulation runs
4. **Advanced Metrics:** Additional segregation and happiness metrics
5. **Export Functionality:** Save results and visualizations in various formats

## Usage

The new visualization can be instantiated with:

```python
visualizer = SchellingVisualizer(
    grid_width=20,
    grid_height=20,
    homophily_threshold=0.3,
    agent_proportions=[0.5, 0.5],
    initial_pattern="random"
)
```

To start the visualization:

```python
visualizer.start()
```

## Technical Improvements

1. **Better Error Handling:** Robust error handling to prevent crashes and provide informative messages
2. **Memory Management:** Improved memory usage for long-running simulations
3. **UI Responsiveness:** Enhanced event handling for a more responsive interface
4. **Modularity:** More modular design for easier extension and customization
5. **Consistent Data Flow:** Reliable synchronization between model state and visualization

## Compatibility

- Compatible with all features of the Mesa framework
- Works with the existing Schelling model implementation
- Supports larger grid sizes than the original visualization
- Handles different agent types and behaviors
