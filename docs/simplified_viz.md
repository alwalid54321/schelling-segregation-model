# simplified_viz.py - Streamlined Schelling Model Visualization

## Overview

`simplified_viz.py` provides a streamlined, minimal implementation of the Schelling model visualization. This version prioritizes stability and core functionality, focusing on delivering a robust visualization that consistently works without errors.

## Key Features

- **Stability Focus:** Designed to be extremely stable and error-resistant
- **Core Functionality:** Provides all essential visualization features without unnecessary complexity
- **Performance Optimized:** Streamlined for better performance with larger grids
- **User-Friendly Interface:** Simplified controls with intuitive layout
- **Robust Error Prevention:** Built to avoid common visualization errors

## Technical Implementation

### Minimalist Architecture

The visualization uses a simplified architecture focusing on reliability:

1. **Single Class Design:** All functionality is contained within a cohesive `SimplifiedSchellingViz` class
2. **Direct Data Access:** Clear, direct access to model state
3. **Simplified Event Handling:** Straightforward event management
4. **Consistent Update Flow:** Reliable update process to prevent synchronization issues

### Core Components

1. **Grid Display:** Simple, efficient grid rendering using Matplotlib's `imshow`
2. **Basic Metrics:** Displays only the most essential metrics:
   - Segregation index over time
   - Average happiness over time
3. **Essential Controls:**
   - Step/Run/Stop buttons
   - Reset button
   - Pattern selection
   - Homophily slider

### Pattern Implementation

Implements the core pattern initializations:

1. **Random:** Standard random distribution of agents
2. **Alternating:** Basic checkerboard pattern
3. **Clusters:** Simple cluster implementation
4. **Stripes:** Basic vertical stripe pattern

### Data Management

- **Minimal State:** Maintains only essential state variables
- **Direct Data Transfer:** Simplified data collection and visualization
- **Synchronized Updates:** Ensures grid and plot updates are always synchronized

## Usage

The simplified visualization can be used with:

```python
viz = SimplifiedSchellingViz(
    width=20,
    height=20,
    homophily=0.3,
    proportions=[0.5, 0.5]
)
```

And started with:

```python
viz.show()
```

## Technical Benefits

1. **Reduced Error Surface:** Fewer components means fewer potential failure points
2. **Consistent Performance:** More predictable performance across different system configurations
3. **Lower Resource Usage:** Reduced memory and processing requirements
4. **Shape Mismatch Prevention:** Specifically designed to avoid Matplotlib shape mismatch errors
5. **Transparent Code:** Easier to understand, debug, and modify

## When to Use

This simplified visualization is ideal when:

1. Stability is the top priority
2. Only core features are needed
3. Resources are limited
4. Quick setup is desired
5. The focus is on model dynamics rather than advanced analysis

## Implementation Notes

- Uses synchronized data collection to prevent shape mismatch errors
- Implements error checking before all plotting operations
- Simplifies UI event handling for more reliable interaction
- Provides direct access to model internals when needed
- Includes basic error recovery mechanisms
