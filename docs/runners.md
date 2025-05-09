# Runner Scripts - Schelling Model Entry Points

## Overview

This document describes the various runner scripts that serve as entry points to run the Schelling segregation model. These scripts provide different ways to initialize, run, and visualize the model.

## Runner Files

### run.py - Basic Runner

`run.py` provides the simplest way to run the Schelling model with the standard visualization:

```python
from model import SchellingModel
from fixed_viz import SchellingVisualization

# Create and display visualization
viz = SchellingVisualization(
    width=20,
    height=20,
    homophily=0.3,
    proportions=[0.5, 0.5],
    initial_pattern="random"
)
viz.show()
```

**Usage:** `python run.py`

### mesa_schelling_run.py - Mesa Framework Runner

`mesa_schelling_run.py` runs the model using Mesa's built-in visualization framework:

```python
from mesa_schelling.model import MesaSchellingModel
from mesa_schelling.visualization import get_mesa_visualization

# Create model and server
model = MesaSchellingModel(20, 20, 0.3, [0.5, 0.5])
server = get_mesa_visualization(model)
server.launch()
```

**Usage:** `python mesa_schelling_run.py`

### run_advanced_schelling.py - Advanced Model Runner

`run_advanced_schelling.py` runs the enhanced version of the model from the advanced_model package:

```python
from advanced_model.model import AdvancedSchellingModel
from advanced_model.visualization import AdvancedVisualization

# Create model and visualization
model = AdvancedSchellingModel(20, 20, 0.3, [0.5, 0.5])
viz = AdvancedVisualization(model)
viz.show()
```

**Usage:** `python run_advanced_schelling.py`

### schelling_matplotlib.py - Simple Matplotlib Visualization

`schelling_matplotlib.py` provides a basic visualization using just Matplotlib, without the interactive controls:

```python
from model import SchellingModel
import matplotlib.pyplot as plt
import numpy as np

# Create model
model = SchellingModel(20, 20, 0.3, [0.5, 0.5])

# Run simulation and visualize
for i in range(100):
    model.step()
    # Visualization code...
```

**Usage:** `python schelling_matplotlib.py`

### schelling_animated.py - Animated Visualization

`schelling_animated.py` creates an animated visualization of the model evolution:

```python
from model import SchellingModel
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Animation code for model evolution
```

**Usage:** `python schelling_animated.py`

### schelling_simple.py - Minimal Implementation

`schelling_simple.py` contains a minimal implementation of the Schelling model without the Mesa framework, useful for educational purposes:

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple implementation of Schelling model
```

**Usage:** `python schelling_simple.py`

## Feature Comparison

| Runner                    | Interactive | Web-based | Framework | Advanced Features |
|---------------------------|-------------|-----------|-----------|-------------------|
| run.py                    | ✓           | ✗         | Mesa      | ✓                 |
| mesa_schelling_run.py     | ✓           | ✓         | Mesa      | ✓                 |
| run_advanced_schelling.py | ✓           | ✗         | Mesa      | ✓✓                |
| schelling_matplotlib.py   | ✗           | ✗         | Mesa      | ✗                 |
| schelling_animated.py     | Limited     | ✗         | Mesa      | ✗                 |
| schelling_simple.py       | ✗           | ✗         | None      | ✗                 |

## Command-Line Arguments

Some runners support command-line arguments for customization:

### run.py

```
python run.py [--width WIDTH] [--height HEIGHT] [--homophily HOMOPHILY] [--pattern PATTERN]
```

### mesa_schelling_run.py

```
python mesa_schelling_run.py [--port PORT] [--width WIDTH] [--height HEIGHT]
```

### run_advanced_schelling.py

```
python run_advanced_schelling.py [--width WIDTH] [--height HEIGHT] [--homophily HOMOPHILY] [--hubs HUBS]
```

## Performance Considerations

- **run.py / fixed_viz.py:** Balanced performance and features, suitable for most use cases
- **mesa_schelling_run.py:** Good for sharing results, potentially slower for large grids
- **run_advanced_schelling.py:** Most feature-rich but may be slower for very large simulations
- **schelling_matplotlib.py / schelling_simple.py:** Best performance, limited features

## Visualization Outputs

The various runners produce different types of output:

1. **Interactive Matplotlib:** Live, interactive grid with controls (fixed_viz.py, new_viz.py)
2. **Web Interface:** Browser-based visualization (mesa_server.py)
3. **Static Images:** Simple grid snapshots (schelling_matplotlib.py)
4. **Animations:** Pre-rendered animations of model evolution (schelling_animated.py)

## Choosing the Right Runner

- For interactive exploration: `run.py` with fixed_viz.py
- For sharing with others: `mesa_schelling_run.py`
- For advanced analysis: `run_advanced_schelling.py`
- For educational purposes: `schelling_simple.py`
- For performance testing: `schelling_matplotlib.py`

## Integration With Your Code

To run the model from your own code:

```python
# Using the standard visualization
from fixed_viz import SchellingVisualization

viz = SchellingVisualization(
    width=20,
    height=20,
    homophily=0.3,
    proportions=[0.5, 0.5],
    initial_pattern="random"
)
viz.show()

# Using just the model
from model import SchellingModel

model = SchellingModel(
    width=20,
    height=20,
    homophily=0.3,
    proportions=[0.5, 0.5]
)

for i in range(100):
    model.step()
    # Process results...
```
