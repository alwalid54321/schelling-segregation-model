# mesa_server.py - Web-Based Visualization Server

## Overview

`mesa_server.py` provides a web-based visualization for the Schelling Segregation Model using Mesa's built-in server capabilities. This approach offers a browser-based interface with interactive elements and real-time model visualization.

## Key Features

- **Web Interface:** Browser-based visualization accessible through any modern web browser
- **Real-time Updates:** Live updates of the model state as it runs
- **Interactive Controls:** Web-based UI controls for model parameters
- **Data Downloads:** Ability to download simulation data for further analysis
- **Shareable Interface:** Can be accessed by multiple users simultaneously

## Technical Implementation

### Server Configuration

The Mesa server is configured with:

1. **Model Parameters:** Defines user-adjustable parameters for the model
2. **Element Visualization:** Sets up grid, chart, and text elements
3. **Page Layout:** Configures the arrangement of visualization elements
4. **Data Collection:** Specifies which model data to collect and display

### Visualization Elements

1. **Grid Visualization:**
   - Color-coded representation of the grid
   - Interactive cell inspection on hover
   - Adjustable visualization properties

2. **Chart Elements:**
   - Line charts for segregation index and happiness
   - Bar charts for agent type distributions
   - Histogram for happiness distribution

3. **Text Elements:**
   - Model statistics and current parameters
   - Step count and runtime information
   - Analysis results and insights

### Interactive Controls

The server provides web controls for:

- Adjusting homophily threshold
- Setting agent type proportions
- Selecting initialization patterns
- Controlling simulation speed
- Starting, pausing, and resetting the simulation

### Data Management

- **Real-time Collection:** Gathers data during model execution
- **CSV Export:** Allows downloading simulation data as CSV
- **State Serialization:** Can save and load model states

## Usage

To start the Mesa server:

```python
python mesa_server.py
```

Then open a web browser and navigate to:

```
http://localhost:8521
```

## Technical Details

1. **Server Architecture:**
   - Uses Mesa's ModularServer
   - Implements custom visualization modules
   - Handles asynchronous updates

2. **Browser Communication:**
   - WebSocket connection for real-time updates
   - JSON data format for model state
   - Browser rendering using HTML5 Canvas

3. **Performance Considerations:**
   - Optimized data transfer for large grids
   - Throttled updates for better performance
   - Efficient browser-side rendering

## Advantages

- **No External Dependencies:** Uses only web technologies built into browsers
- **Cross-Platform:** Works on any device with a modern web browser
- **Shareable:** Can be accessed by multiple users if hosted on a network
- **Extensible:** Can be extended with custom JavaScript visualizations
- **Interactive:** Provides rich interaction possibilities beyond Matplotlib

## Limitations

- **Performance:** May be slower than native visualization for very large grids
- **Flexibility:** Limited to visualization elements supported by Mesa
- **Customization:** Requires JavaScript knowledge for advanced customization

## Network Configuration

By default, the server runs on:
- Host: localhost (127.0.0.1)
- Port: 8521

To make it accessible from other machines on the network, change the host to '0.0.0.0':

```python
server.launch(host='0.0.0.0', port=8521)
```
