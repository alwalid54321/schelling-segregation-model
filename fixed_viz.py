"""
Schelling Segregation Model with Matplotlib Visualization
A simple and clean implementation that doesn't rely on Mesa's visualization components
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pickle
import time
from collections import defaultdict

from model import SchellingModel
from agent import SchellingAgent

# Custom visualization class
class SchellingVisualization:
    def __init__(self, width=20, height=20, density=1.0, homophily=0.3, num_agent_types=2):
        """Initialize visualization with grid and controls"""
        print("Starting Schelling Segregation Model with Matplotlib Visualization...")
        
        # Create figure with larger size to prevent UI elements from overlapping
        self.fig = plt.figure(figsize=(16, 12))
        
        # Set window title
        self.fig.canvas.manager.set_window_title('Schelling Segregation Model')
        
        # Create a unified grid layout to ensure proper spacing
        grid_spec = self.fig.add_gridspec(6, 6, hspace=0.4, wspace=0.4)
        
        # Create grid with proper boundaries - use less total space to allow more room for controls
        self.ax_grid = self.fig.add_subplot(grid_spec[0:5, 0:4])
        self.ax_grid.set_title('Schelling Grid', fontsize=12)
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        # Data plots for metrics (bottom row)
        self.ax_plot = self.fig.add_subplot(grid_spec[5, 0:4])
        self.ax_plot.set_title('Model Metrics', fontsize=12)
        self.ax_plot.set_xlabel('Steps')
        self.ax_plot.set_ylabel('Value')
        
        # Happiness line
        self.happiness_line, = self.ax_plot.plot([], [], 'g-', linewidth=2, label='Happiness')
        # Segregation line
        self.segregation_line, = self.ax_plot.plot([], [], 'r-', linewidth=2, label='Segregation')
        self.ax_plot.legend(loc='upper left')
        self.ax_plot.set_ylim(0, 1)
        
        # Controls in the right column
        # Pattern selection radio
        self.pattern_ax = self.fig.add_subplot(grid_spec[0:1, 4:6])
        self.pattern_ax.set_title('Pattern', fontsize=12)
        self.pattern_ax.axis('off')
        self.pattern_radio = RadioButtons(
            self.pattern_ax,
            ('Alternating', 'Random', 'Clusters', 'Stripes'),
            active=0
        )
        self.pattern_radio.on_clicked(self.change_pattern)
        
        # View mode radio
        self.view_ax = self.fig.add_subplot(grid_spec[1:2, 4:6])
        self.view_ax.set_title('View Mode', fontsize=12)
        self.view_ax.axis('off')
        self.view_radio = RadioButtons(
            self.view_ax,
            ('Normal', 'Networks', 'Happiness'),
            active=0
        )
        self.view_mode = 'normal'  # Default view mode
        self.view_radio.on_clicked(self.change_view_mode)
        
        # Parameter sliders with more spacing
        self.homophily_ax = self.fig.add_subplot(grid_spec[2:3, 4:6])
        self.homophily_ax.set_title('Homophily', fontsize=12)
        self.homophily_slider = Slider(
            self.homophily_ax,
            'Threshold',
            0.0,
            1.0,
            valinit=homophily
        )
        self.homophily_slider.on_changed(self.update_homophily)
        
        self.types_ax = self.fig.add_subplot(grid_spec[3:4, 4:6])
        self.types_ax.set_title('Agent Types', fontsize=12)
        self.types_slider = Slider(
            self.types_ax,
            'Types',
            2,
            5,
            valinit=num_agent_types,
            valstep=1
        )
        self.types_slider.on_changed(self.update_agent_types)
        
        # Control buttons with better spacing
        button_axes = {
            'step': self.fig.add_subplot(grid_spec[4, 4]),
            'run': self.fig.add_subplot(grid_spec[4, 5]),
            'reset': self.fig.add_subplot(grid_spec[5, 4]),
            'save': self.fig.add_subplot(grid_spec[5, 5])
        }
        
        # Create a separate axis for the load button with proper positioning
        load_ax = plt.axes([0.77, 0.05, 0.1, 0.04])  # [left, bottom, width, height]
        button_axes['load'] = load_ax
        
        # Remove axes decorations from button subplots
        for ax in button_axes.values():
            ax.axis('off')
        
        # Step and run buttons
        self.step_button = Button(button_axes['step'], 'Step')
        self.step_button.on_clicked(self.step_model)
        
        self.run_button = Button(button_axes['run'], 'Run')
        self.run_button.on_clicked(self.toggle_running)
        
        self.reset_button = Button(button_axes['reset'], 'Reset')
        self.reset_button.on_clicked(self.reset_model)
        
        # Save/load buttons
        self.save_button = Button(button_axes['save'], 'Save')
        self.save_button.on_clicked(self.save_model)
        
        self.load_button = Button(button_axes['load'], 'Load')
        self.load_button.on_clicked(self.load_model)
        
        # Create the model with the specified parameters
        self.model = SchellingModel(
            width=width,
            height=height,
            density=density,
            homophily=homophily,
            num_agent_types=num_agent_types
        )
        
        # Store the current pattern type (for UI only, not used in model)
        self.current_pattern = 'alternating'
        
        # Status text for displaying metrics
        self.status_text = self.ax_grid.text(
            1.02, 0.98, "", transform=self.ax_grid.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Network visualization elements
        self.network_lines = []
        
        # Animation control
        self.is_running = False
        self.ani = None
        
        # Data for tracking metrics
        self.x_data = [0]  # For plotting steps
        self.happiness_data = []
        self.segregation_data = []
        
        # Initialize visualization
        self.update_grid()
        self.setup_plot()
        
        # Use constrained_layout to prevent overlapping elements
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.35)
        
    def update_grid(self):
        """Update the grid visualization"""
        # Initialize grid matrix
        grid_matrix = np.zeros((self.model.height, self.model.width))
        
        # Clear the axes
        self.ax_grid.clear()
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        # Get the current state of the grid
        for cell_content, pos in self.model.grid.coord_iter():
            # In Mesa's MultiGrid, cell_content is a list of agents
            if cell_content:  # If the cell is not empty
                for agent in cell_content:
                    if isinstance(agent, SchellingAgent):
                        x, y = pos
                        grid_matrix[y][x] = agent.agent_type + 1  # +1 so that 0 is empty
        
        # Choose colormap based on number of agent types and matplotlib version
        try:
            # For newer matplotlib versions (3.7+)
            cmap = plt.colormaps['tab10']
        except AttributeError:
            try:
                # For matplotlib versions between 3.6 and 3.7
                cmap = plt.cm.colormaps['tab10']
            except (AttributeError, KeyError):
                # For older matplotlib versions
                cmap = plt.cm.get_cmap('tab10', self.model.num_agent_types + 1)
        
        # Display the grid
        if self.view_mode == 'normal':
            # Standard grid view
            self.im = self.ax_grid.imshow(grid_matrix, cmap=cmap, interpolation='nearest')
            
            # Show segregation
            # Get data from datacollector
            if self.model.datacollector is not None and len(self.model.datacollector.model_vars['Segregation']) > 0:
                segregation = self.model.datacollector.model_vars['Segregation'][-1]
            else:
                # Calculate segregation manually if not available
                segregation = 0
                
            happy_pct = (len(self.model.agents) - len(self.model.unhappy_agents)) / len(self.model.agents) if self.model.agents else 0
            
            status_text = f"Step: {len(self.x_data) - 1}\nHappy: {happy_pct:.2%}\nSegregation: {segregation:.3f}"
            status_text += f"\nPattern: {self.current_pattern}"
            
            self.status_text.set_text(status_text)
            
        elif self.view_mode == 'network':
            # Network view - show connections between similar agents
            self.im = self.ax_grid.imshow(grid_matrix, cmap=cmap, interpolation='nearest', alpha=0.7)
            
            # Clear existing network lines
            for line in self.network_lines:
                line.remove()
            self.network_lines = []
            
            # Draw connections between agents of the same type
            for cell_content, pos in self.model.grid.coord_iter():
                # In Mesa's MultiGrid, cell_content is a list of agents
                if cell_content:  # If the cell is not empty
                    for agent in cell_content:
                        if isinstance(agent, SchellingAgent):
                            x1, y1 = pos
                            neighbors = self.model.grid.get_neighbors(pos, moore=True, include_center=False)
                            
                            for neighbor in neighbors:
                                if isinstance(neighbor, SchellingAgent) and neighbor.agent_type == agent.agent_type:
                                    x2, y2 = neighbor.pos
                                    line = self.ax_grid.plot([x1, x2], [y1, y2], 'w-', alpha=0.3, linewidth=0.5)[0]
                                    self.network_lines.append(line)
            
            # Show network metrics
            connections = len(self.network_lines) // 2  # Each connection is counted twice
            max_connections = len(self.model.agents) * 8 // 2  # Max possible connections (8 neighbors per agent)
            connection_density = connections / max_connections if max_connections > 0 else 0
            
            status_text = f"Step: {len(self.x_data) - 1}\nConnections: {connections}\nDensity: {connection_density:.2f}"
            self.status_text.set_text(status_text)
            
        elif self.view_mode == 'happiness':
            # Create a happiness heatmap
            happiness_grid = np.zeros((self.model.height, self.model.width))
            
            for agent in self.model.agents:
                x, y = agent.pos
                happiness_score = agent.calculate_happiness_at(agent.pos)
                happiness_grid[y][x] = happiness_score
            
            self.im = self.ax_grid.imshow(happiness_grid, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
            
            # Add a colorbar
            if not hasattr(self, 'cbar'):
                divider = make_axes_locatable(self.ax_grid)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar = self.fig.colorbar(self.im, cax=cax)
                self.cbar.set_label('Happiness')
            else:
                self.cbar.update_normal(self.im)
        
        # Update metrics in the plot
        self._update_metrics_plot()
    
    def setup_plot(self):
        """Set up the animation"""
        self.ani = animation.FuncAnimation(
            self.fig, 
            self._animate,
            interval=50,  # milliseconds
            save_count=50  # Prevent warning about save_count
        )
    
    def _animate(self, i):
        """Animation function"""
        if self.is_running:
            self.step_model()
        return [self.im]
    
    def _update_metrics_plot(self):
        """Update the metrics plot"""
        # Get current metrics
        if len(self.happiness_data) < len(self.x_data):
            # Only collect metrics if they haven't been collected for this step
            happy_pct = (len(self.model.agents) - len(self.model.unhappy_agents)) / len(self.model.agents) if self.model.agents else 0
            
            # Get segregation data from datacollector
            if self.model.datacollector is not None and len(self.model.datacollector.model_vars['Segregation']) > 0:
                segregation = self.model.datacollector.model_vars['Segregation'][-1]
            else:
                segregation = 0
                
            self.happiness_data.append(happy_pct)
            self.segregation_data.append(segregation)
        
        # Update the plot
        self.happiness_line.set_data(self.x_data, self.happiness_data)
        self.segregation_line.set_data(self.x_data, self.segregation_data)
        
        # Adjust limits if needed
        self.ax_plot.set_xlim(0, max(len(self.x_data) + 1, 10))
        
        # Redraw the plot
        self.ax_plot.figure.canvas.draw_idle()
    
    def step_model(self, event=None):
        """Step the model forward once"""
        self.model.step()
        
        # Update data
        self.x_data.append(len(self.x_data))
        
        # Get metrics data from the model
        happy_pct = (len(self.model.agents) - len(self.model.unhappy_agents)) / len(self.model.agents) if self.model.agents else 0
        self.happiness_data.append(happy_pct)
        
        # Get segregation data from datacollector
        if self.model.datacollector is not None and len(self.model.datacollector.model_vars['Segregation']) > 0:
            segregation = self.model.datacollector.model_vars['Segregation'][-1]
        else:
            segregation = 0
        self.segregation_data.append(segregation)
        
        # Update visualization
        self.update_grid()
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def toggle_running(self, event=None):
        """Toggle the animation on/off"""
        self.is_running = not self.is_running
        self.run_button.label.set_text('Stop' if self.is_running else 'Run')
        self.fig.canvas.draw_idle()
    
    def reset_model(self, event=None):
        """Reset the model"""
        # Stop running if it's running
        if self.is_running:
            self.toggle_running()
        
        # Get current parameters
        width = self.model.width
        height = self.model.height
        density = self.model.density
        homophily = self.model.homophily
        num_agent_types = self.model.num_agent_types
        pattern = self.model.pattern if hasattr(self.model, 'pattern') else 'alternating'
        
        # Create a new model
        self.model = SchellingModel(
            width=width,
            height=height,
            density=density,
            homophily=homophily,
            num_agent_types=num_agent_types,
            pattern=pattern
        )
        
        # Reset data
        self.x_data = [0]
        self.happiness_data = []
        self.segregation_data = []
        
        # Update visualization
        self.update_grid()
        
        # Update UI
        self.run_button.label.set_text('Run')
        self.fig.canvas.draw_idle()
    
    def update_homophily(self, val):
        """Update the homophily parameter"""
        # Get slider value
        homophily = val
        
        # Update model
        self.model.homophily = homophily
        
        # Recalculate happiness for each agent
        for agent in self.model.agents:
            agent.step()
            
        # Update visualization
        self.update_grid()
    
    def update_agent_types(self, val):
        """Update the number of agent types"""
        # Only reset if the number of types has changed
        if int(val) != self.model.num_agent_types:
            # Create a new model with the new number of agent types
            self.model = SchellingModel(
                width=self.model.width,
                height=self.model.height,
                density=self.model.density,
                homophily=self.model.homophily,
                num_agent_types=int(val)
            )
            
            # Apply the current pattern initialization
            if self.current_pattern == 'alternating':
                self._initialize_alternating_pattern()
            elif self.current_pattern == 'clusters':
                self._initialize_cluster_pattern()
            elif self.current_pattern == 'stripes':
                self._initialize_stripe_pattern()
            
            # Reset data
            self.x_data = [0]
            self.happiness_data = []
            self.segregation_data = []
            
            # Update visualization
            self.update_grid()
            
            # Update UI
            self.fig.canvas.draw_idle()
    
    def change_pattern(self, label):
        """Change the initialization pattern"""
        pattern_map = {
            'Alternating': 'alternating',
            'Random': 'random',
            'Clusters': 'clusters',
            'Stripes': 'stripes'
        }
        
        new_pattern = pattern_map.get(label, 'alternating')
        
        # Only reset if pattern has changed
        if self.current_pattern != new_pattern:
            self.current_pattern = new_pattern
            
            # Create a new model with default parameters
            width = self.model.width
            height = self.model.height
            density = self.model.density  # Keep density at 1.0 for full grid
            homophily = self.model.homophily
            num_agent_types = self.model.num_agent_types
            
            # Create a new model instance
            self.model = SchellingModel(
                width=width,
                height=height,
                density=density,
                homophily=homophily,
                num_agent_types=num_agent_types
            )
            
            # Now manually implement the patterns by modifying the grid
            if new_pattern == 'alternating':
                self._initialize_alternating_pattern()
            elif new_pattern == 'random':
                # Random is already the default with SchellingModel
                pass
            elif new_pattern == 'clusters':
                self._initialize_cluster_pattern()
            elif new_pattern == 'stripes':
                self._initialize_stripe_pattern()
            
            # Reset data tracking
            self.x_data = [0]
            self.happiness_data = []
            self.segregation_data = []
            
            # Update visualization
            self.update_grid()
            
            # Update UI
            self.fig.canvas.draw_idle()
    
    def _initialize_alternating_pattern(self):
        """Initialize grid with alternating pattern (checkerboard)"""
        # First remove all agents from the model's scheduler and grid
        for agent in list(self.model.agents):
            self.model.grid.remove_agent(agent)
            self.model.schedule.remove(agent)
        
        # Create a new set of agents in a checkerboard pattern
        agent_id = 0
        for x in range(self.model.width):
            for y in range(self.model.height):
                # Determine agent type based on position (alternating)
                agent_type = (x + y) % self.model.num_agent_types
                agent = SchellingAgent(agent_id, self.model, agent_type)
                self.model.schedule.add(agent)
                self.model.grid.place_agent(agent, (x, y))
                agent_id += 1
        
        # Recalculate happiness for all agents
        for agent in self.model.agents:
            agent.step()
    
    def _initialize_cluster_pattern(self):
        """Initialize grid with clustered agents"""
        # First remove all agents from the model's scheduler and grid
        for agent in list(self.model.agents):
            self.model.grid.remove_agent(agent)
            self.model.schedule.remove(agent)
        
        # Create cluster centers
        num_clusters = self.model.num_agent_types * 2
        cluster_centers = []
        for _ in range(num_clusters):
            cx = self.model.random.randint(0, self.model.width - 1)
            cy = self.model.random.randint(0, self.model.height - 1)
            cluster_type = self.model.random.randint(0, self.model.num_agent_types - 1)
            cluster_centers.append((cx, cy, cluster_type))
        
        # Place agents with probability based on distance to nearest same-type cluster
        agent_id = 0
        for x in range(self.model.width):
            for y in range(self.model.height):
                # Calculate distances to all cluster centers
                distances = []
                for cx, cy, ctype in cluster_centers:
                    dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                    distances.append((dist, ctype))
                
                # Sort by distance
                distances.sort()
                
                # Assign type based on closest cluster
                agent_type = distances[0][1]  # Type of closest cluster center
                
                # Create and place agent
                agent = SchellingAgent(agent_id, self.model, agent_type)
                self.model.schedule.add(agent)
                self.model.grid.place_agent(agent, (x, y))
                agent_id += 1
        
        # Recalculate happiness for all agents
        for agent in self.model.agents:
            agent.step()
    
    def _initialize_stripe_pattern(self):
        """Initialize grid with horizontal stripes of different agent types"""
        # First remove all agents from the model's scheduler and grid
        for agent in list(self.model.agents):
            self.model.grid.remove_agent(agent)
            self.model.schedule.remove(agent)
        
        # Calculate stripe height (at least 2 cells per stripe)
        stripe_height = max(2, self.model.height // (self.model.num_agent_types * 2))
        
        # Create agents in horizontal stripes
        agent_id = 0
        for x in range(self.model.width):
            for y in range(self.model.height):
                # Determine agent type based on y-position (stripes)
                stripe_index = (y // stripe_height) % self.model.num_agent_types
                agent = SchellingAgent(agent_id, self.model, stripe_index)
                self.model.schedule.add(agent)
                self.model.grid.place_agent(agent, (x, y))
                agent_id += 1
        
        # Recalculate happiness for all agents
        for agent in self.model.agents:
            agent.step()
    
    def change_view_mode(self, label):
        """Change the visualization mode"""
        mode_map = {
            'Normal': 'normal',
            'Networks': 'networks',
            'Happiness': 'happiness'
        }
        
        self.view_mode = mode_map.get(label, 'normal')
        self.update_grid()
    
    def save_model(self, event=None):
        """Save the current model state to a file"""
        try:
            # Create saved_models directory if it doesn't exist
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            
            # Generate unique filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"saved_models/schelling_model_{timestamp}.pkl"
            
            # Save model data
            save_data = {
                'model': self.model,
                'happiness_data': self.happiness_data,
                'segregation_data': self.segregation_data,
                'x_data': self.x_data,
                'view_mode': self.view_mode
            }
            
            # Save to file
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Model saved to {filename}")
            
            # Update status text with save information
            self._update_status_text()
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, event=None):
        """Load a saved model state from a file"""
        try:
            # Check if saved_models directory exists
            if not os.path.exists('saved_models'):
                print("No saved models found.")
                return
            
            # Get list of saved model files
            saved_files = [f for f in os.listdir('saved_models') if f.startswith('schelling_model_') and f.endswith('.pkl')]
            
            if not saved_files:
                print("No saved models found.")
                return
            
            # Get most recent file
            latest_file = max(saved_files)
            filepath = os.path.join('saved_models', latest_file)
            
            # Load model data
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Stop animation if running
            if self.is_running:
                self.toggle_running()
            
            # Restore saved state
            self.model = save_data['model']
            self.happiness_data = save_data['happiness_data']
            self.segregation_data = save_data['segregation_data']
            self.x_data = save_data['x_data']
            self.view_mode = save_data.get('view_mode', 'normal')  # Default to normal if not saved
            
            # Update UI controls to match loaded model
            self.homophily_slider.set_val(self.model.homophily)
            self.types_slider.set_val(self.model.num_agent_types)
            
            # Update view mode selector
            view_labels = {'normal': 'Normal', 'networks': 'Networks', 'happiness': 'Happiness'}
            active_idx = list(view_labels.keys()).index(self.view_mode) if self.view_mode in view_labels else 0
            self.view_radio.set_active(active_idx)
            
            # Update pattern selector
            pattern_labels = {'alternating': 'Alternating', 'random': 'Random', 'clusters': 'Clusters', 'stripes': 'Stripes'}
            if hasattr(self.model, 'pattern') and self.model.pattern in pattern_labels:
                active_idx = list(pattern_labels.keys()).index(self.model.pattern)
                self.pattern_radio.set_active(active_idx)
            
            # Update visualization
            self.update_grid()
            self.setup_plot()
            
            print(f"Model loaded from {filepath}")
            
            # Update the figure
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def show(self):
        """Show the visualization"""
        plt.show()

# Run the visualization
if __name__ == "__main__":
    print("Starting Schelling Segregation Model with Improved Matplotlib Visualization...")
    viz = SchellingVisualization()
    viz.show()
