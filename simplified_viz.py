#!/usr/bin/env python
# simplified_viz.py - A minimal and robust visualization for Schelling model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.colors import LinearSegmentedColormap
import os
import pickle
import time

from model import SchellingModel
from agent import SchellingAgent


class SchellingVisualization:
    """Minimal visualization for the Schelling Segregation Model"""
    
    def __init__(self, width=20, height=20, density=1.0, homophily=0.3, num_agent_types=2):
        """Initialize the visualization"""
        # Initialize parameters
        self.width = width
        self.height = height
        self.density = density
        self.homophily = homophily
        self.num_agent_types = num_agent_types
        self.current_pattern = 'random'  # Default pattern
        self.view_mode = 'normal'
        self.is_running = False
        self.steps = 0
        self.happiness_data = []
        self.segregation_data = []
        
        # Create the model
        self.model = SchellingModel(
            width=width,
            height=height,
            density=density,
            homophily=homophily,
            num_agent_types=num_agent_types
        )
        
        # Set up plot
        self.setup_figure()
        
        # Initialize grid display
        self.update_grid()
        
        plt.show()
    
    def setup_figure(self):
        """Set up the figure and axes for visualization"""
        # Create figure with 2 rows and 2 columns
        self.fig = plt.figure(figsize=(12, 8))
        
        # Main grid display
        self.ax_grid = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
        self.ax_grid.set_title('Schelling Segregation Model')
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        # Metrics plot
        self.ax_plot = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
        self.ax_plot.set_title('Metrics Over Time')
        self.ax_plot.set_xlabel('Steps')
        self.ax_plot.set_ylabel('Metric Value')
        self.ax_plot.set_xlim(0, 10)
        self.ax_plot.set_ylim(0, 1)
        
        # Initialize metrics lines with empty data
        self.happiness_line, = self.ax_plot.plot([], [], 'g-', label='Happiness')
        self.segregation_line, = self.ax_plot.plot([], [], 'r-', label='Segregation')
        self.ax_plot.legend()
        
        # Status display
        self.ax_status = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0.05, 0.95, "", transform=self.ax_status.transAxes, 
                                              verticalalignment='top')
        
        # Controls area
        self.ax_controls = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
        self.ax_controls.axis('off')
        
        self.setup_controls()
        
        plt.tight_layout(pad=3.0)
    
    def setup_controls(self):
        """Set up the UI controls"""
        # Button positions
        button_width = 0.65
        button_height = 0.08
        button_left = 0.2
        
        # Step button
        step_button_pos = [button_left, 0.8, button_width, button_height]
        self.ax_step = plt.axes(step_button_pos)
        self.step_button = Button(self.ax_step, 'Step')
        self.step_button.on_clicked(self.step_model)
        
        # Run/Stop button
        run_button_pos = [button_left, 0.7, button_width, button_height]
        self.ax_run = plt.axes(run_button_pos)
        self.run_button = Button(self.ax_run, 'Run')
        self.run_button.on_clicked(self.toggle_running)
        
        # Reset button
        reset_button_pos = [button_left, 0.6, button_width, button_height]
        self.ax_reset = plt.axes(reset_button_pos)
        self.reset_button = Button(self.ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset_model)
        
        # Pattern selection
        pattern_radios_pos = [button_left, 0.35, button_width, 0.15]
        self.ax_pattern = plt.axes(pattern_radios_pos)
        self.pattern_radio = RadioButtons(
            self.ax_pattern,
            ['Random', 'Alternating', 'Clusters', 'Stripes'],
            active=0
        )
        self.pattern_radio.on_clicked(self.change_pattern)
        
        # View mode selection
        view_radios_pos = [button_left, 0.1, button_width, 0.15]
        self.ax_view = plt.axes(view_radios_pos)
        self.view_radio = RadioButtons(
            self.ax_view,
            ['Normal', 'Network', 'Happiness'],
            active=0
        )
        self.view_radio.on_clicked(self.change_view_mode)
        
        # Homophily slider
        homophily_slider_pos = [0.2, 0.53, 0.65, 0.03]
        self.ax_homophily = plt.axes(homophily_slider_pos)
        self.homophily_slider = Slider(
            self.ax_homophily,
            'Homophily',
            0.0, 1.0,
            valinit=self.homophily
        )
        self.homophily_slider.on_changed(self.update_homophily)
        
        # Agent types slider
        agent_types_slider_pos = [0.2, 0.45, 0.65, 0.03]
        self.ax_agent_types = plt.axes(agent_types_slider_pos)
        self.agent_types_slider = Slider(
            self.ax_agent_types,
            'Agent Types',
            2, 5,
            valinit=self.num_agent_types,
            valstep=1
        )
        self.agent_types_slider.on_changed(self.update_agent_types)
    
    def update_grid(self):
        """Update the grid visualization"""
        # Initialize grid matrix
        grid_matrix = np.zeros((self.model.height, self.model.width))
        
        # Clear the axes
        self.ax_grid.clear()
        self.ax_grid.set_title('Schelling Segregation Model')
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
        
        # Choose appropriate colormap
        if self.num_agent_types <= 10:
            try:
                # For newer matplotlib versions
                cmap = plt.colormaps['tab10']
            except (AttributeError, KeyError):
                # For older matplotlib versions
                cmap = plt.cm.get_cmap('tab10')
        else:
            # Create a colormap for more agent types
            cmap = plt.cm.get_cmap('viridis', self.num_agent_types + 1)
        
        # Display the grid based on view mode
        if self.view_mode == 'normal':
            # Standard grid view
            self.im = self.ax_grid.imshow(grid_matrix, cmap=cmap, interpolation='nearest')
            
            # Update status text with metrics
            happy_pct = self.calculate_happiness_percentage()
            segregation = self.get_segregation_value()
            
            status_text = (f"Step: {self.steps}\n"
                          f"Pattern: {self.current_pattern}\n"
                          f"Happy: {happy_pct:.2%}\n"
                          f"Segregation: {segregation:.3f}\n"
                          f"Homophily: {self.model.homophily:.2f}\n"
                          f"Agent Types: {self.model.num_agent_types}")
            
            self.status_text.set_text(status_text)
            
        elif self.view_mode == 'network':
            # Network view - connections between similar agents
            self.im = self.ax_grid.imshow(grid_matrix, cmap=cmap, interpolation='nearest', alpha=0.7)
            
            # Draw connections between agents of the same type
            connection_count = 0
            
            for cell_content, pos in self.model.grid.coord_iter():
                if not cell_content:
                    continue
                    
                for agent in cell_content:
                    if not isinstance(agent, SchellingAgent):
                        continue
                        
                    x1, y1 = pos
                    neighbors = self.model.grid.get_neighbors(pos, moore=True, include_center=False)
                    
                    for neighbor in neighbors:
                        if (isinstance(neighbor, SchellingAgent) and 
                            neighbor.agent_type == agent.agent_type):
                            x2, y2 = neighbor.pos
                            self.ax_grid.plot([x1, x2], [y1, y2], 'w-', alpha=0.3, linewidth=0.5)
                            connection_count += 1
            
            # Show network metrics
            happy_pct = self.calculate_happiness_percentage()
            segregation = self.get_segregation_value()
            
            status_text = (f"Step: {self.steps}\n"
                          f"Pattern: {self.current_pattern}\n"
                          f"Connections: {connection_count//2}\n"  # Divide by 2 as each is counted twice
                          f"Happy: {happy_pct:.2%}\n"
                          f"Segregation: {segregation:.3f}")
            
            self.status_text.set_text(status_text)
            
        elif self.view_mode == 'happiness':
            # Create a happiness matrix
            happiness_matrix = np.zeros((self.model.height, self.model.width))
            
            for cell_content, pos in self.model.grid.coord_iter():
                if not cell_content:
                    continue
                    
                for agent in cell_content:
                    if isinstance(agent, SchellingAgent):
                        x, y = pos
                        # Set cell value based on agent happiness
                        happiness_matrix[y][x] = 1 if agent.is_happy else 0
            
            # Use a red-green colormap for happiness
            happiness_cmap = LinearSegmentedColormap.from_list(
                'happiness_cmap', [(1, 0, 0), (0, 1, 0)]
            )
            
            self.im = self.ax_grid.imshow(happiness_matrix, cmap=happiness_cmap, 
                                         interpolation='nearest', vmin=0, vmax=1)
            
            # Show happiness metrics
            happy_count = sum(1 for agent in self.model.agents if agent.is_happy)
            total_agents = len(self.model.agents)
            happy_pct = happy_count / total_agents if total_agents > 0 else 0
            
            status_text = (f"Step: {self.steps}\n"
                          f"Pattern: {self.current_pattern}\n"
                          f"Happy Agents: {happy_count}/{total_agents}\n"
                          f"Happy: {happy_pct:.2%}\n"
                          f"Red = Unhappy, Green = Happy")
            
            self.status_text.set_text(status_text)
        
        # Update the metrics plot
        self.update_metrics_plot()
        
        # Draw the figure
        plt.draw()
    
    def update_metrics_plot(self):
        """Update the metrics plot with current data"""
        x_data = list(range(len(self.happiness_data)))
        
        # Make sure data arrays are the same length
        if len(x_data) > 0:
            # Update the plot data - ensuring the arrays are the same length
            self.happiness_line.set_data(x_data, self.happiness_data)
            self.segregation_line.set_data(x_data, self.segregation_data)
            
            # Adjust x-axis to fit all data
            max_x = max(10, len(x_data))
            self.ax_plot.set_xlim(0, max_x)
            
            # If we have data, adjust y-axis to fit
            if self.happiness_data and self.segregation_data:
                max_y = max(max(self.happiness_data), max(self.segregation_data), 1)
                min_y = min(min(self.happiness_data), min(self.segregation_data), 0)
                padding = (max_y - min_y) * 0.1
                self.ax_plot.set_ylim(max(0, min_y - padding), min(1, max_y + padding))
    
    def step_model(self, event=None):
        """Step the model forward one step"""
        # Run a step of the model
        self.model.step()
        self.steps += 1
        
        # Calculate current metrics
        happy_pct = self.calculate_happiness_percentage()
        segregation = self.get_segregation_value()
        
        # Store metrics for plotting
        self.happiness_data.append(happy_pct)
        self.segregation_data.append(segregation)
        
        # Update the visualization
        self.update_grid()
    
    def toggle_running(self, event=None):
        """Toggle animation running state"""
        self.is_running = not self.is_running
        self.run_button.label.set_text('Stop' if self.is_running else 'Run')
        
        if self.is_running:
            # If running, start a repeated timer for stepping
            self.timer = self.fig.canvas.new_timer(interval=100)
            self.timer.add_callback(self.step_model)
            self.timer.start()
        else:
            # If stopped, cancel the timer
            if hasattr(self, 'timer'):
                self.timer.stop()
    
    def reset_model(self, event=None):
        """Reset the model to initial state"""
        # Stop running if it's running
        if self.is_running:
            self.toggle_running()
        
        # Create a new model with current parameters
        self.model = SchellingModel(
            width=self.width,
            height=self.height,
            density=self.density,
            homophily=self.homophily,
            num_agent_types=self.num_agent_types
        )
        
        # Apply the current pattern initialization
        self.apply_pattern(self.current_pattern)
        
        # Reset data
        self.steps = 0
        self.happiness_data = []
        self.segregation_data = []
        
        # Update the visualization
        self.update_grid()
    
    def update_homophily(self, val):
        """Update homophily parameter"""
        self.homophily = val
        self.model.homophily = val
        
        # Recalculate happiness for all agents
        for agent in self.model.agents:
            agent.step()
        
        # Update the visualization
        self.update_grid()
    
    def update_agent_types(self, val):
        """Update number of agent types"""
        new_num_types = int(val)
        if new_num_types != self.num_agent_types:
            self.num_agent_types = new_num_types
            
            # Create a new model with the updated number of agent types
            self.model = SchellingModel(
                width=self.width,
                height=self.height,
                density=self.density,
                homophily=self.homophily,
                num_agent_types=new_num_types
            )
            
            # Apply the current pattern
            self.apply_pattern(self.current_pattern)
            
            # Reset data
            self.steps = 0
            self.happiness_data = []
            self.segregation_data = []
            
            # Update visualization
            self.update_grid()
    
    def change_pattern(self, label):
        """Change the initialization pattern"""
        pattern_map = {
            'Random': 'random',
            'Alternating': 'alternating',
            'Clusters': 'clusters',
            'Stripes': 'stripes'
        }
        
        pattern = pattern_map.get(label, 'random')
        
        if pattern != self.current_pattern:
            self.current_pattern = pattern
            
            # Create a new model with current parameters
            self.model = SchellingModel(
                width=self.width,
                height=self.height,
                density=self.density,
                homophily=self.homophily,
                num_agent_types=self.num_agent_types
            )
            
            # Apply the selected pattern
            self.apply_pattern(pattern)
            
            # Reset data
            self.steps = 0
            self.happiness_data = []
            self.segregation_data = []
            
            # Update visualization
            self.update_grid()
    
    def apply_pattern(self, pattern):
        """Apply a specific initialization pattern"""
        if pattern == 'alternating':
            self.initialize_alternating_pattern()
        elif pattern == 'clusters':
            self.initialize_cluster_pattern()
        elif pattern == 'stripes':
            self.initialize_stripe_pattern()
        # Random is already the default from the model's _populate_grid method
    
    def initialize_alternating_pattern(self):
        """Initialize grid with alternating pattern (checkerboard)"""
        # Clear the existing grid and scheduler
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
    
    def initialize_cluster_pattern(self):
        """Initialize grid with clustered agents"""
        # Clear the existing grid and scheduler
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
    
    def initialize_stripe_pattern(self):
        """Initialize grid with horizontal stripes of different agent types"""
        # Clear the existing grid and scheduler
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
            'Network': 'network',
            'Happiness': 'happiness'
        }
        
        mode = mode_map.get(label, 'normal')
        if mode != self.view_mode:
            self.view_mode = mode
            self.update_grid()
    
    def calculate_happiness_percentage(self):
        """Calculate the percentage of happy agents"""
        if not self.model.agents:
            return 0
        happy_count = sum(1 for agent in self.model.agents if agent.is_happy)
        return happy_count / len(self.model.agents)
    
    def get_segregation_value(self):
        """Get the current segregation value from the model's datacollector"""
        if (self.model.datacollector is not None and 
            'Segregation' in self.model.datacollector.model_vars and 
            self.model.datacollector.model_vars['Segregation']):
            return self.model.datacollector.model_vars['Segregation'][-1]
        return 0


if __name__ == "__main__":
    print("Starting Schelling Segregation Model with Simplified Visualization...")
    viz = SchellingVisualization()
