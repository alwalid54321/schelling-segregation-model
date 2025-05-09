"""
Schelling Segregation Model with Matplotlib Visualization
A simple and clean implementation that doesn't rely on Mesa's visualization components
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.lines as mlines
import random
from collections import defaultdict
import pickle
import os
import time

class Grid:
    """Simple grid implementation for the Schelling model"""
    def __init__(self, width, height, torus=True):
        self.width = width
        self.height = height
        self.torus = torus
        # Just use a simple dictionary with position tuples as keys and agents as values
        self.grid = {}
        
    def place_agent(self, agent, pos):
        """Place an agent at the specified position"""
        if pos in self.grid:
            # Replace any existing agent
            old_agent = self.grid[pos]
            old_agent.pos = None
        
        agent.pos = pos
        self.grid[pos] = agent
    
    def move_agent(self, agent, new_pos):
        """Move an agent from its current position to a new position"""
        old_pos = agent.pos
        if old_pos in self.grid and self.grid[old_pos] is agent:
            del self.grid[old_pos]
        
        self.place_agent(agent, new_pos)
    
    def swap_agents(self, agent1, agent2):
        """Swap the positions of two agents"""
        pos1 = agent1.pos
        pos2 = agent2.pos
        
        # Directly swap positions
        agent1.pos = pos2
        agent2.pos = pos1
        
        # Update grid
        self.grid[pos1] = agent2
        self.grid[pos2] = agent1
        
        return True
    
    def remove_agent(self, agent):
        """Remove agent from the grid"""
        if agent.pos in self.grid and self.grid[agent.pos] is agent:
            del self.grid[agent.pos]
            agent.pos = None
    
    def get_neighbors(self, pos, moore=True, include_center=False, radius=1):
        """Return a list of neighbors to a certain position"""
        x, y = pos
        neighbors = []
        
        # For Moore neighborhood (8 neighbors), use a square
        # For Von Neumann neighborhood (4 neighbors), use a diamond
        deltas = list(range(-radius, radius + 1))
        
        for dx in deltas:
            for dy in deltas:
                if not include_center and dx == 0 and dy == 0:
                    continue
                    
                # Skip corners for Von Neumann neighborhood
                if not moore and abs(dx) + abs(dy) > radius:
                    continue
                
                # Get neighbor coordinates with torus adjustment if needed
                nx, ny = x + dx, y + dy
                if self.torus:
                    nx = nx % self.width
                    ny = ny % self.height
                else:
                    # Skip if out of bounds
                    if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                        continue
                
                # Get the agent at those coordinates
                cell_pos = (nx, ny)
                if cell_pos in self.grid:
                    neighbors.append(self.grid[cell_pos])
        
        return neighbors
    
    def get_cell_list_contents(self, cell_list):
        """Return a list of agents in the given cells"""
        contents = []
        for pos in cell_list:
            if pos in self.grid:
                contents.append(self.grid[pos])
        return contents
    
    def get_all_cell_contents(self):
        """Return a list of all agents in the grid"""
        return list(self.grid.values())

class SchellingAgent:
    """Schelling segregation agent"""
    def __init__(self, unique_id, model, agent_type):
        self.unique_id = unique_id
        self.model = model
        self.type = agent_type
        self.pos = None
        self.is_happy = True

    def step(self):
        """Determine if agent is happy and move if necessary"""
        # Get the surrounding cells
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        
        # Count number of similar neighbors
        similar_neighbors = [neighbor for neighbor in neighbors if neighbor.type == self.type]
        similar = len(similar_neighbors)
        total = len(neighbors)
        
        # Evaluate how many similar neighbors are connected to each other
        connection_score = self.calculate_connected_neighbors(similar_neighbors)
        
        # Happiness now depends on both the ratio of similar neighbors and their connectivity
        # Agents are happier when similar neighbors are adjacent to each other
        happiness_threshold = 0.6  # Customize this threshold
        happiness_score = (similar / total if total > 0 else 0) + 0.2 * connection_score
        
        # If unhappy, try to swap with another agent
        if happiness_score < happiness_threshold:
            self.is_happy = False
            
            # Try to find a swap that improves happiness
            self.try_to_swap()
        else:
            self.is_happy = True
            
    def calculate_connected_neighbors(self, similar_neighbors):
        """Calculate how many similar neighbors are connected to each other"""
        # If fewer than 2 similar neighbors, there can't be connections
        if len(similar_neighbors) < 2:
            return 0
            
        # Check for pairs of similar neighbors that are adjacent to each other
        connected_pairs = 0
        
        # Consider each pair of similar neighbors
        for i in range(len(similar_neighbors)):
            for j in range(i+1, len(similar_neighbors)):
                # Calculate Manhattan distance between these neighbors
                n1_pos = similar_neighbors[i].pos
                n2_pos = similar_neighbors[j].pos
                
                # Check if they're directly adjacent (distance = 1 or 2)
                dx = abs(n1_pos[0] - n2_pos[0])
                dy = abs(n1_pos[1] - n2_pos[1])
                
                # In a Moore neighborhood, diagonal neighbors have dx=dy=1
                if (dx == 1 and dy == 0) or (dx == 0 and dy == 1) or (dx == 1 and dy == 1):
                    connected_pairs += 1
        
        # Normalize by the maximum possible connected pairs
        max_pairs = (len(similar_neighbors) * (len(similar_neighbors) - 1)) / 2
        return connected_pairs / max_pairs if max_pairs > 0 else 0
            
    def try_to_swap(self):
        """Try to swap positions with another agent to improve happiness"""
        # Get all positions in the grid (full coverage)
        all_positions = [(x, y) for x in range(self.model.width) for y in range(self.model.height)]
        # Shuffle positions to avoid bias
        random.shuffle(all_positions)
        
        # Try each position until finding one that improves happiness
        for pos in all_positions:
            # Skip our own position
            if pos == self.pos:
                continue
                
            # Get the agent at this position
            if pos in self.model.grid.grid:
                other_agent = self.model.grid.grid[pos]
            else:
                continue  # Should never happen with full grid
                
            # Calculate current happiness at current positions
            current_happiness_self = self.calculate_happiness_at(self.pos)
            current_happiness_other = other_agent.calculate_happiness_at(other_agent.pos)
            
            # Calculate happiness if swapped
            new_happiness_self = self.calculate_happiness_at(other_agent.pos)
            new_happiness_other = other_agent.calculate_happiness_at(self.pos)
            
            # If both agents would be happier (or at least one happier and one the same)
            # then swap
            if (new_happiness_self > current_happiness_self and new_happiness_other >= current_happiness_other) or \
               (new_happiness_self >= current_happiness_self and new_happiness_other > current_happiness_other):
                
                # Perform the swap
                self.model.grid.swap_agents(self, other_agent)
                return True
                
        return False  # Couldn't find a suitable swap
    
    def calculate_happiness_at(self, pos):
        """Calculate how happy this agent would be at the given position"""
        # Simulate being at the new position
        neighbors = self.model.grid.get_neighbors(pos, moore=True)
        
        # Count number of similar neighbors
        similar_neighbors = [neighbor for neighbor in neighbors if neighbor.type == self.type]
        similar = len(similar_neighbors)
        total = len(neighbors)
        
        # Evaluate how many similar neighbors are connected to each other
        connection_score = self.calculate_connected_neighbors(similar_neighbors)
        
        # Calculate happiness score using both ratio and connectivity
        # This should match the calculation in the step method
        if total > 0:
            happiness_score = (similar / total) + 0.2 * connection_score
            return happiness_score
        return 0

class SchellingModel:
    """Schelling segregation model"""
    def __init__(self, width=20, height=20, density=1.0, homophily=0.3, num_agent_types=2, pattern='alternating'):
        self.width = width
        self.height = height
        self.density = density
        self.homophily = homophily
        self.num_agent_types = num_agent_types
        self.pattern = pattern
        
        # Initialize the grid
        self.grid = Grid(width, height)
        
        # Initialize agents
        self.agents = []
        self.unhappy_agents = set()
        self.next_id = 0
        
        # Initialize datacollector for tracking metrics
        self.datacollector = {'happy': [], 'segregation': []}
        self.total_moves = 0
        
        # Choose initialization method based on pattern
        if pattern == 'alternating':
            self.initialize_alternating_pattern()
        elif pattern == 'random':
            self.initialize_random_pattern()
        elif pattern == 'clusters':
            self.initialize_cluster_pattern()
        elif pattern == 'stripes':
            self.initialize_stripe_pattern()
        else:  # Default to alternating
            self.initialize_alternating_pattern()
    
    def initialize_alternating_pattern(self):
        """Initialize the grid with alternating pattern of agent types (checkerboard)"""
        # Create agents for every cell
        self.agents = []
        self.unhappy_agents = set()
        self.agent_counts = [0] * self.num_agent_types
        
        # Alternating pattern (checkerboard style)
        for y in range(self.height):
            for x in range(self.width):
                # Determine agent type based on position (alternating)
                # For 2 agent types: 0, 1, 0, 1, ...)
                # For 3 agent types: 0, 1, 2, 0, 1, 2, ...)
                agent_type = (x + y) % self.num_agent_types
                
                # Create agent and add to scheduler
                agent = SchellingAgent(self.next_id, self, agent_type)
                self.next_id += 1
                self.agents.append(agent)
                self.agent_counts[agent_type] += 1
                
                # Place agent on grid
                self.grid.place_agent(agent, (x, y))
        
        # Calculate initial happiness for all agents
        self._calculate_initial_happiness()
    
    def initialize_random_pattern(self):
        """Initialize the grid with randomly distributed agent types"""
        # Create agents for every cell
        self.agents = []
        self.unhappy_agents = set()
        self.agent_counts = [0] * self.num_agent_types
        
        # Create position list and shuffle it for random placement
        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        random.shuffle(all_positions)
        
        num_agents = int(self.width * self.height * self.density)
        for i in range(num_agents):
            pos = all_positions[i]
            agent_type = random.randrange(self.num_agent_types)
            
            # Create agent and add to scheduler
            agent = SchellingAgent(self.next_id, self, agent_type)
            self.next_id += 1
            self.agents.append(agent)
            self.agent_counts[agent_type] += 1
            
            # Place agent on grid
            self.grid.place_agent(agent, pos)
        
        # Calculate initial happiness for all agents
        self._calculate_initial_happiness()
    
    def initialize_cluster_pattern(self):
        """Initialize the grid with pre-formed clusters of agents by type"""
        # Create agents for every cell
        self.agents = []
        self.unhappy_agents = set()
        self.agent_counts = [0] * self.num_agent_types
        
        # Calculate number of clusters per type
        clusters_per_type = 3
        cluster_radius = min(self.width, self.height) // 8
        
        # Generate cluster centers for each type
        cluster_centers = []
        for agent_type in range(self.num_agent_types):
            type_centers = []
            for _ in range(clusters_per_type):
                # Try to find centers with minimum distance from existing ones
                max_attempts = 10
                best_center = None
                max_min_distance = 0
                
                for _ in range(max_attempts):
                    center = (random.randrange(self.width), random.randrange(self.height))
                    if not cluster_centers:  # First center
                        best_center = center
                        break
                    
                    # Calculate minimum distance to existing centers
                    min_distance = min(abs(center[0] - c[0]) + abs(center[1] - c[1]) 
                                     for centers in cluster_centers for c in centers)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_center = center
                
                type_centers.append(best_center)
            cluster_centers.append(type_centers)
        
        # Fill the grid positions
        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        random.shuffle(all_positions)
        assigned_positions = set()
        
        # First assign positions around cluster centers based on type
        for agent_type, centers in enumerate(cluster_centers):
            for center in centers:
                cx, cy = center
                for dist in range(1, cluster_radius + 1):
                    for dx in range(-dist, dist + 1):
                        for dy in range(-dist, dist + 1):
                            # Skip if beyond this distance
                            if abs(dx) + abs(dy) > dist:
                                continue
                            
                            x, y = (cx + dx) % self.width, (cy + dy) % self.height
                            pos = (x, y)
                            
                            # Skip already assigned or beyond radius
                            if pos in assigned_positions:
                                continue
                            
                            # Probability decreases with distance
                            prob = 1 - (dist / cluster_radius)
                            if random.random() < prob:
                                agent = SchellingAgent(self.next_id, self, agent_type)
                                self.next_id += 1
                                self.agents.append(agent)
                                self.agent_counts[agent_type] += 1
                                
                                # Place agent on grid
                                self.grid.place_agent(agent, pos)
                                assigned_positions.add(pos)
        
        # Fill remaining positions randomly if density is high enough
        remaining_positions = [(x, y) for x in range(self.width) for y in range(self.height) 
                            if (x, y) not in assigned_positions]
        random.shuffle(remaining_positions)
        
        remaining_agents = int(self.width * self.height * self.density) - len(assigned_positions)
        remaining_agents = max(0, remaining_agents)  # Ensure non-negative
        
        for i in range(min(remaining_agents, len(remaining_positions))):
            pos = remaining_positions[i]
            agent_type = random.randrange(self.num_agent_types)
            
            agent = SchellingAgent(self.next_id, self, agent_type)
            self.next_id += 1
            self.agents.append(agent)
            self.agent_counts[agent_type] += 1
            
            # Place agent on grid
            self.grid.place_agent(agent, pos)
        
        # Calculate initial happiness for all agents
        self._calculate_initial_happiness()
    
    def initialize_stripe_pattern(self):
        """Initialize the grid with horizontal stripes of different agent types"""
        # Create agents for every cell
        self.agents = []
        self.unhappy_agents = set()
        self.agent_counts = [0] * self.num_agent_types
        
        # Determine stripe thickness based on grid height and number of types
        stripe_thickness = max(1, self.height // (self.num_agent_types * 2))
        
        # Create striped pattern
        for y in range(self.height):
            # Determine agent type based on y-coordinate
            agent_type = (y // stripe_thickness) % self.num_agent_types
            
            for x in range(self.width):
                # Skip some positions if density is less than 1.0
                if random.random() > self.density:
                    continue
                
                # Create agent
                agent = SchellingAgent(self.next_id, self, agent_type)
                self.next_id += 1
                self.agents.append(agent)
                self.agent_counts[agent_type] += 1
                
                # Place agent on grid
                self.grid.place_agent(agent, (x, y))
        
        # Calculate initial happiness for all agents
        self._calculate_initial_happiness()
    
    def _calculate_initial_happiness(self):
        """Helper method to calculate initial happiness for all agents"""
        for agent in self.agents:
            # Get the surrounding agents
            neighbors = self.grid.get_neighbors(agent.pos, moore=True)
            
            # Count number of similar neighbors
            similar_neighbors = [neighbor for neighbor in neighbors if neighbor.type == agent.type]
            similar = len(similar_neighbors)
            total = len(neighbors)
            
            # Evaluate how many similar neighbors are connected to each other
            connection_score = agent.calculate_connected_neighbors(similar_neighbors)
            
            # Calculate happiness score using both ratio and connectivity
            happiness_threshold = 0.6  # Same threshold as in step method
            happiness_score = (similar / total if total > 0 else 0) + 0.2 * connection_score
            
            # Set initial happiness
            if happiness_score >= happiness_threshold:
                agent.is_happy = True
            else:
                agent.is_happy = False
                self.unhappy_agents.add(agent)
        
        # Collect initial data
        self.collect_data()
        
    def step(self):
        """Advance the model by one step"""
        self.steps += 1
        
        # Shuffle agents to avoid order bias
        random.shuffle(self.agents)
        
        # Let each agent step
        for agent in self.agents:
            agent.step()
        
        # Collect data
        self.collect_data()
    
    def collect_data(self):
        """Collect data about the model state"""
        happy_agents = sum(1 for agent in self.agents if agent.is_happy)
        self.datacollector['happy'].append(happy_agents / len(self.agents) if self.agents else 0)
        self.datacollector['segregation'].append(self.calculate_segregation_index())
    
    def calculate_segregation_index(self):
        """Calculate a simple segregation index based on neighbor similarity"""
        total_agents = 0
        total_similarity = 0
        
        for agent in self.agents:
            neighbors = self.grid.get_neighbors(agent.pos, moore=True)
            if neighbors:
                similarity = sum(1 for neighbor in neighbors if neighbor.type == agent.type) / len(neighbors)
                total_similarity += similarity
                total_agents += 1
        
        return total_similarity / total_agents if total_agents > 0 else 0

# Custom visualization class
class SchellingVisualization:
    def __init__(self, width=20, height=20, density=1.0, homophily=0.3, num_agent_types=2):
        """Initialize visualization with grid and controls"""
        print("Starting Schelling Segregation Model with Matplotlib Visualization...")
        
        # Create figure with larger size to prevent UI elements from overlapping
        self.fig = plt.figure(figsize=(16, 12))
        
        # Set window title
        self.fig.canvas.manager.set_window_title('Schelling Segregation Model')
        
        # Create grid with proper boundaries - use less total space to allow more room for controls
        self.ax_grid = plt.subplot2grid((6, 6), (0, 0), rowspan=5, colspan=4)
        self.ax_grid.set_title('Schelling Grid')
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
{{ ... }}
        # Data plots for metrics
        self.ax_plot = plt.subplot2grid((5, 5), (4, 0), colspan=3)
        self.ax_plot.set_title('Model Metrics')
        self.ax_plot.set_xlabel('Steps')
        self.ax_plot.set_ylabel('Value')
        
        # Control panel on the right side
        # Pattern selection radio button
        self.pattern_ax = plt.subplot2grid((5, 5), (0, 3), colspan=2)
        self.pattern_ax.set_title('Pattern')
        self.pattern_ax.axis('off')
        self.pattern_radio = RadioButtons(
            self.pattern_ax,
            ('Alternating', 'Random', 'Clusters', 'Stripes'),
            active=0
        )
        self.pattern_radio.on_clicked(self.change_pattern)
        
        # View mode selection radio button
        self.view_ax = plt.subplot2grid((5, 5), (1, 3), colspan=2)
        self.view_ax.set_title('View Mode')
        self.view_ax.axis('off')
        self.view_radio = RadioButtons(
            self.view_ax,
            ('Normal', 'Networks', 'Happiness'),
            active=0
        )
        self.view_mode = 'normal'  # Default view mode
        self.view_radio.on_clicked(self.change_view_mode)
        
        # Parameter sliders
        self.homophily_ax = plt.subplot2grid((5, 5), (2, 3), colspan=2)
        self.homophily_ax.set_title('Homophily')
        self.homophily_slider = Slider(
            self.homophily_ax,
            'Threshold',
            0.0,
            1.0,
            valinit=homophily
        )
        self.homophily_slider.on_changed(self.update_homophily)
        
        self.types_ax = plt.subplot2grid((5, 5), (3, 3), colspan=2)
        self.types_ax.set_title('Agent Types')
        self.types_slider = Slider(
            self.types_ax,
            'Types',
            2,
            5,
            valinit=num_agent_types,
            valstep=1
        )
        self.types_slider.on_changed(self.update_agent_types)
        
        # Control buttons
        button_width = 0.35
        button_height = 0.04
        button_x = 0.62
        
        # Step and run buttons
        self.step_button_ax = plt.axes([button_x, 0.32, button_width, button_height])
        self.step_button = Button(self.step_button_ax, 'Step')
        self.step_button.on_clicked(self.step_model)
        
        self.run_button_ax = plt.axes([button_x, 0.26, button_width, button_height])
        self.run_button = Button(self.run_button_ax, 'Run')
        self.run_button.on_clicked(self.toggle_running)
        
        self.reset_button_ax = plt.axes([button_x, 0.20, button_width, button_height])
        self.reset_button = Button(self.reset_button_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_model)
        
        # Save/load buttons
        self.save_button_ax = plt.axes([button_x, 0.14, button_width/2-0.01, button_height])
        self.save_button = Button(self.save_button_ax, 'Save')
        self.save_button.on_clicked(self.save_model)
        
        self.load_button_ax = plt.axes([button_x + button_width/2 + 0.01, 0.14, button_width/2-0.01, button_height])
        self.load_button = Button(self.load_button_ax, 'Load')
        self.load_button.on_clicked(self.load_model)
        
        # Create the model with the specified parameters
        self.model = SchellingModel(
            width=width,
            height=height,
            density=density,
            homophily=homophily,
            num_agent_types=num_agent_types,
            pattern='alternating'  # Default pattern
        )
        
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
        
        # Adjust layout to prevent overlap
        self.fig.set_constrained_layout(True)
        
        # Use constrained_layout instead of tight_layout to avoid warnings
        self.fig.set_constrained_layout(True)
    
    def update_grid(self):
        """Update the grid visualization"""
        # Create a matrix to visualize
        grid_matrix = np.zeros((self.model.height, self.model.width))
        
        self.ax_grid.clear()
        self.ax_grid.set_title('Schelling Grid (Step: {})'.format(len(self.x_data) - 1))
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        # Clear previous network lines if any
        for line in self.network_lines:
            if line in self.ax_grid.lines:
                line.remove()
        self.network_lines = []
        
        # Different visualization modes
        if self.view_mode == 'normal':
            # Standard visualization - just show agent types
            for agent in self.model.agents:
                if agent.pos is not None:  # Make sure agent has a position
                    x, y = agent.pos
                    grid_matrix[y, x] = agent.type + 1  # Add 1 to avoid 0 (which would be invisible)
            
            # Create a discrete colormap
            cmap = plt.cm.get_cmap('tab10', self.model.num_agent_types + 1)
            
            # Display the grid
            self.ax_grid.imshow(grid_matrix, cmap=cmap, vmin=0, vmax=self.model.num_agent_types)
            
            # Add happiness markers (X for unhappy)
            for agent in self.model.agents:
                if agent.pos is not None and not agent.is_happy:
                    x, y = agent.pos
                    self.ax_grid.plot(x, y, 'x', color='white', markersize=4)
                    
        elif self.view_mode == 'networks':
            # Network visualization - show connections between similar agents
            for agent in self.model.agents:
                if agent.pos is not None:  # Make sure agent has a position
                    x, y = agent.pos
                    grid_matrix[y, x] = agent.type + 1  # Add 1 to avoid 0 (which would be invisible)
            
            # Create a discrete colormap
            cmap = plt.cm.get_cmap('tab10', self.model.num_agent_types + 1)
            
            # Display the grid
            self.ax_grid.imshow(grid_matrix, cmap=cmap, vmin=0, vmax=self.model.num_agent_types)
            
            # Add connections between similar neighboring agents
            for agent in self.model.agents:
                if agent.pos is not None:
                    # Get similar neighbors
                    neighbors = self.model.grid.get_neighbors(agent.pos, moore=True)
                    similar_neighbors = [n for n in neighbors if n.type == agent.type]
                    
                    # Draw connection lines to each similar neighbor
                    agent_x, agent_y = agent.pos
                    for neighbor in similar_neighbors:
                        neighbor_x, neighbor_y = neighbor.pos
                        # Only draw connections in one direction to avoid duplicates
                        if agent.unique_id < neighbor.unique_id:
                            line = self.ax_grid.plot(
                                [agent_x, neighbor_x], [agent_y, neighbor_y],
                                '-', color='white', alpha=0.3, linewidth=0.5
                            )[0]
                            self.network_lines.append(line)
                    
        elif self.view_mode == 'happiness':
            # Happiness visualization - color by happiness instead of type
            for agent in self.model.agents:
                if agent.pos is not None:
                    x, y = agent.pos
                    # 1 for happy, 0.5 for unhappy
                    grid_matrix[y, x] = 1.0 if agent.is_happy else 0.5
            
            # Use a different colormap for happiness
            cmap = plt.cm.get_cmap('RdYlGn', 3)  # Red for unhappy, green for happy
            
            # Display the grid
            self.ax_grid.imshow(grid_matrix, cmap=cmap, vmin=0, vmax=1)
            
            # Add type number labels inside cells
            for agent in self.model.agents:
                if agent.pos is not None:
                    x, y = agent.pos
                    self.ax_grid.text(x, y, str(agent.type), 
                               ha='center', va='center', color='black', fontsize=6)
        
        # Show stats
        self._update_status_text()
        
    def _update_status_text(self):
        """Update the status text with current metrics"""
        # Calculate current metrics
        happy_agents = sum(1 for agent in self.model.agents if agent.is_happy)
        total_agents = len(self.model.agents)
        happy_pct = happy_agents / total_agents if total_agents > 0 else 0
        segregation = self.model.calculate_segregation_index()
        
        # Create legend text
        type_names = {0: 'Red', 1: 'Blue', 2: 'Green', 3: 'Yellow', 4: 'Purple'}
        legend_text = "Agent Types:\n"
        for i in range(self.model.num_agent_types):
            if i in self.model.agent_counts:
                legend_text += f"{type_names[i]}: {self.model.agent_counts[i]}\n"
        
        # Create statistics text
        stats_text = f"Step: {len(self.x_data) - 1}\n"
        stats_text += f"Moves: {self.model.total_moves}\n"
        stats_text += f"Happy: {happy_pct:.1%} ({happy_agents}/{total_agents})\n"
        stats_text += f"Unhappy: {total_agents - happy_agents}\n"
        stats_text += f"Segregation: {segregation:.2f}\n\n"
        stats_text += f"Pattern: {self.model.pattern}\n"
        stats_text += f"View: {self.view_mode}\n\n"
        stats_text += legend_text
        
        # Update the status text
        self.status_text.set_text(stats_text)
    
    def setup_plot(self):
        """Set up the data plot"""
        self.ax_plot.clear()
        self.happiness_line, = self.ax_plot.plot(self.x_data, [0], 'g-', label='Happiness')
        self.segregation_line, = self.ax_plot.plot(self.x_data, [0], 'r-', label='Segregation')
        self.ax_plot.set_xlabel('Steps')
        self.ax_plot.set_ylabel('Value')
        self.ax_plot.set_ylim(0, 1)
        self.ax_plot.set_title('Model Metrics')
        self.ax_plot.legend()
    
    def update_plot(self):
        """Update the data plot with new values"""
        # Our datacollector is now a simple dictionary
        self.happiness_data = self.model.datacollector['happy']
        self.segregation_data = self.model.datacollector['segregation']
        self.x_data = list(range(len(self.happiness_data)))
        
        self.happiness_line.set_data(self.x_data, self.happiness_data)
        self.segregation_line.set_data(self.x_data, self.segregation_data)
        
        self.ax_plot.set_xlim(0, max(10, len(self.x_data)))
        
    def step_model(self, event=None):
        """Step the model forward once"""
        self.model.step()
        self.update_grid()
        self.update_plot()
        self.fig.canvas.draw_idle()
    
    def toggle_running(self, event=None):
        """Toggle between running and paused states"""
        self.is_running = not self.is_running
        if self.is_running:
            self.run_button.label.set_text('Pause')
            # Add save_count and cache_frame_data parameters to avoid warning
            self.ani = animation.FuncAnimation(
                self.fig, 
                lambda i: self.step_model(), 
                interval=200, 
                save_count=100,  # Limit to 100 saved frames
                cache_frame_data=False  # Don't cache frame data
            )
        else:
            self.run_button.label.set_text('Run')
            if self.ani is not None:
                self.ani.event_source.stop()
                self.ani = None
        self.fig.canvas.draw_idle()
    
    def reset_model(self, event=None):
        """Reset the model with the current settings"""
        if self.is_running:
            self.toggle_running()
        
        # Get current parameters from UI controls
        homophily = self.homophily_slider.val
        num_agent_types = int(self.types_slider.val)
        pattern = self.model.pattern  # Keep the current pattern
        
        # Create new model with the same parameters
        self.model = SchellingModel(
            self.model.width, 
            self.model.height, 
            1.0,  # Density always 1.0 for full grid 
            homophily, 
            num_agent_types,
            pattern=pattern
        )
        
        # Reset plot data
        self.x_data = [0]
        self.happiness_data = []
        self.segregation_data = []
        
        # Update visuals
        self.update_grid()
        self.setup_plot()
        self.fig.canvas.draw_idle()
    
    def change_pattern(self, label):
        """Change the model initialization pattern"""
        # Only allow changes when not running
        if self.is_running:
            return
        
        # Convert radio button label to pattern name
        pattern_map = {
            'Alternating': 'alternating',
            'Random': 'random',
            'Clusters': 'clusters',
            'Stripes': 'stripes'
        }
        
        pattern = pattern_map.get(label, 'alternating')
        
        # Create new model with the selected pattern
        self.model = SchellingModel(
            self.model.width, 
            self.model.height, 
            1.0,  # Density always 1.0 for full grid
            self.homophily_slider.val, 
            int(self.types_slider.val),
            pattern=pattern
        )
        
        # Reset plot data
        self.x_data = [0]
        self.happiness_data = []
        self.segregation_data = []
        
        # Update visuals
        self.update_grid()
        self.setup_plot()
        self.fig.canvas.draw_idle()
    
    def change_view_mode(self, label):
        """Change the visualization mode"""
        # Convert radio button label to view mode
        view_map = {
            'Normal': 'normal',
            'Networks': 'networks',
            'Happiness': 'happiness'
        }
        
        self.view_mode = view_map.get(label, 'normal')
        
        # Update grid with new view mode
        self.update_grid()
        self.fig.canvas.draw_idle()
    
    def update_homophily(self, val):
        """Update the homophily threshold"""
        # Only update when not running
        if not self.is_running:
            self.model.homophily = val
            # Recalculate agent happiness
            for agent in self.model.agents:
                agent.step()
            self.update_grid()
            self.fig.canvas.draw_idle()
    
    def update_agent_types(self, val):
        """Update the number of agent types - requires model reset"""
        # Only update when not running
        if not self.is_running:
            # Create new model with updated number of agent types
            self.model = SchellingModel(
                self.model.width, 
                self.model.height, 
                1.0,  # Density always 1.0 for full grid
                self.homophily_slider.val, 
                int(val),
                pattern=self.model.pattern
            )
            
            # Reset plot data
            self.x_data = [0]
            self.happiness_data = []
            self.segregation_data = []
            
            # Update visuals
            self.update_grid()
            self.setup_plot()
            self.fig.canvas.draw_idle()
    
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
    print("Starting Schelling Segregation Model with Matplotlib Visualization...")
    viz = SchellingVisualization()
    viz.show()
