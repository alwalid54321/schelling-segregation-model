"""
Schelling Segregation Model with Animation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import time  # Importing regular time module for animations
import random
import argparse
from matplotlib.widgets import Slider, Button

class SchellingModel:
    """
    A professional implementation of the Schelling segregation model with animation.
    """
    
    def __init__(self, width=50, height=50, density=0.8, homophily=0.3, num_agent_types=2, 
                 pattern_type="alternating", animation_interval=200, shape="square"):
        self.width = width
        self.height = height
        self.density = density
        self.homophily = homophily
        self.num_agent_types = num_agent_types
        self.pattern_type = pattern_type
        self.animation_interval = animation_interval
        self.shape = shape
        
        # Create shape mask
        self.mask = self._create_shape_mask()
        
        # Create the grid with None for empty cells
        self.grid = [[None for _ in range(height)] for _ in range(width)]
        
        # Animation state - start paused
        self.is_running = False
        
        # Metrics tracking
        self.unhappy_agents = []
        self.steps = 0
        self.total_moves = 0
        self.segregation_data = []
        self.happiness_data = []
        self.unhappy_count_data = []
        
        # Animation state
        self.is_running = False
        self.fig = None
        self.ax_grid = None
        self.ax_happiness = None
        self.ax_segregation = None
        self.im = None
        self.happiness_line = None
        self.segregation_line = None
        
        # Initialize the grid based on the pattern type
        self.initialize_grid()
        
        # Make all agents initially unhappy
        self._make_all_agents_unhappy()
        
        # Collect initial metrics
        self.collect_data()
    
    def _create_shape_mask(self):
        """Create a mask for the grid based on the specified shape"""
        mask = np.ones((self.width, self.height), dtype=bool)
        
        if self.shape == "square":
            # Square is default (all cells available)
            pass
        
        elif self.shape == "circle":
            # Create a circular mask
            center_x, center_y = self.width // 2, self.height // 2
            radius = min(center_x, center_y) - 2  # Leave a small border
            
            for x in range(self.width):
                for y in range(self.height):
                    # If outside the circle, mask it out
                    if (x - center_x)**2 + (y - center_y)**2 > radius**2:
                        mask[x, y] = False
        
        elif self.shape == "human":
            # Create a simple human-like shape (head and body)
            center_x, center_y = self.width // 2, self.height // 2
            head_radius = min(center_x, center_y) // 3
            head_center_y = center_y - head_radius
            
            # Start with all masked out
            mask.fill(False)
            
            # Create head (circle)
            for x in range(self.width):
                for y in range(self.height):
                    # If inside the head circle
                    if (x - center_x)**2 + (y - head_center_y)**2 <= head_radius**2:
                        mask[x, y] = True
            
            # Create body (rectangle)
            body_width = head_radius * 2
            body_height = self.height // 2
            body_top = head_center_y + head_radius
            body_left = center_x - body_width // 2
            
            for x in range(body_left, body_left + body_width):
                for y in range(body_top, body_top + body_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
            
            # Create arms (rectangles)
            arm_width = self.width // 4
            arm_height = head_radius
            arm_top = body_top + head_radius
            
            # Left arm
            for x in range(body_left - arm_width, body_left):
                for y in range(arm_top, arm_top + arm_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
            
            # Right arm
            for x in range(body_left + body_width, body_left + body_width + arm_width):
                for y in range(arm_top, arm_top + arm_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
            
            # Create legs (rectangles)
            leg_width = head_radius
            leg_height = self.height // 4
            leg_top = body_top + body_height
            
            # Left leg
            left_leg_left = center_x - leg_width - leg_width // 2
            for x in range(left_leg_left, left_leg_left + leg_width):
                for y in range(leg_top, leg_top + leg_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
            
            # Right leg
            right_leg_left = center_x + leg_width // 2
            for x in range(right_leg_left, right_leg_left + leg_width):
                for y in range(leg_top, leg_top + leg_height):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        mask[x, y] = True
        
        return mask
    
    def _make_all_agents_unhappy(self):
        """Make all agents initially unhappy"""
        self.unhappy_agents = []
        for x in range(self.width):
            for y in range(self.height):
                agent = self.grid[x][y]
                if agent is not None:
                    agent['is_happy'] = False
                    self.unhappy_agents.append((agent, x, y))
    
    def initialize_grid(self):
        """
        Initialize the grid based on the specified pattern.
        """
        if self.pattern_type == "random":
            self._initialize_random()
        elif self.pattern_type == "checkerboard":
            self._initialize_checkerboard()
        elif self.pattern_type == "stripes":
            self._initialize_stripes()
        elif self.pattern_type == "clusters":
            self._initialize_clusters()
        elif self.pattern_type == "alternating":
            self._initialize_alternating()
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")
    
    def _initialize_random(self):
        """Standard random initialization"""
        agent_id = 0
        
        for x in range(self.width):
            for y in range(self.height):
                # Only place agents where the mask allows
                if self.mask[x, y] and random.random() < self.density:
                    # Create agent
                    agent_type = random.randint(0, self.num_agent_types - 1)
                    agent = {
                        'id': agent_id,
                        'type': agent_type,
                        'x': x,
                        'y': y,
                        'is_happy': False,  # Start unhappy
                        'moves': 0
                    }
                    agent_id += 1
                    
                    # Place agent in grid
                    self.grid[x][y] = agent
    
    def _initialize_checkerboard(self):
        """Initialize in a checkerboard pattern"""
        agent_id = 0
        
        for x in range(self.width):
            for y in range(self.height):
                # Determine if this cell should be filled based on density
                if random.random() < self.density:
                    # Alternate agent types in a checkerboard pattern
                    # Use modulo to determine the type based on position
                    agent_type = (x + y) % self.num_agent_types
                    
                    agent = {
                        'id': agent_id,
                        'type': agent_type,
                        'x': x,
                        'y': y,
                        'is_happy': True,
                        'moves': 0
                    }
                    agent_id += 1
                    
                    # Place agent in grid
                    self.grid[x][y] = agent
    
    def _initialize_stripes(self):
        """Initialize in horizontal stripes"""
        agent_id = 0
        
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < self.density:
                    # Create stripes based on y coordinate
                    stripe_width = max(1, self.height // (self.num_agent_types * 2))
                    agent_type = (y // stripe_width) % self.num_agent_types
                    
                    agent = {
                        'id': agent_id,
                        'type': agent_type,
                        'x': x,
                        'y': y,
                        'is_happy': True,
                        'moves': 0
                    }
                    agent_id += 1
                    
                    # Place agent in grid
                    self.grid[x][y] = agent
    
    def _initialize_clusters(self):
        """Initialize with initial clusters of agent types"""
        agent_id = 0
        # Create cluster centers
        centers = []
        for i in range(self.num_agent_types):
            # Create multiple centers for each type
            for _ in range(3):  # 3 clusters per type
                centers.append({
                    'x': random.randint(0, self.width - 1),
                    'y': random.randint(0, self.height - 1),
                    'type': i
                })
        
        # Assign agents to closest cluster center of their type
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < self.density:
                    # Find the closest center and use its type
                    min_dist = float('inf')
                    chosen_type = 0
                    
                    for center in centers:
                        dist = (x - center['x'])**2 + (y - center['y'])**2
                        if dist < min_dist:
                            min_dist = dist
                            chosen_type = center['type']
                    
                    agent = {
                        'id': agent_id,
                        'type': chosen_type,
                        'x': x,
                        'y': y,
                        'is_happy': True,
                        'moves': 0
                    }
                    agent_id += 1
                    
                    # Place agent in grid
                    self.grid[x][y] = agent
                    
    def _initialize_alternating(self):
        """Initialize with a strict alternating pattern (one red, one blue, etc.)"""
        agent_id = 0
        current_type = 0
        
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < self.density:
                    # Strictly alternate agent types
                    agent = {
                        'id': agent_id,
                        'type': current_type,
                        'x': x,
                        'y': y,
                        'is_happy': True,
                        'moves': 0
                    }
                    agent_id += 1
                    
                    # Place agent in grid
                    self.grid[x][y] = agent
                    
                    # Switch to next type for perfect alternation
                    current_type = (current_type + 1) % self.num_agent_types
    
    def get_neighbors(self, x, y, radius=1):
        """
        Get the neighbors of a cell within the specified radius
        """
        neighbors = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip the center cell
                
                nx = (x + dx) % self.width  # Wrap around the grid
                ny = (y + dy) % self.height
                
                if self.grid[nx][ny] is not None:
                    neighbors.append(self.grid[nx][ny])
        
        return neighbors
    
    def step(self):
        """
        Run one step of the model
        """
        self.unhappy_agents = []
        
        # Check happiness of all agents
        for x in range(self.width):
            for y in range(self.height):
                agent = self.grid[x][y]
                if agent is not None:
                    self.check_happiness(agent, x, y)
        
        # Move unhappy agents
        self.move_unhappy_agents()
        
        # Collect metrics
        self.collect_data()
        
        self.steps += 1
        
        # Return the grid for animation
        return self.get_grid_values()
    
    def check_happiness(self, agent, x, y):
        """
        Check if an agent is happy with its current location
        """
        neighbors = self.get_neighbors(x, y)
        
        if not neighbors:
            return  # No neighbors, agent is automatically happy
        
        # Count similar neighbors
        similar = sum(1 for n in neighbors if n['type'] == agent['type'])
        total = len(neighbors)
        
        # Calculate happiness
        happiness_score = similar / total if total > 0 else 0
        
        # Check if agent wants to move
        if happiness_score < self.homophily:
            agent['is_happy'] = False
            self.unhappy_agents.append((agent, x, y))
        else:
            agent['is_happy'] = True
    
    def move_unhappy_agents(self):
        """
        Move unhappy agents to random empty cells using a smarter strategy
        to maximize happiness
        """
        # Find empty cells
        empty_cells = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x][y] is None:
                    empty_cells.append((x, y))
        
        # Sort unhappy agents by how unhappy they are (those with fewest similar neighbors first)
        sorted_unhappy = []
        for agent, x, y in self.unhappy_agents:
            neighbors = self.get_neighbors(x, y)
            if not neighbors:
                continue
            similar = sum(1 for n in neighbors if n['type'] == agent['type'])
            total = len(neighbors)
            happiness_score = similar / total if total > 0 else 0
            sorted_unhappy.append((agent, x, y, happiness_score))
        
        # Sort by happiness score (least happy first)
        sorted_unhappy.sort(key=lambda item: item[3])
        
        # For each unhappy agent, try to find the best position
        for agent, x, y, _ in sorted_unhappy:
            if not empty_cells:
                break
                
            # Try to find a position that would make the agent happy
            best_position = None
            best_score = -1
            
            # Check a sample of positions (checking all would be too slow)
            sample_size = min(10, len(empty_cells))
            sample = random.sample(empty_cells, sample_size)
            
            for new_x, new_y in sample:
                # Temporarily place agent at new position
                self.grid[x][y] = None
                self.grid[new_x][new_y] = agent
                
                # Calculate happiness at new position
                neighbors = self.get_neighbors(new_x, new_y)
                if neighbors:
                    similar = sum(1 for n in neighbors if n['type'] == agent['type'])
                    total = len(neighbors)
                    happiness_score = similar / total if total > 0 else 0
                    
                    # If this position makes the agent happy, choose it
                    if happiness_score >= self.homophily and happiness_score > best_score:
                        best_score = happiness_score
                        best_position = (new_x, new_y)
                
                # Move agent back to original position for now
                self.grid[new_x][new_y] = None
                self.grid[x][y] = agent
            
            # If found a good position, move there
            if best_position:
                new_x, new_y = best_position
                empty_cells.remove(best_position)
            # Otherwise just pick a random position
            elif empty_cells:
                new_x, new_y = empty_cells.pop()
            else:
                continue  # No empty cells left
            
            # Update agent position
            self.grid[x][y] = None
            self.grid[new_x][new_y] = agent
            agent['x'] = new_x
            agent['y'] = new_y
            agent['moves'] += 1
            self.total_moves += 1
            
            # Add the old position to empty cells
            empty_cells.append((x, y))
    
    def collect_data(self):
        """
        Collect data for analysis
        """
        # Calculate happiness percentage
        total_agents = sum(1 for x in range(self.width) for y in range(self.height) if self.grid[x][y] is not None)
        happy_agents = total_agents - len(self.unhappy_agents)
        happiness = happy_agents / total_agents if total_agents > 0 else 0
        self.happiness_data.append(happiness)
        self.unhappy_count_data.append(len(self.unhappy_agents))
        
        # Calculate segregation index
        segregation_sum = 0
        agent_count = 0
        
        for x in range(self.width):
            for y in range(self.height):
                agent = self.grid[x][y]
                if agent is not None:
                    neighbors = self.get_neighbors(x, y)
                    if neighbors:
                        same_type = sum(1 for n in neighbors if n['type'] == agent['type'])
                        segregation_sum += same_type / len(neighbors)
                        agent_count += 1
        
        segregation = segregation_sum / agent_count if agent_count > 0 else 0
        self.segregation_data.append(segregation)
    
    def get_grid_values(self):
        """
        Return the grid as a numpy array for visualization
        """
        # Create a matrix of agent types
        grid_values = np.zeros((self.width, self.height))
        
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x][y] is not None:
                    grid_values[x, y] = self.grid[x][y]['type'] + 1  # +1 so empty cells are 0
        
        return grid_values

    def run_animation(self):
        """
        Run the model with animation
        """
        # Create figure with subplots
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle("Schelling Segregation Model", fontsize=16)
        
        # Grid subplot
        self.ax_grid = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_grid.set_title(f"Agent Distribution - {self.shape.capitalize()} Shape")
        
        # Get initial grid values
        grid_values = self.get_grid_values()
        
        # Create color map - white for empty, and colors for agents
        cmap_colors = ['white'] + list(plt.cm.tab10.colors[:self.num_agent_types])
        cmap = mcolors.ListedColormap(cmap_colors)
        
        # Display grid
        self.im = self.ax_grid.imshow(grid_values, cmap=cmap, vmin=0, vmax=self.num_agent_types)
        
        # Add a colorbar
        cbar = self.fig.colorbar(self.im, ax=self.ax_grid, ticks=range(self.num_agent_types+1))
        cbar.set_label('Agent Type (0 = Empty)')
        
        # Metrics subplots
        self.ax_happiness = plt.subplot2grid((2, 3), (0, 2))
        self.ax_happiness.set_title("Happiness Over Time")
        self.ax_happiness.set_xlabel("Step")
        self.ax_happiness.set_ylabel("% Happy")
        self.ax_happiness.set_ylim(0, 1.05)
        self.ax_happiness.grid(True)
        
        self.happiness_line, = self.ax_happiness.plot([], [], 'g-', linewidth=2)
        
        self.ax_segregation = plt.subplot2grid((2, 3), (1, 2))
        self.ax_segregation.set_title("Segregation Index")
        self.ax_segregation.set_xlabel("Step")
        self.ax_segregation.set_ylabel("Segregation")
        self.ax_segregation.set_ylim(0, 1.05)
        self.ax_segregation.grid(True)
        
        self.segregation_line, = self.ax_segregation.plot([], [], 'r-', linewidth=2)
        
        # Add status text for steps, moves, etc.
        self.status_text = self.ax_grid.text(0.02, 0.02, "", transform=self.ax_grid.transAxes,
                             bbox=dict(facecolor='white', alpha=0.8))
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, self.update_animation, interval=self.animation_interval, 
            frames=100, blit=False, repeat=False, save_count=100
        )
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Add play/pause button - start with 'Start' since simulation is paused initially
        ax_button = plt.axes([0.45, 0.01, 0.1, 0.04])
        self.button = Button(ax_button, 'Start')
        self.button.on_clicked(self.toggle_animation)
        
        # Add slider for homophily
        ax_slider = plt.axes([0.2, 0.01, 0.2, 0.03])
        self.homophily_slider = Slider(
            ax_slider, 'Homophily', 0.0, 1.0, 
            valinit=self.homophily, valstep=0.05
        )
        self.homophily_slider.on_changed(self.update_homophily)
        
        # Simulation starts paused
        self.is_running = False
        
        # Initial status update
        status = f"Step: {self.steps}\n"
        status += f"Happy: {self.happiness_data[-1]:.1%}\n"
        status += f"Unhappy: {len(self.unhappy_agents)}\n"
        status += f"Moves: {self.total_moves}"
        self.status_text.set_text(status)
        
        # Show plot
        plt.show()
        
        return ani
    
    def update_animation(self, frame):
        """
        Update the animation for each frame
        """
        if not self.is_running:
            return self.im, self.happiness_line, self.segregation_line, self.status_text
        
        # Run a step
        grid_values = self.step()
        
        # Update grid
        self.im.set_array(grid_values)
        
        # Update metrics
        x_data = list(range(len(self.happiness_data)))
        self.happiness_line.set_data(x_data, self.happiness_data)
        self.segregation_line.set_data(x_data, self.segregation_data)
        
        # Auto-adjust axes as needed
        self.ax_happiness.relim()
        self.ax_happiness.autoscale_view(scalex=True, scaley=False)
        self.ax_segregation.relim()
        self.ax_segregation.autoscale_view(scalex=True, scaley=False)
        
        # Update status text
        status = f"Step: {self.steps}\n"
        status += f"Happy: {self.happiness_data[-1]:.1%}\n"
        status += f"Unhappy: {len(self.unhappy_agents)}\n"
        status += f"Moves: {self.total_moves}"
        self.status_text.set_text(status)
        
        # Stop if all agents are happy
        if not self.unhappy_agents:
            if self.is_running:
                print(f"All agents happy after {self.steps} steps!")
                self.is_running = False
                self.button.label.set_text('Resume')
        
        return self.im, self.happiness_line, self.segregation_line, self.status_text
    
    def toggle_animation(self, event):
        """Toggle animation play/pause"""
        self.is_running = not self.is_running
        self.button.label.set_text('Pause' if self.is_running else 'Start')
    
    def update_homophily(self, val):
        """Update the homophily threshold from the slider"""
        self.homophily = val


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Schelling Segregation Model with Animation')
    parser.add_argument('--width', type=int, default=50, help='Width of grid')
    parser.add_argument('--height', type=int, default=50, help='Height of grid')
    parser.add_argument('--density', type=float, default=0.8, help='Population density')
    parser.add_argument('--homophily', type=float, default=0.3, 
                        help='Minimum ratio of similar neighbors for happiness')
    parser.add_argument('--types', type=int, default=2, help='Number of agent types')
    parser.add_argument('--pattern', type=str, default='alternating', 
                      choices=['random', 'checkerboard', 'stripes', 'clusters', 'alternating'],
                      help='Initial pattern of agents')
    parser.add_argument('--interval', type=int, default=200, 
                      help='Animation interval in milliseconds')
    parser.add_argument('--shape', type=str, default='square',
                      choices=['square', 'circle', 'human'],
                      help='Shape of the grid')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"Initializing Schelling model with {args.pattern} pattern...")
    
    # Create model
    model = SchellingModel(
        width=args.width,
        height=args.height,
        density=args.density,
        homophily=args.homophily,
        num_agent_types=args.types,
        pattern_type=args.pattern,
        animation_interval=args.interval,
        shape=args.shape
    )
    
    # Run animation
    model.run_animation()
