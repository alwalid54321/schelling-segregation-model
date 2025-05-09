"""
Schelling Segregation Model - Simple implementation
Using the core Mesa framework
"""

import matplotlib.pyplot as plt
import numpy as np
import random

# Since we're having issues with the Mesa installation, let's create a simple version
# that doesn't rely on all Mesa components


class SchellingModel:
    """
    A simple implementation of the Schelling segregation model.
    """
    
    def __init__(self, width=20, height=20, density=0.8, homophily=0.3, num_agent_types=2):
        self.width = width
        self.height = height
        self.density = density
        self.homophily = homophily
        self.num_agent_types = num_agent_types
        
        # Create the grid with None for empty cells
        self.grid = [[None for _ in range(height)] for _ in range(width)]
        
        # Track unhappy agents
        self.unhappy_agents = []
        self.steps = 0
        self.total_moves = 0
        
        # Create agents
        self.populate_grid()
        
        # Store metrics
        self.segregation_data = []
        self.happiness_data = []
    
    def populate_grid(self):
        """
        Populate the grid with agents
        """
        agent_id = 0
        
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < self.density:
                    # Create agent (just a dictionary for simplicity)
                    agent_type = random.randint(0, self.num_agent_types - 1)
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
    
    def get_neighbors(self, x, y):
        """
        Get the neighbors of a cell
        """
        neighbors = []
        
        # Check the 8 surrounding cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
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
        Move unhappy agents to random empty cells
        """
        # Find empty cells
        empty_cells = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x][y] is None:
                    empty_cells.append((x, y))
        
        # Shuffle empty cells
        random.shuffle(empty_cells)
        
        # Move unhappy agents
        for agent, x, y in self.unhappy_agents:
            if empty_cells:
                # Get a new position
                new_x, new_y = empty_cells.pop()
                
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
    
    def run(self, steps=100):
        """
        Run the model for a specified number of steps
        """
        for _ in range(steps):
            self.step()
            
            # Optional: print progress
            if _ % 5 == 0:
                happy_pct = 100 * self.happiness_data[-1]
                print(f"Step {_}: {happy_pct:.1f}% agents happy, {self.total_moves} total moves")
            
            # Stop if all agents are happy
            if not self.unhappy_agents:
                print(f"All agents happy after {_+1} steps!")
                break
    
    def plot_results(self):
        """
        Plot the results of the simulation
        """
        plt.figure(figsize=(10, 8))
        
        # Plot grid
        plt.subplot(2, 1, 1)
        self.plot_grid()
        plt.title('Agent Distribution')
        
        # Plot metrics
        plt.subplot(2, 2, 3)
        plt.plot(self.happiness_data, 'g-')
        plt.xlabel('Step')
        plt.ylabel('% Happy')
        plt.title('Happiness Over Time')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.segregation_data, 'r-')
        plt.xlabel('Step')
        plt.ylabel('Segregation')
        plt.title('Segregation Index')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('schelling_results_simple.png')
        print("Results saved to 'schelling_results_simple.png'")
        plt.show()
    
    def plot_grid(self):
        """
        Plot the current state of the grid
        """
        # Create a matrix of agent types
        grid_values = np.zeros((self.width, self.height))
        
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x][y] is not None:
                    grid_values[x, y] = self.grid[x][y]['type'] + 1  # +1 so empty cells are 0
        
        # Plot the grid
        colors = ['white', 'red', 'blue', 'green', 'yellow', 'purple']
        plt.imshow(grid_values, cmap=plt.cm.colors.ListedColormap(colors[:self.num_agent_types+1]))
        plt.colorbar(ticks=range(self.num_agent_types+1), 
                    label='Agent Type (0 = Empty)')
        plt.xticks([])
        plt.yticks([])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Schelling Segregation Model')
    parser.add_argument('--width', type=int, default=20, help='Width of grid')
    parser.add_argument('--height', type=int, default=20, help='Height of grid')
    parser.add_argument('--density', type=float, default=0.8, help='Population density')
    parser.add_argument('--homophily', type=float, default=0.3, 
                        help='Minimum ratio of similar neighbors for happiness')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps to run')
    parser.add_argument('--types', type=int, default=2, help='Number of agent types')
    
    args = parser.parse_args()
    
    # Create and run model
    model = SchellingModel(
        width=args.width,
        height=args.height,
        density=args.density,
        homophily=args.homophily,
        num_agent_types=args.types
    )
    
    print("Running model...")
    model.run(args.steps)
    
    # Print summary statistics
    final_segregation = model.segregation_data[-1]
    final_happiness = model.happiness_data[-1]
    
    print("\n--- SUMMARY STATISTICS ---")
    print(f"Final Segregation Index: {final_segregation:.4f}")
    print(f"Final Happiness: {final_happiness:.2%}")
    print(f"Total Moves: {model.total_moves}")
    print(f"Steps: {model.steps}")
    
    # Plot results
    model.plot_results()
