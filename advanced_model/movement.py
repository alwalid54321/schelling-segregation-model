"""
Movement strategies for the Advanced Schelling Segregation Model.
"""

import random
import numpy as np
from collections import defaultdict


def move_unhappy_agents(self):
    """
    Advanced agent relocation strategy.
    
    This implementation uses:
    - Smart location selection to maximize happiness
    - Income-based location preferences 
    - Consideration of barriers and premium locations
    - Agent's social network connections
    """
    # Find empty cells
    empty_cells = []
    for x in range(self.width):
        for y in range(self.height):
            if (self.grid[x][y] is None and 
                self.mask[x, y] and 
                (x, y) not in self.barriers):
                
                # Calculate cell's desirability (premium locations are more desirable)
                desirability = 1.0
                if (x, y) in self.premium_locations:
                    desirability = 2.0
                
                empty_cells.append((x, y, desirability))
    
    # Sort unhappy agents by:
    # 1. Income (higher income agents get priority)
    # 2. Unhappiness level (most unhappy first)
    sorted_unhappy = []
    
    for agent in self.unhappy_agents:
        if len(agent.happiness_history) > 0:
            happiness_score = agent.happiness_history[-1]
        else:
            happiness_score = 0
            
        # Priority score combines income and unhappiness
        priority = agent.income / 100 + (1 - happiness_score)
        sorted_unhappy.append((agent, priority))
    
    # Sort by priority (highest first)
    sorted_unhappy.sort(key=lambda x: x[1], reverse=True)
    
    # For each unhappy agent, try to find the best position
    moves_made = 0
    
    for agent, _ in sorted_unhappy:
        if not empty_cells:
            break  # No more empty cells
        
        # Get current position
        x, y = agent.x, agent.y
        
        # Try to find a position that would make the agent happy
        best_position = None
        best_score = -1
        
        # Sample of positions to try (checking all would be too slow)
        num_to_sample = min(15, len(empty_cells))
        
        # Weight sampling by desirability and agent's income
        weights = []
        for _, _, desirability in empty_cells[:num_to_sample]:
            # Higher income agents care more about desirability
            weight = desirability ** (agent.income / 50)
            weights.append(weight)
        
        # Normalize weights
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]
        else:
            weights = None
        
        # Sample positions
        sampled_indices = random.choices(
            range(min(num_to_sample, len(empty_cells))), 
            weights=weights, 
            k=min(num_to_sample, len(empty_cells))
        )
        
        for idx in sampled_indices:
            new_x, new_y, _ = empty_cells[idx]
            
            # Check if we'd be happy at this new position
            # Get potential neighbors
            potential_neighbors = self.get_neighbors(new_x, new_y)
            
            # Evaluate potential location
            potential_score = agent.evaluate_potential_location(
                new_x, new_y, potential_neighbors
            )
            
            # If score is good enough, choose this position
            if potential_score >= agent.tolerance and potential_score > best_score:
                best_score = potential_score
                best_position = (new_x, new_y)
        
        # If found a good position, move there
        if best_position:
            new_x, new_y = best_position
            # Find and remove this position from empty cells
            for i, (ex, ey, _) in enumerate(empty_cells):
                if ex == new_x and ey == new_y:
                    empty_cells.pop(i)
                    break
        # Otherwise just pick a random position
        elif empty_cells:
            idx = random.randrange(len(empty_cells))
            new_x, new_y, _ = empty_cells.pop(idx)
        else:
            continue  # No empty cells left
        
        # Update agent position
        self.grid[x][y] = None
        self.grid[new_x][new_y] = agent
        agent.add_location_to_history(new_x, new_y)
        agent.decrease_attachment()  # Reset attachment to new location
        
        # Update metrics
        self.total_moves += 1
        moves_made += 1
        
        # Add the old position to empty cells
        empty_cells.append((x, y, 1.0))  # Regular desirability
    
    return moves_made


def collect_data(self):
    """
    Collect comprehensive data for analysis.
    """
    # Calculate happiness percentage
    total_agents = len(self.agents)
    happy_agents = total_agents - len(self.unhappy_agents)
    happiness = happy_agents / total_agents if total_agents > 0 else 0
    self.happiness_data.append(happiness)
    
    # Calculate average number of moves
    avg_moves = sum(agent.moves for agent in self.agents.values()) / total_agents if total_agents > 0 else 0
    self.avg_moves_data.append(avg_moves)
    
    # Calculate segregation index - measure of clustering by type
    segregation_sum = 0
    agent_count = 0
    
    for agent_id, agent in self.agents.items():
        neighbors = self.get_neighbors(agent.x, agent.y)
        if neighbors:
            same_type = sum(1 for n in neighbors if n.type == agent.type)
            segregation_sum += same_type / len(neighbors)
            agent_count += 1
    
    segregation = segregation_sum / agent_count if agent_count > 0 else 0
    self.segregation_data.append(segregation)
    
    # Calculate income segregation - measure how much income is clustered
    income_segregation_sum = 0
    
    for agent_id, agent in self.agents.items():
        neighbors = self.get_neighbors(agent.x, agent.y)
        if neighbors:
            # Calculate income similarity with neighbors
            agent_income = agent.income
            neighbor_incomes = [n.income for n in neighbors]
            avg_neighbor_income = sum(neighbor_incomes) / len(neighbors)
            
            # Lower difference = higher similarity
            income_diff = abs(agent_income - avg_neighbor_income)
            max_possible_diff = 100  # Assuming income range 0-100
            income_similarity = 1 - (income_diff / max_possible_diff)
            
            income_segregation_sum += income_similarity
    
    income_segregation = income_segregation_sum / agent_count if agent_count > 0 else 0
    self.income_segregation_index.append(income_segregation)
    
    # Calculate type distribution
    type_counts = defaultdict(int)
    for agent in self.agents.values():
        type_counts[agent.type] += 1
    
    self.type_distributions.append(type_counts)


def get_grid_values(self):
    """
    Return the grid as a numpy array for visualization.
    Includes different layers of information.
    """
    # Create grid arrays for different visualizations
    type_grid = np.zeros((self.width, self.height))
    income_grid = np.zeros((self.width, self.height))
    happiness_grid = np.zeros((self.width, self.height))
    
    # Fill with agent data
    for x in range(self.width):
        for y in range(self.height):
            agent = self.grid[x][y]
            
            if agent is not None:
                type_grid[x, y] = agent.type + 1  # +1 so empty cells are 0
                income_grid[x, y] = agent.income / 100  # Normalize to 0-1
                happiness_grid[x, y] = 0.5 + (0.5 if agent.is_happy else 0)  # 0.5=unhappy, 1=happy
            elif (x, y) in self.barriers:
                # Mark barriers
                type_grid[x, y] = -1
                income_grid[x, y] = -1
                happiness_grid[x, y] = -1
            elif not self.mask[x, y]:
                # Mark masked areas
                type_grid[x, y] = -2
                income_grid[x, y] = -2
                happiness_grid[x, y] = -2
            elif (x, y) in self.premium_locations:
                # Mark premium locations (only visible when empty)
                type_grid[x, y] = -3
                income_grid[x, y] = -3
                happiness_grid[x, y] = -3
    
    return {
        'type': type_grid,
        'income': income_grid,
        'happiness': happiness_grid
    }
