"""
Visualization module for the Advanced Schelling Segregation Model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, RadioButtons
import seaborn as sns


class SchellingVisualizer:
    """
    Advanced visualization for the Schelling model with:
    - Multiple view modes (agent type, income, happiness)
    - Detailed analytics dashboard
    - Interactive controls
    - Geographic features visualization
    """
    
    def __init__(self, model, animation_interval=200):
        """
        Initialize the visualizer with a model.
        
        Args:
            model: An AdvancedSchellingModel instance
            animation_interval: Milliseconds between animation frames
        """
        self.model = model
        self.animation_interval = animation_interval
        self.is_running = False
        self.view_mode = 'type'  # Default view mode
        self.show_hubs = True    # Whether to show hub locations
        self.ani = None
        
        # Visualization elements
        self.fig = None
        self.ax_grid = None
        self.ax_happiness = None
        self.ax_segregation = None
        self.ax_income = None
        self.ax_distribution = None
        self.im = None
        self.cbar = None
        self.status_text = None
        self.lines = {}
        
    def create_visualization(self):
        """Set up the visualization figure and subplots."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("Advanced Schelling Segregation Model", fontsize=16)
        
        # Grid visualization (left side)
        self.ax_grid = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
        self.ax_grid.set_title(f"Agent Distribution - {self.model.shape.capitalize()} Shape")
        
        # Hub markers
        self.hub_markers = []
        
        # Metrics visualizations (right side)
        self.ax_happiness = plt.subplot2grid((3, 3), (0, 2))
        self.ax_happiness.set_title("Happiness Over Time")
        self.ax_happiness.set_xlabel("Step")
        self.ax_happiness.set_ylabel("% Happy")
        self.ax_happiness.set_ylim(0, 1.05)
        self.ax_happiness.grid(True)
        
        self.ax_segregation = plt.subplot2grid((3, 3), (1, 2))
        self.ax_segregation.set_title("Segregation Indices")
        self.ax_segregation.set_xlabel("Step")
        self.ax_segregation.set_ylabel("Segregation")
        self.ax_segregation.set_ylim(0, 1.05)
        self.ax_segregation.grid(True)
        
        # Metrics visualization (bottom row)
        self.ax_income = plt.subplot2grid((3, 3), (2, 0))
        self.ax_income.set_title("Income Distribution")
        self.ax_income.set_xlabel("Income")
        self.ax_income.set_ylabel("Density")
        
        self.ax_distribution = plt.subplot2grid((3, 3), (2, 1), colspan=2)
        self.ax_distribution.set_title("Agent Characteristics")
        self.ax_distribution.set_xlabel("Step")
        self.ax_distribution.set_ylabel("Count")
        
        # Get initial grid values
        grid_values = self.model.get_grid_values()
        
        # Create color maps for different view modes
        self._setup_colormaps()
        
        # Display initial grid
        self._update_grid_display()
        
        # Add status text for steps, moves, etc.
        self.status_text = self.ax_grid.text(0.02, 0.02, "", transform=self.ax_grid.transAxes,
                                      bbox=dict(facecolor='white', alpha=0.8))
        
        # Initialize line plots
        self._initialize_line_plots()
        
        # Add interactive controls
        self._add_controls()
        
        # Update initial status
        self._update_status_text()
        
    def _setup_colormaps(self):
        """Set up color maps for different view modes."""
        self.colormaps = {}
        
        # Type colormap - unique color for each agent type, white for empty, black for barriers
        type_colors = ['white', 'black', 'gray', '#333333']  # For empty, barriers, masked, premium
        type_colors.extend(list(plt.cm.tab10.colors[:self.model.num_agent_types]))
        self.colormaps['type'] = mcolors.ListedColormap(type_colors)
        
        # Income colormap - gradient for income levels, distinctive colors for special cells
        self.colormaps['income'] = plt.cm.viridis.copy()
        self.colormaps['income'].set_bad('black')  # Barriers
        
        # Happiness colormap - red to green gradient
        self.colormaps['happiness'] = plt.cm.RdYlGn.copy()
        self.colormaps['happiness'].set_bad('black')  # Barriers
        
    def _update_grid_display(self):
        """Update the grid display based on current view mode."""
        grid_values = self.model.get_grid_values()
        current_grid = grid_values[self.view_mode]
        
        # Clear any existing image and hub markers
        if hasattr(self, 'im') and self.im is not None:
            self.im.remove()
        if hasattr(self, 'cbar') and self.cbar is not None:
            self.cbar.remove()
        
        # Remove existing hub markers
        for marker in self.hub_markers:
            if marker in self.ax_grid.collections:
                marker.remove()
        self.hub_markers = []
        
        # Display grid with appropriate colormap
        if self.view_mode == 'type':
            # Type view shows agent types with discrete colors
            vmin, vmax = -3, self.model.num_agent_types
            self.im = self.ax_grid.imshow(
                current_grid, 
                cmap=self.colormaps[self.view_mode],
                vmin=vmin, vmax=vmax
            )
            
            # Add colorbar with custom ticks
            ticks = list(range(-3, self.model.num_agent_types + 1))
            self.cbar = self.fig.colorbar(self.im, ax=self.ax_grid, ticks=ticks)
            
            # Custom labels for the colorbar
            tick_labels = ['Premium', 'Masked', 'Barrier', 'Empty']
            tick_labels.extend([f'Type {i}' for i in range(self.model.num_agent_types)])
            self.cbar.set_ticklabels(tick_labels)
            
        elif self.view_mode == 'income':
            # Income view shows income levels with a gradient
            # Mask barriers and non-agent cells
            masked_grid = np.ma.masked_where(
                (current_grid < 0) | (current_grid == 0), 
                current_grid
            )
            self.im = self.ax_grid.imshow(
                masked_grid,
                cmap=self.colormaps[self.view_mode],
                vmin=0, vmax=1
            )
            self.cbar = self.fig.colorbar(self.im, ax=self.ax_grid)
            self.cbar.set_label('Income Level')
            
        elif self.view_mode == 'happiness':
            # Happiness view shows agent happiness levels
            # Mask barriers and non-agent cells
            masked_grid = np.ma.masked_where(
                (current_grid < 0) | (current_grid == 0), 
                current_grid
            )
            self.im = self.ax_grid.imshow(
                masked_grid,
                cmap=self.colormaps[self.view_mode],
                vmin=0, vmax=1
            )
            self.cbar = self.fig.colorbar(self.im, ax=self.ax_grid)
            self.cbar.set_label('Happiness Level')
        
        # Display hubs view if selected
        if self.view_mode == 'hubs' and hasattr(self.model, 'use_hubs') and self.model.use_hubs:
            # Create a special visualization for hubs
            # Base grid shows agent types
            base_grid = grid_values['type']
            self.im = self.ax_grid.imshow(
                base_grid, 
                cmap=self.colormaps['type'],
                vmin=-3, vmax=self.model.num_agent_types
            )
            
            # Add colorbar with custom ticks
            ticks = list(range(-3, self.model.num_agent_types + 1))
            self.cbar = self.fig.colorbar(self.im, ax=self.ax_grid, ticks=ticks)
            
            # Custom labels for the colorbar
            tick_labels = ['Premium', 'Masked', 'Barrier', 'Empty']
            tick_labels.extend([f'Type {i}' for i in range(self.model.num_agent_types)])
            self.cbar.set_ticklabels(tick_labels)
            
            # Add hub influence visualization as a contour overlay
            self._add_hub_visualization()
        
        # Update grid title
        self.ax_grid.set_title(f"{self.view_mode.capitalize()} View - {self.model.shape.capitalize()} Shape")
        
        # Add hub markers if enabled and not in hubs view
        if (self.show_hubs and hasattr(self.model, 'use_hubs') and 
                self.model.use_hubs and self.view_mode != 'hubs'):
            self._add_hub_markers()
        
    def _initialize_line_plots(self):
        """Initialize the line plots for metrics."""
        # Happiness line
        self.lines['happiness'], = self.ax_happiness.plot(
            [], [], 'g-', linewidth=2, label='Happiness'
        )
        
        # Segregation lines
        self.lines['segregation'], = self.ax_segregation.plot(
            [], [], 'r-', linewidth=2, label='Type Segregation'
        )
        self.lines['income_segregation'], = self.ax_segregation.plot(
            [], [], 'b-', linewidth=2, label='Income Segregation'
        )
        
        # Add hub segregation lines if hubs are enabled
        if hasattr(self.model, 'use_hubs') and self.model.use_hubs:
            self.lines['hub_segregation'], = self.ax_segregation.plot(
                [], [], 'g--', linewidth=1.5, label='Hub Segregation'
            )
            self.lines['hub_income_segregation'], = self.ax_segregation.plot(
                [], [], 'm--', linewidth=1.5, label='Hub Income Segregation'
            )
        self.ax_segregation.legend()
        
        # Agent count lines by type
        self.lines['agent_counts'] = []
        for i in range(self.model.num_agent_types):
            line, = self.ax_distribution.plot(
                [], [], '-', 
                color=plt.cm.tab10.colors[i % 10], 
                linewidth=2, 
                label=f'Type {i}'
            )
            self.lines['agent_counts'].append(line)
        self.ax_distribution.legend()
    
    def _add_controls(self):
        """Add interactive controls to the visualization."""
        # Adjust layout to make room for controls
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        # Add animation control button
        ax_button = plt.axes([0.4, 0.01, 0.08, 0.04])
        self.button = Button(ax_button, 'Start')
        self.button.on_clicked(self.toggle_animation)
        
        # Add reset button
        ax_reset = plt.axes([0.5, 0.01, 0.08, 0.04])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset_model)
        
        # Add slider for homophily
        ax_slider = plt.axes([0.2, 0.02, 0.18, 0.03])
        self.homophily_slider = Slider(
            ax_slider, 'Homophily', 0.0, 1.0, 
            valinit=self.model.global_homophily, 
            valstep=0.05
        )
        self.homophily_slider.on_changed(self.update_homophily)
        
        # Add radio buttons for view mode
        view_options = ['type', 'income', 'happiness']
        if hasattr(self.model, 'use_hubs') and self.model.use_hubs:
            view_options.append('hubs')
            
        ax_radio = plt.axes([0.65, 0.01, 0.15, 0.06])
        self.radio = RadioButtons(
            ax_radio, view_options,
            active=0
        )
        self.radio.on_clicked(self.change_view_mode)
        
        # Add toggle for hub markers if hubs are enabled
        if hasattr(self.model, 'use_hubs') and self.model.use_hubs:
            ax_hub_toggle = plt.axes([0.82, 0.01, 0.15, 0.04])
            self.hub_toggle = Button(ax_hub_toggle, 'Hide Hubs')
            self.hub_toggle.on_clicked(self.toggle_hub_markers)
            
    def toggle_animation(self, event):
        """Start or stop the animation."""
        self.is_running = not self.is_running
        self.button.label.set_text('Pause' if self.is_running else 'Start')
        
    def update_homophily(self, val):
        """Update the homophily threshold from the slider."""
        self.model.global_homophily = val
        
    def change_view_mode(self, new_mode):
        """Change the grid view mode."""
        self.view_mode = new_mode
        self._update_grid_display()
        
    def reset_model(self, event):
        """Reset the model to its initial state."""
        # Pause the animation if it's running
        was_running = self.is_running
        self.is_running = False
        self.button.label.set_text('Start')
        
        # Create a new model with the same parameters
        new_model = self.model.__class__(
            width=self.model.width,
            height=self.model.height,
            density=self.model.density,
            global_homophily=self.model.global_homophily,
            num_agent_types=self.model.num_agent_types,
            pattern_type=self.model.pattern_type,
            shape=self.model.shape,
            use_networks=self.model.use_networks,
            geographic_features=self.model.geographic_features,
            income_segregation=self.model.income_segregation,
            use_hubs=self.model.use_hubs if hasattr(self.model, 'use_hubs') else False
        )
        
        # Update the model reference
        self.model = new_model
        
        # Reset all plots and data
        self._clear_all_plots()
        self._update_grid_display()
        self._initialize_line_plots()
        self._update_status_text()
        
        # Restart the animation if it was running
        
        self.ax_segregation.clear()
        self.ax_segregation.set_title("Segregation Indices")
        self.ax_segregation.set_xlabel("Step")
        self.ax_segregation.set_ylabel("Segregation")
        self.ax_segregation.set_ylim(0, 1.05)
        self.ax_segregation.grid(True)
        
        self.ax_distribution.clear()
        self.ax_distribution.set_title("Agent Characteristics")
        self.ax_distribution.set_xlabel("Step")
        self.ax_distribution.set_ylabel("Count")
        
        # Reset income plot
        self.ax_income.clear()
        self.ax_income.set_title("Income Distribution")
        self.ax_income.set_xlabel("Income")
        self.ax_income.set_ylabel("Density")
        
        # Clear other plots/elements by clearing the axis instead of removing artists
        # This avoids the NotImplementedError when trying to remove matplotlib artists
        self.ax_grid.clear()
        self.ax_grid.set_title(f"Agent Distribution - {self.model.shape.capitalize()} Shape")
        
        # Reset the image and colorbar references without trying to remove them
        self.im = None
        self.cbar = None
        
        # Clear hub markers
        for marker in self.hub_markers:
            if marker in self.ax_grid.collections:
                marker.remove()
        self.hub_markers = []
        
        # Add status text
        if self.status_text is None:
            self.status_text = self.ax_grid.text(0.02, 0.02, "", transform=self.ax_grid.transAxes,
                                       bbox=dict(facecolor='white', alpha=0.8))
    
    def toggle_hub_markers(self, event):
        """Toggle the display of hub markers."""
        self.show_hubs = not self.show_hubs
        self.hub_toggle.label.set_text('Show Hubs' if not self.show_hubs else 'Hide Hubs')
        self._update_grid_display()
    
    def _add_hub_markers(self):
        """Add markers for hubs on the grid."""
        if not hasattr(self.model, 'hubs'):
            return
            
        # Hub type colors
        hub_colors = {
            'work': 'red',
            'education': 'blue',
            'leisure': 'green',
            'shopping': 'orange',
            'transit': 'purple'
        }
        
        # Add a marker for each hub
        for hub in self.model.hubs:
            # Marker size based on importance
            size = 50 + hub.importance * 100
            
            # Marker color based on hub type
            color = hub_colors.get(hub.type, 'gray')
            
            # Create marker
            marker = self.ax_grid.scatter(
                hub.y, hub.x,  # Note: x,y are flipped for imshow
                s=size,
                color=color,
                alpha=0.6,
                edgecolors='white',
                linewidths=1,
                marker='o'
            )
            
            self.hub_markers.append(marker)
    
    def _add_hub_visualization(self):
        """Add detailed hub visualization for hubs view mode."""
        if not hasattr(self.model, 'hubs') or not self.model.hubs:
            return
            
        # Create a grid of hub influence
        influence_grid = np.zeros((self.model.width, self.model.height))
        
        # Calculate hub influence at each position
        for x in range(self.model.width):
            for y in range(self.model.height):
                # Skip barriers and off-grid areas
                if not self.model.mask[x, y] or (x, y) in self.model.barriers:
                    continue
                    
                # Accumulate influence from each hub
                total_influence = 0
                for hub in self.model.hubs:
                    # Calculate distance-based influence
                    dist = ((x - hub.x)**2 + (y - hub.y)**2)**0.5
                    if dist <= hub.radius * 2:
                        # Influence diminishes with distance
                        influence = max(0, 1 - (dist / (hub.radius * 2)))
                        influence *= hub.importance
                        
                        # Add segregation factor
                        influence *= (1 + hub.segregation_index)
                        
                        total_influence += influence
                
                influence_grid[x, y] = min(1.0, total_influence)
        
    def _update_status_text(self):
        """Update the status text with current metrics."""
        status = f"Step: {self.model.steps}\n"
        
        # Calculate happiness percentage
        total_agents = len(self.model.agents)
        if total_agents > 0:
            happy_percent = 100 * (total_agents - len(self.model.unhappy_agents)) / total_agents
            status += f"Happy: {happy_percent:.1f}%\n"
        else:
            status += "Happy: N/A\n"
            
        status += f"Unhappy: {len(self.model.unhappy_agents)}\n"
        status += f"Total Moves: {self.model.total_moves}"
        
        self.status_text.set_text(status)
    
    def update_animation(self, frame):
        """Update function for the animation."""
        if not self.is_running:
            return self.im, self.status_text
        
        # Run a step of the model
        self.model.step()
        
        # Update grid visualization
        grid_values = self.model.get_grid_values()
        current_grid = grid_values[self.view_mode]
        
        if self.view_mode == 'type':
            self.im.set_array(current_grid)
        else:
            # For income and happiness, mask special cells
            masked_grid = np.ma.masked_where(
                (current_grid < 0) | (current_grid == 0), 
                current_grid
            )
            self.im.set_array(masked_grid)
        
        # Update line plots
        x_data = list(range(len(self.model.happiness_data)))
        
        # Update happiness line
        self.lines['happiness'].set_data(x_data, self.model.happiness_data)
        
        # Update segregation lines
        self.lines['segregation'].set_data(x_data, self.model.segregation_data)
        self.lines['income_segregation'].set_data(x_data, self.model.income_segregation_index)
        
        # Update hub segregation lines if applicable
        if hasattr(self.model, 'use_hubs') and self.model.use_hubs:
            if hasattr(self.model, 'hub_segregation_data') and self.model.hub_segregation_data:
                self.lines['hub_segregation'].set_data(x_data, self.model.hub_segregation_data)
            if hasattr(self.model, 'hub_income_segregation_data') and self.model.hub_income_segregation_data:
                self.lines['hub_income_segregation'].set_data(x_data, self.model.hub_income_segregation_data)
        
        # Update agent type counts
        if self.model.type_distributions:
            for i in range(self.model.num_agent_types):
                y_data = [dist.get(i, 0) for dist in self.model.type_distributions]
                self.lines['agent_counts'][i].set_data(x_data, y_data)
        
        # Update income distribution plot
        if self.model.steps % 5 == 0:  # Update every 5 steps to avoid performance issues
            self.ax_income.clear()
            self.ax_income.set_title("Income Distribution")
            self.ax_income.set_xlabel("Income")
            self.ax_income.set_ylabel("Density")
            
            # Get income data from agents
            incomes = [agent.income for agent in self.model.agents.values()]
            if incomes:
                sns.kdeplot(incomes, ax=self.ax_income)
                self.ax_income.axvline(np.mean(incomes), color='r', linestyle='--', label='Mean')
                self.ax_income.legend()
        
        # Auto-adjust axes as needed
        for ax in [self.ax_happiness, self.ax_segregation, self.ax_distribution]:
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)
        
        # Update status text
        self._update_status_text()
        
        # Stop if all agents are happy
        if not self.model.unhappy_agents:
            if self.is_running:
                print(f"All agents happy after {self.model.steps} steps!")
                self.is_running = False
                self.button.label.set_text('Resume')
        
        return self.im, self.status_text
    
    def run_animation(self):
        """Run the animation."""
        # Create the visualization elements
        self.create_visualization()
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_animation, 
            interval=self.animation_interval, 
            frames=1000, blit=False, repeat=False
        )
        
        # Show the visualization
        plt.show()
        
        return self.ani
