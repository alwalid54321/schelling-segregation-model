# run.py
import argparse
from model import SchellingModel
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=20)
    parser.add_argument('--height', type=int, default=20)
    parser.add_argument('--density', type=float, default=1.0)  # Default to full grid
    parser.add_argument('--homophily', type=float, default=0.3)
    parser.add_argument('--agent-types', type=int, default=2)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    model = SchellingModel(
        width=args.width,
        height=args.height,
        density=args.density,
        homophily=args.homophily,
        num_agent_types=args.agent_types
    )
    
    # Track metrics
    unhappy_history = []
    happiness_history = []
    segregation_history = []
    
    # Run model for specified steps or until all agents are happy
    for i in range(args.steps):
        model.step()
        
        # Collect metrics
        unhappy_fraction = len(model.unhappy_agents) / len(model.agents)
        unhappy_history.append(unhappy_fraction)
        
        # Get data from datacollector
        model_vars = model.datacollector.get_model_vars_dataframe().iloc[-1]
        happiness_history.append(model_vars["Happiness"])
        segregation_history.append(model_vars["Segregation"])
        
        # Stop if all agents are happy
        if not model.unhappy_agents:
            print(f"All agents happy after {i+1} steps!")
            break

    # Print final stats
    print(f"Final stats after {model.steps} steps:")
    print(f"  Happy agents: {len(model.agents) - len(model.unhappy_agents)}/{len(model.agents)} ({1.0 - unhappy_history[-1]:.1%})")
    print(f"  Segregation index: {segregation_history[-1]:.3f}")
    print(f"  Total moves: {model.total_moves}")

    if args.plot:
        # Set up figure with multiple subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot unhappy fraction
        axes[0].plot(unhappy_history, 'r-', label='Unhappy Fraction')
        axes[0].plot(happiness_history, 'g-', label='Happiness')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Fraction')
        axes[0].set_title('Agent Happiness Over Time')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot segregation
        axes[1].plot(segregation_history, 'b-', label='Segregation Index')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Index Value')
        axes[1].set_title('Segregation Index Over Time')
        axes[1].grid(True)
        
        # Display the grid state (create a grid visualization)
        # Get grid state
        grid_state = np.zeros((model.width, model.height))
        for cell_content, (x, y) in model.grid.coord_iter():
            for agent in cell_content:
                grid_state[x][y] = agent.agent_type + 1
        
        # Add grid visualization as inset in top plot
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(axes[0], width="30%", height="30%", loc=4)
        ax_inset.imshow(grid_state, cmap='tab10', interpolation='nearest')
        ax_inset.set_title('Final Grid State')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('schelling_results.png')
        plt.show()


if __name__ == '__main__':
    main()