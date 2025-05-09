#!/usr/bin/env python
"""
Main script to run the Advanced Schelling Segregation Model.
This version includes multiple improvements:

1. Advanced Agent Properties:
   - Income levels affecting location preferences
   - Education levels affecting tolerance
   - Age demographics 
   - Individual tolerance thresholds
   - Social networks between agents

2. Enhanced Environment:
   - Geographic barriers (rivers, highways)
   - Premium locations (parks, waterfronts)
   - Multiple grid shapes (square, circle, human)
   - Urban activity hubs (work, education, leisure, etc.)

3. Improved Dynamics:
   - Smart relocation strategy
   - Location attachment mechanism
   - Income-based segregation metrics
   - Social network influence
   - Hub-based activity and segregation

4. Detailed Visualization:
   - Multiple view modes (type, income, happiness)
   - Extensive metrics dashboard
   - Interactive controls
   - Dynamic income distribution plots
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import seaborn as sns

from advanced_model.agent import Agent
from advanced_model.model import AdvancedSchellingModel
from advanced_model.visualization import SchellingVisualizer
from advanced_model.hub_features import add_hub_methods_to_model

# Import specific functions from model_methods
from advanced_model.model_methods import (
    initialize_grid,
    _initialize_random,
    _initialize_alternating,
    _initialize_checkerboard,
    _initialize_stripes,
    _initialize_clusters,
    _initialize_income_stratified,
    _make_all_agents_unhappy,
    _create_social_networks,
    get_neighbors,
    step
)

# Import functions from movement
from advanced_model.movement import (
    move_unhappy_agents,
    collect_data,
    get_grid_values
)

# Add methods to the model class
AdvancedSchellingModel.initialize_grid = initialize_grid
AdvancedSchellingModel._initialize_random = _initialize_random
AdvancedSchellingModel._initialize_alternating = _initialize_alternating
AdvancedSchellingModel._initialize_checkerboard = _initialize_checkerboard
AdvancedSchellingModel._initialize_stripes = _initialize_stripes
AdvancedSchellingModel._initialize_clusters = _initialize_clusters
AdvancedSchellingModel._initialize_income_stratified = _initialize_income_stratified
AdvancedSchellingModel._make_all_agents_unhappy = _make_all_agents_unhappy
AdvancedSchellingModel._create_social_networks = _create_social_networks
AdvancedSchellingModel.get_neighbors = get_neighbors
AdvancedSchellingModel.step = step
AdvancedSchellingModel.move_unhappy_agents = move_unhappy_agents
AdvancedSchellingModel.collect_data = collect_data
AdvancedSchellingModel.get_grid_values = get_grid_values

# Add hub-based methods
add_hub_methods_to_model(AdvancedSchellingModel)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Advanced Schelling Segregation Model')
    
    # Grid parameters
    parser.add_argument('--width', type=int, default=50, help='Width of grid')
    parser.add_argument('--height', type=int, default=50, help='Height of grid')
    parser.add_argument('--density', type=float, default=0.7, help='Population density')
    
    # Agent parameters
    parser.add_argument('--homophily', type=float, default=0.3, 
                        help='Base homophily threshold for happiness')
    parser.add_argument('--types', type=int, default=2, help='Number of agent types')
    
    # Initialization parameters
    parser.add_argument('--pattern', type=str, default='alternating', 
                      choices=['random', 'checkerboard', 'stripes', 'clusters', 
                               'alternating', 'income_stratified'],
                      help='Initial pattern of agents')
    parser.add_argument('--shape', type=str, default='square',
                      choices=['square', 'circle', 'human'],
                      help='Shape of the grid')
    
    # Advanced features
    parser.add_argument('--networks', action='store_true', 
                      help='Enable social networks between agents')
    parser.add_argument('--geographic', action='store_true',
                      help='Add geographic features like barriers and premium locations')
    parser.add_argument('--income', action='store_true',
                      help='Enable income-based segregation mechanisms')
    parser.add_argument('--hubs', action='store_true',
                      help='Enable urban activity hubs (work, education, leisure, etc.)')
    
    # Visualization parameters
    parser.add_argument('--interval', type=int, default=200, 
                      help='Animation interval in milliseconds')
    
    # Quick demo mode with predefined parameters
    parser.add_argument('--demo', type=str, choices=['basic', 'income', 'barriers', 'full'],
                      help='Run a predefined demo configuration')
    
    args = parser.parse_args()
    
    # If demo mode is specified, override other parameters
    if args.demo:
        setup_demo_parameters(args)
    
    return args


def setup_demo_parameters(args):
    """Set up predefined parameters for demo modes."""
    if args.demo == 'basic':
        # Simple demo with default settings
        args.pattern = 'alternating'
        args.shape = 'square'
        args.networks = False
        args.geographic = False
        args.income = False
        args.hubs = False
    
    elif args.demo == 'income':
        # Demo focusing on income-based segregation
        args.pattern = 'income_stratified'
        args.shape = 'square'
        args.networks = False
        args.geographic = False
        args.income = True
        args.hubs = False
    
    elif args.demo == 'barriers':
        # Demo with geographic features
        args.pattern = 'random'
        args.shape = 'square'
        args.networks = False
        args.geographic = True
        args.income = False
        args.hubs = False
    
    elif args.demo == 'full':
        # Full featured demo
        args.pattern = 'income_stratified'
        args.shape = 'circle'
        args.networks = True
        args.geographic = True
        args.income = True
        args.hubs = True
        args.types = 3


def main():
    """Main function to run the advanced Schelling model."""
    args = parse_arguments()
    
    print("Initializing Advanced Schelling Model...")
    print(f"Configuration: {args.pattern} pattern, {args.shape} shape")
    if args.networks:
        print("- Social networks enabled")
    if args.geographic:
        print("- Geographic features enabled")
    if args.income:
        print("- Income-based segregation enabled")
    if args.hubs:
        print("- Urban activity hubs enabled")
    
    # Create model
    model = AdvancedSchellingModel(
        width=args.width,
        height=args.height,
        density=args.density,
        global_homophily=args.homophily,
        num_agent_types=args.types,
        pattern_type=args.pattern,
        shape=args.shape,
        use_networks=args.networks,
        geographic_features=args.geographic,
        income_segregation=args.income,
        use_hubs=args.hubs
    )
    
    # Create visualizer
    visualizer = SchellingVisualizer(
        model=model,
        animation_interval=args.interval
    )
    
    # Run animation
    print("Starting visualization...")
    visualizer.run_animation()
    
    return model, visualizer


if __name__ == "__main__":
    main()
