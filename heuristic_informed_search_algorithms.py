import numpy as np
from single_agent_planner import compute_heuristics, a_star, in_map



def select_potential_targets_with_momentum(momentum_field_map, current_position, potential_targets):
    '''
    Calculate the best potential targets with momentum leading in a certain direction and choose those targets for consideration
    when finding next target to explore.
    '''
    best_target = None
    best_score = -np.inf
    current_momentum = momentum_field_map[current_position[0], current_position[1]]
    if np.linalg.norm(current_momentum) > 0:
        current_momentum = current_momentum / np.linalg.norm(current_momentum)  # Ensure normalization
    
    for target in potential_targets:
        # Calculate direction to target
        direction_to_target = np.array(target) - np.array(current_position)
        if np.linalg.norm(direction_to_target) > 0:
            direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
        
        # Score based on alignment with momentum
        score = np.dot(current_momentum, direction_to_target)  # Dot product measures alignment
        
        if score > best_score:
            best_score = score
            best_target = target
            
    return best_target

def calculate_repulsion_potential(x, y, agents, A=1, sigma=1):
    """
    Calculate the repulsion potential at a point (x, y) due to all agents.
    agents: List of tuples representing the (x, y) positions of each agent.
    A: Amplitude of the repulsion.
    sigma: Spread of the repulsion effect.
    """
    v_total = 0
    for x_a, y_a in agents:
        distance_squared = (x-x_a)**2 + (y-y_a)**2
        v = A*np.exp(-distance_squared/(2*sigma**2))
        v_total += v
    return v_total

def repel_agents_cost(start_loc, agent_paths):
    '''
    Definition: Repels agents to search away from each other by adding cost to search
    '''
    final_agents_pos = []
    #print(len(np.shape(agent_paths)))
    for agent_path in agent_paths:
        if len(np.shape(agent_path))>1:
            final_position = agent_path[-1]
            final_agents_pos.append(final_position)
        else:
            final_agents_pos.append(agent_path)
    #print(final_agents_pos)
    v_total = calculate_repulsion_potential(start_loc[0], start_loc[1], final_agents_pos)
    return v_total

def calculate_exploration_density(my_map, information_map, i, j, radius):
    '''
    Finds the density of the unexplored cells around the cell needing to be explored.
    '''
    unexplored_count = 0
    total_count = 0
    for change_i in range(-radius, radius + 1):
        for change_j in range(-radius, radius + 1):
            if (change_i**2 + change_j**2) <= radius**2:  # Within circular radius
                neighbor_i, neighbor_j = i + change_i, j + change_j
                neighbor_cell = (neighbor_i, neighbor_j)
                # Ensure the cell is within the bounds of the map
                if in_map(my_map, neighbor_cell):
                    total_count += 1
                    # Check if the cell is unexplored, using a threshold (assumes unexplored < 0.5)
                    if information_map[neighbor_i][neighbor_j] < 0.5:
                        unexplored_count += 1

    density = unexplored_count / total_count if total_count else 0
    return density
                
def generate_potential_target_goals(start_loc, direction_vector, num_points, step_size):
    potential_targets = []
    for i in range(num_points):
        bias_direction = np.array(direction_vector)*step_size
        new_point = np.array(start_loc) + bias_direction +np.random.randn(2)*step_size
        potential_targets.append(tuple(new_point))
    return potential_targets

def select_target(my_map, information_map, start_loc, h_values, agent, constraints, agent_paths):
    '''
    Definition: Main target selection algorithm which selects best targets based on information gain and other heurisitic categories
    '''
    #print("here1")
    best_combined_heuristic = 0
    best_target = None
    best_path = None
    radius = 3
    max_radius = 10
    found_solution = False
    while not found_solution and radius <= max_radius:
        for i in range(len(my_map)):
            for j in range(len(my_map[0])):
                goal_loc = (i,j)
                #if information_map[goal_loc[0]][goal_loc[1]] == 0:
                 #   continue
                if (not in_map(my_map, goal_loc)) or (my_map[goal_loc[0]][goal_loc[1]]):
                    continue
                if np.sqrt((start_loc[0]-i)**2 + (start_loc[1]-j)**2) >= radius:
                    continue
                current_h_values = compute_heuristics(my_map, goal_loc)
                #print(goal_loc)
                #print(start_loc)
                path = a_star(my_map, start_loc, goal_loc, current_h_values, agent, constraints, information_map)
                if path is None:
                    continue
                # calculate total information gain
                information_gain = 0
                for loc in path:
                    information_gain += information_map[loc[0]][loc[1]]
                exploration_density = calculate_exploration_density(my_map, information_map, i, j, radius=5)
                # update best target
                repulsion_potential = repel_agents_cost(start_loc, agent_paths)
                combined_heuristic = information_gain+exploration_density-repulsion_potential
                if combined_heuristic > best_combined_heuristic:
                    best_combined_heuristic = combined_heuristic
                    best_target = (i,j)
                    best_path = path
                    found_solution = True

        if not found_solution:
            radius += 1
    #print("here2")
    return best_target, best_path