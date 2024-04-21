import heapq
import numpy as np

node_count = 0
def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0,0)]
    # directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def move_joint_state(locs, dir):
    # import pdb; pdb.set_trace()
    new_locs = []
    for i in range(len(locs)):
        new_locs.append((locs[i][0] + dir[i][0], locs[i][1] + dir[i][1]))
        
    
    return new_locs

def generate_motions_recursive(num_agents,cur_agent):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    
    if cur_agent == num_agents:
        return [()]
    
    joint_state_motions = []

    for direction in directions:
        for subsequent_motion in generate_motions_recursive(num_agents, cur_agent + 1):
            joint_state_motions.append((direction,) + subsequent_motion)


    return joint_state_motions


def is_valid_motion(old_loc, new_loc):
    ##############################
    # Task 1.3/1.4: Check if a move from old_loc to new_loc is valid
    # Check if two agents are in the same location (vertex collision)
    # TODO
    # if any locs in new loc are the same
    if len(set(new_loc)) < len(new_loc):
        return False

    # Check edge collision
    # TODO
    # if any pair of agent swap locations
    for i in range(len(new_loc)):
        for j in range(i+1, len(new_loc)):
            if old_loc[i] == new_loc[j] and old_loc[j] == new_loc[i]:
                return False

    return True

def get_sum_of_cost(paths):
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3/1.4: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    
    table = {}
    # if constraints['time_step'] not in table.keys():
    #     table[constraints['time_step']] = [constraints['loc']]
    # else:
    #     table[constraints['time_step']].append(constraints['loc'])
    
    # assume only one constraint for each time step
    for constraint in constraints:
        if agent == constraint['agent']:
            if constraint['time_step'] not in table.keys():
                table[constraint['time_step']] = [constraint['loc']]
            else:
                table[constraint['time_step']].append(constraint['loc'])
        #     table[constraint['time_step']] = constraint['loc']
    

    return table



def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3/1.4: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    # vertex constraint
    # print(curr_loc, next_loc, next_time)
    if next_time in constraint_table.keys():
        # vertex constraint
        if next_loc in constraint_table[next_time]:
            return True
        # edge constraint
        if [curr_loc, next_loc] in constraint_table[next_time] or [next_loc,curr_loc] in constraint_table[next_time]:
        # if [curr_loc, next_loc] in constraint_table[next_time]:
            return True
    
    return False


def push_node(open_list, node):
    global node_count
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node_count, node))
    node_count += 1


def pop_node(open_list):
    _, _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def in_map(map, loc):
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or min(loc) < 0:
        return False
    else:
        return True

def all_in_map(map, locs):
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True

def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, information_map):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.2/1.3/1.4: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.
    # print(agent)
    visited_penalty = 100
    Manhattan_distance = abs(start_loc[0] - goal_loc[0]) + abs(start_loc[1] - goal_loc[1])
    open_list = []
    closed_list = dict()
    closed_list_space = []
    earliest_goal_timestep = 0
    # print("start location: ", start_loc)
    h_value = h_values[start_loc]
    # constraints = [{'agent': 0, 'loc': (1, 5), 'time_step': 4}]
    constraints_table = build_constraint_table(constraints, agent)
    # print(constraints_table)
    # import pdb;pdb.set_trace()
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'time_stamp': 0}
    push_node(open_list, root)
    closed_list[(root['loc'], root['time_stamp'])] = root
    closed_list_space.append(root['loc'])
    repetative_step = 0
    while len(open_list) > 0:
        curr = pop_node(open_list)
        #############################
        # Task 2.2: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc:
            # goal location at this time step cannot collide with other path at this time step
            goal_valid = True
            time_stamp = curr['time_stamp']
            # print(time_stamp, constraints_table.keys())
            while time_stamp in constraints_table.keys():
                # print(goal_loc, constraints_table[time_stamp])
                if goal_loc in constraints_table[time_stamp]:
                    goal_valid = False
                    break
                time_stamp += 1
                
            # if any node collide with the goal in any future step, fail:
            # import pdb; pdb.set_trace()
            keys_list = list(constraints_table.keys())
            if keys_list:
                # print(keys_list, time_stamp)
                if keys_list[-1] > time_stamp:
                    for i in range(time_stamp+1, keys_list[-1]+1):
                        if i in constraints_table.keys():
                            if goal_loc in constraints_table[i]:
                                goal_valid = False
                                break

            if goal_valid:
                return get_path(curr)
            
        # terminal condition
        # if curr['loc'] in closed_list_space:
        #     repetative_step += 1
        #     if repetative_step > Manhattan_distance*300 and repetative_step > 5:
        #         # pass
        #         # print("repetative step: ", repetative_step)
        #         # print("Manhattan distance: ", Manhattan_distance)
        #         # print("execute repetative steps, no solution found")
        #         return None
        # else:
        #     repetative_step = 0
        if len(closed_list_space) > 0.8*len(my_map)*len(my_map[0]):
            return None
                
        # print("current location: ", curr['loc'], "time step: ", curr['time_stamp'])
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if not in_map(my_map, child_loc) or my_map[child_loc[0]][child_loc[1]]:
                continue
            if is_constrained(curr['loc'], child_loc, curr['time_stamp']+1 , constraints_table):
                continue
            visited_cost = 0
            if information_map[child_loc[0]][child_loc[1]] == 0:  # Check if the cell was visited
                visited_cost += visited_penalty  # Add penalty for visited cells
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1 + visited_cost,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'time_stamp': curr['time_stamp'] + 1}
            # print("child location: ", child['loc'], "time step: ", child['time_stamp'])
            if ((child['loc'], child['time_stamp'])) in closed_list:
                existing_node = closed_list[(child['loc'], child['time_stamp'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['time_stamp'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['time_stamp'])] = child
                if child['loc'] not in closed_list_space:
                    closed_list_space.append(child['loc'])
                push_node(open_list, child)

    return None  # Failed to find solutions

def select_target(my_map, information_map, start_loc, h_values, agent, constraints):
    best_information_gain = 0
    best_target = None
    best_path = None
    radius = 3
    max_radius = 10
    found_solution = False
    while not found_solution and radius <= max_radius:
        for i in range(len(my_map)):
            for j in range(len(my_map[0])):
                goal_loc = (i,j)
                if (not in_map(my_map, goal_loc)) or (my_map[goal_loc[0]][goal_loc[1]]):
                    continue
                if np.sqrt((start_loc[0]-i)**2 + (start_loc[1]-j)**2) >= radius:
                    continue
                # print("goal location: ", goal_loc)
                current_h_values = compute_heuristics(my_map, goal_loc)
                #print(goal_loc)
                
                path = a_star(my_map, start_loc, goal_loc, current_h_values, agent, constraints, information_map)
                if path is None:
                    continue
                # calculate total information gain
                information_gain = 0
                # print("path: ", path)
                for loc in path:
                    # print("path location: ", loc)
                    # print("get info gain: ", information_map[loc[0]][loc[1]])
                    information_gain += information_map[loc[0]][loc[1]]
                # update best target
                # print("expect information gain: ", information_gain)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_target = (i,j)
                    best_path = path
                    found_solution = True
        # print("select target: ", best_target, "radius: ", radius, "information gain: ", best_information_gain)
        if not found_solution:
            radius += 1
    if not found_solution:
        print("start location: ", start_loc)
        print("information map: ", information_map)
        import pdb;pdb.set_trace()
    return best_target, best_path

def joint_state_a_star(my_map, starts, goals, h_values, num_agents):
    """ my_map      - binary obstacle map
        start_loc   - start positions
        goal_loc    - goal positions
        num_agent   - total number of agents in fleet
    """

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = 0
     ##############################
    # Task 1.1: Iterate through starts and use list of h_values to calculate total h_value for root node
    #
    # TODO
    h_value = sum([h_values[i][starts[i]] for i in range(num_agents)])
    root = {'loc': starts, 'g_val': 0, 'h_val': h_value, 'parent': None }
    push_node(open_list, root)
    closed_list[tuple(root['loc'])] = root

     ##############################
    # Task 1.1:  Generate set of all possible motions in joint state space
    #
    # TODO
    directions = generate_motions_recursive(num_agents,0)
    while len(open_list) > 0:
        curr = pop_node(open_list)
        
        if curr['loc'] == goals:
            print("reach goal")
            print("curr location: ", curr['loc'])
            print("curr h value: ", curr['h_val'])
            print("curr g value: ", curr['g_val'])
            
            return get_path(curr)
        
        for dir in directions:
            # import pdb;pdb.set_trace()
            # print("generate child node")
            ##############################
            # Task 1.1:  Update position of each agent
            #
            # TODO
            child_loc = move_joint_state(curr['loc'], dir)
            
            if not all_in_map(my_map, child_loc):
                continue
             ##############################
            # Task 1.1:  Check if any agent is in an obstacle
            #
            valid_move = True
            # TODO
            for i in range(num_agents):
                # map[x][y] is True if there is an obstacle at (x,y)
                if my_map[child_loc[i][0]][child_loc[i][1]]:
                    valid_move = False
                    break
                
            if not valid_move:
                continue

             ##############################
            # Task 1.1:   check for collisions
            #
            # TODO
            if not is_valid_motion(curr['loc'],child_loc):
                continue
            
             ##############################
            # Task 1.1:  Calculate heuristic value
            #
            # TODO
            h_value = sum([h_values[i][(child_loc[i][0],child_loc[i][1])] for i in range(num_agents)])
            # print("h value: ", h_value)
            # for i in range(num_agents):
            #     print(h_values[i][(child_loc[i][0],child_loc[i][1])])
            # import pdb;pdb.set_trace()
            
            # calculate g value
            g_val = curr['g_val']
            for (i,move) in enumerate(dir):
                if child_loc[i] != goals[i]:
                    g_val += 1
                    
            

            # g_val += num_agents
            # if g_val-curr['g_val'] != num_agents:
                # import pdb;pdb.set_trace()
            
            # g_val = curr['g_val'] + num_agents
            # print("g value: ", g_val)
            # Create child node
            child = {'loc': child_loc,
                    'g_val': g_val,
                    'h_val': h_value,
                    'parent': curr}
            if (tuple(child['loc'])) in closed_list:
                existing_node = closed_list[tuple(child['loc'])]
                if compare_nodes(child, existing_node):
                    closed_list[tuple(child['loc'])] = child
                    push_node(open_list, child)
            else:
                closed_list[tuple(child['loc'])] = child
                push_node(open_list, child)
            # print(child)
        #     print("child cost: ", child['h_val'] + child['g_val'])
        # print("current node")
        print(curr)
        print("curr cost: ", curr['h_val'] + curr['g_val'])
        # print("curr location: ", curr['loc'])
        # print("curr h value: ", curr['h_val'])
        # print("curr g value: ", curr['g_val'])
        
        # import pdb;pdb.set_trace()
    return None  # Failed to find solutions

if __name__ == "__main__":
    # test generate_motions_recursive
    # print(generate_motions_recursive(3, 0))
    # print(len(generate_motions_recursive(3, 0)))
    
    # test move_join_state
    # print(move_joint_state([(0, 0), (0, 1), (0, 2)], 3))
    pass