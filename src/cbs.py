import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost, select_target
import copy

def detect_first_collision_for_path_pair(path1, path2):
    ##############################
    # Task 2.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    # determine longer path
    longer_path = None
    shorter_path = None
    if len(path1) > len(path2):
        longer_path = path1
        shorter_path = path2
    else:
        longer_path = path2
        shorter_path = path1
    
    time_step = 0
    while time_step < len(longer_path):
        # vertex collision
        if time_step < len(shorter_path):
            if get_location(path1, time_step) == get_location(path2, time_step):
                return {'location': get_location(path1, time_step), 'time_stamp': time_step}
        else:
            if get_location(longer_path, time_step) == get_location(shorter_path, len(shorter_path) - 1):
                return {'location': get_location(longer_path, time_step), 'time_stamp': time_step}
            
        # edge collision
        if get_location(path1, time_step) == get_location(path2, time_step - 1) and get_location(path1, time_step - 1) == get_location(path2, time_step):
            return {'location': [get_location(path1, time_step), get_location(path1, time_step - 1)], 'time_stamp': time_step}
        
        time_step += 1
    
    return None


def detect_collisions_among_all_paths(paths):
    ##############################
    # Task 2.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    collisions = []
    for i in range(len(paths) - 1):
        for j in range(i + 1, len(paths)):
            collision = detect_first_collision_for_path_pair(paths[i], paths[j])
            if collision is not None:
                collision['a1'] = i
                collision['a2'] = j
                collisions.append(collision)
    
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 2.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    constraints = []
    constraints.append({'agent': collision['a1'], 'loc': collision['location'], 'time_step': collision['time_stamp']})
    constraints.append({'agent': collision['a2'], 'loc': collision['location'], 'time_step': collision['time_stamp']})
    
    return constraints




class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, information_map_copy):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        information_map_copy - copy of current temperary infomration gain used for global planning
        """

        self.my_map = my_map
        self.starts = starts
        self.num_of_agents = len(self.starts)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []
        self.information_map_copy = information_map_copy

        # compute heuristics for the low-level search
        self.heuristics = []

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node
    
    def update_information_map(self, paths):
        # Update the information map based on the paths
        for path in paths:
            for loc in path:
                self.information_map_copy[loc[0]][loc[1]] = 0  # Mark as explored

    def is_fully_explored(self):
        return all(cell == 0 for row in self.information_map_copy for cell in row)

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'targets': [],
                'collisions': []}
        # print("information_map_copy: ", self.information_map_copy)
        for i in range(self.num_of_agents):  # Find initial path for each agent
            target, path = select_target(self.my_map, self.information_map_copy, self.starts[i], self.heuristics, i, [])
            if target in root['targets']:
                print("target already in root['targets']")
                self.information_map_copy[target[0]][target[1]] = 0
                target, path = select_target(self.my_map, self.information_map_copy, self.starts[i], self.heuristics, i, [])
            print("found target for agent ", i, " at ", target)
            if path is None:
                return None, None
                raise BaseException('No solutions')
            root['paths'].append(path)
            root['targets'].append(target)

        print("here")
        print("targets: ", root['targets'])
        print("paths: ", root['paths'])
        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'])
        self.push_node(root)

        # Task 2.1: Testing
        # print(root['collisions'])

        # Task 2.2: Testing
        # for collision in root['collisions']:
        #     print(standard_splitting(collision))

        ##############################
        # Task 2.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
        while self.open_list:
            # print("pop a node")
            current_node = self.pop_node()
            # print(curr)
            # print(curr['cost'])
            # return solution if this node has no collision
            if not current_node['collisions']:
                # import pdb;pdb.set_trace()
                return current_node['paths'], current_node['targets']
            # print("curent node: ", current_node)
            # choose the first collision and convert to a list of constraints
            collision = current_node['collisions'][0]
            #for collision in current_node['collisions']:
            constraints = standard_splitting(collision)
            # Add a new child node to the open list for each constraint
            for constraint in constraints:
                # print("Add a new child node to the open list for each constraint")
                new_node = copy.deepcopy(current_node)
                if constraint not in new_node['constraints']:
                    new_node['constraints'].append(constraint)
                    agent_index = constraint['agent']
                    start_loc = self.starts[agent_index]
                    new_target, new_path = select_target(self.my_map, self.information_map_copy, start_loc, self.heuristics, agent_index, new_node['constraints'])
                if new_path is None:
                    continue
                new_node['paths'][constraint['agent']] = new_path
                new_node['targets'][constraint['agent']] = new_target
                new_node['cost'] = get_sum_of_cost(new_node['paths'])
                new_node['collisions'] = detect_collisions_among_all_paths(new_node['paths'])
                # print(constraint)
                # print(new_path)
                # print(new_node['collisions'])
                self.push_node(new_node)
            # import pdb;pdb.set_trace()

    def explore_environment(self):
        # Initialize exploration by running CBS once to get initial paths
        final_paths = [[] for _ in range(self.num_of_agents)]
        initial_paths, next_start = self.find_solution()
        if initial_paths is None:
            raise Exception("No initial exploration paths found.")
        for i, path in enumerate(initial_paths):
            final_paths[i].extend(path)  # Accumulate initial paths
        # Update information map based on initial exploration
        self.update_information_map(initial_paths)

        # Continue exploration until all areas are explored
        # while not self.is_fully_explored():

        # Run CBS with the new targets to resolve conflicts and get new paths
        self.starts = next_start
        self.open_list = []
        self.heuristics = []
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        print(self.starts)
        #print("here")
        new_paths, next_start = self.find_solution()
        if new_paths is None:
            return final_paths
            raise Exception("No further exploration paths found.")
        for i, path in enumerate(new_paths):
            final_paths[i].extend(path[1:])  # Skip the first position as it's the last of the previous path
        # Update information map based on the new exploration
        self.update_information_map(new_paths)
        return final_paths


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
