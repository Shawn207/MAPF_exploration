import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from heuristic_informed_search_algorithms import select_target
import copy
import numpy as np

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

def initialize_momentum_field_map(map_size):
        # Create a momentum field map initialized with zero vectors
        momentum_field_map = np.zeros((map_size[0], map_size[1], 2))  # Assuming 2D map, store (dx, dy) at each cell
        return momentum_field_map


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.num_of_agents = len(self.starts)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []
        self.information_map = copy.deepcopy(my_map)
        self.map_size = (len(my_map), len(my_map[0]))
        self.momentum_map = initialize_momentum_field_map(self.map_size)

        for i in range(len(self.information_map)):
            for j in range(len(self.information_map[0])):
                self.information_map[i][j] = 1 - self.information_map[i][j]


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
                self.information_map[loc[0]][loc[1]] = 0  # Mark as explored
    
    def update_momentum_field_map(self, paths):
        """
        Update the momentum field map with paths from multiple agents.
        :param momentum_field_map: A 3D numpy array where the first two dimensions correspond to spatial coordinates
                                and the third dimension stores a 2D vector.
        :param paths: A list of paths, each path is a list of tuples (x, y) coordinates.
        """
        for path in paths:
            for i in range(len(path) - 1):
                current_pos = path[i]
                next_pos = path[i + 1]
                
                # Calculate the movement vector
                movement_vector = np.array(next_pos) - np.array(current_pos)
                
                # Normalize the movement vector
                if np.linalg.norm(movement_vector) > 0:
                    movement_vector = movement_vector / np.linalg.norm(movement_vector)
                
                # Update the momentum field map at the current position
                if 0 <= current_pos[0] < self.momentum_map.shape[0] and 0 <= current_pos[1] < self.momentum_map.shape[1]:
                    self.momentum_map[current_pos[0], current_pos[1]] += movement_vector

        # Optionally normalize the vectors in the momentum map
        for i in range(self.momentum_map.shape[0]):
            for j in range(self.momentum_map.shape[1]):
                if np.linalg.norm(self.momentum_map[i, j]) > 0:
                    self.momentum_map[i, j] = self.momentum_map[i, j] / np.linalg.norm(self.momentum_map[i, j])

    def is_fully_explored(self):
        return all(cell == 0 for row in self.information_map for cell in row)

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """
        #print("here1")
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
        #print(self.starts)
        for i in range(self.num_of_agents):  # Find initial path for each agent
            target, path = select_target(self.my_map, self.information_map, self.starts[i], self.heuristics, i, [], self.starts)
            if path is None:
                return None, None
                raise BaseException('No solutions')
            root['paths'].append(path)
            root['targets'].append(target)

        #print("here")
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
                #print("here2")
                return current_node['paths'], current_node['targets']
   
            # choose the first collision and convert to a list of constraints
            collision = current_node['collisions'][0]
            #for collision in current_node['collisions']:
            constraints = standard_splitting(collision)
            # Add a new child node to the open list for each constraint
            # Create a child node for each new constraint
            for constraint in constraints:
                child_node = copy.deepcopy(current_node)
                child_node['constraints'].append(constraint)
                # Replan paths for all agents considering the new constraint
                agent_id = constraint['agent']  # This should correspond to the agent involved in the collision
                start_loc = self.starts[agent_id]
                new_target, new_path = select_target(self.my_map, self.information_map, start_loc, self.heuristics, agent_id, child_node['constraints'], current_node['paths'])
                print(new_target)
                if new_path is None:
                    # If a path can't be found for any agent, skip this child node
                    break
                # This else belongs to the for loop, executed only if the loop wasn't broken
                child_node['paths'][agent_id] = new_path
                child_node['targets'][agent_id] = new_target
                child_node['cost'] = get_sum_of_cost(child_node['paths'])
                child_node['collisions'] = detect_collisions_among_all_paths(child_node['paths'])
                self.push_node(child_node)

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
        self.update_momentum_field_map(initial_paths)

        # Continue exploration until all areas are explored
        while not self.is_fully_explored():

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
            self.update_momentum_field_map(new_paths)
        return final_paths


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
